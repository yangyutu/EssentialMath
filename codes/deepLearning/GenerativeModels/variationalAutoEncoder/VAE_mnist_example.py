# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:49:42 2020

@author: yangy
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import seaborn as sns
from matplotlib.colors import ListedColormap
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
sns.set_style(style='white')

# Network parameters
#tf.flags.DEFINE_float('learning_rate', .0005, 'Initial learning rate.')
#tf.flags.DEFINE_integer('epochs', 20, 'Number of steps to run trainer.')
#tf.flags.DEFINE_integer('batch_size', 128, 'Minibatch size')
#tf.flags.DEFINE_integer('latent_dim', 2, 'Number of latent dimensions')
#tf.flags.DEFINE_integer('test_image_number', 5, 'Number of test images to recover during training')
#tf.flags.DEFINE_integer('inputs_decoder', 49, 'Size of decoder input layer')
#tf.flags.DEFINE_string('dataset', 'mnist', 'Dataset name [mnist, fashion-mnist]')
#tf.flags.DEFINE_string('logdir', './logs', 'Logs folder')
#tf.flags.DEFINE_bool('plot_latent', True, 'Plot latent space')
#
#FLAGS = tf.flags.FLAGS


learning_rate = 0.001
epochs = 30
batch_size = 128
latent_dim = 2
inputs_decoder = 49
data_name = 'mnist'

# Get data
# data = keras.datasets.mnist if data_name == 'mnist' else keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = data.load_data()
#
# # Create tf dataset
#
# dataset = tf.data.Dataset.from_tensor_slices(train_images)
# dataset = dataset.map(lambda x: tf.image.convert_image_dtype([x], dtype=tf.float32))
# dataset = dataset.batch(batch_size=batch_size).prefetch(batch_size)
#
# #iterator = dataset.make_initializable_iterator()
# #input_batch = iterator.get_next()
# #input_batch = tf.reshape(input_batch, shape=[-1, 28, 28, 1])
# dataset_iter = iter(dataset)
# testData = next(dataset_iter)


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        activation = tf.nn.relu
        self.dense1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same',
                                             activation=activation)
        self.dense2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same',
                                             activation=activation)
        self.dense3 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='same',
                                             activation=activation)
        self.flatten = tf.keras.layers.Flatten()
        self.mean = tf.keras.layers.Dense(latent_dim)
        self.std_dev = tf.keras.layers.Dense(latent_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.flatten(x)

        mean = self.mean(x)
        log_var = self.std_dev(x)
        # Reparametrization trick
        epsilon = tf.random.normal(tf.stack([tf.shape(x)[0], latent_dim]), name='epsilon')
        z = mean + tf.exp(0.5 * log_var) * epsilon
        return mean, log_var, z
    
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        activation = tf.nn.relu
        self.linear1 = tf.keras.layers.Dense(inputs_decoder, activation=activation)
        self.linear2 = tf.keras.layers.Dense(inputs_decoder, activation=activation)
        
        self.dense1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        self.dense2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        self.dense3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        self.flatten = tf.keras.layers.Flatten()
        self.linear3 = tf.keras.layers.Dense(28 * 28, activation=None)
        self.linear4 = tf.keras.layers.Dense(28 * 28, activation=tf.nn.sigmoid)


    def call(self, z):
        x = self.linear1(z)
        x = self.linear2(x)
        recovered_size = int(np.sqrt(inputs_decoder))
        x = tf.reshape(x, [-1, recovered_size, recovered_size, 1])
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        x = self.flatten(x)
        
        x = self.linear3(x)
        x = self.linear4(x)
        
        img = tf.reshape(x, shape=[-1, 28, 28, 1])
        return img        

# Link encoder and decoder
encoder = Encoder()
decoder = Decoder()


# define loss

def loss(data):
    z_mean, z_log_var, z = encoder(data)
    reconstruction = decoder(z)
    reconstruction_loss = tf.reduce_mean(
        keras.losses.binary_crossentropy(data, reconstruction)
    )
    reconstruction_loss *= 28 * 28
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss

def data_iter(features, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #shuffle
    for i in range(0, num_examples, batch_size):
        indexs = indices[i: min(i + batch_size, num_examples)]
        yield tf.gather(features,indexs)

(x_train, y_train), _ = keras.datasets.mnist.load_data() if data_name == 'mnist' else keras.datasets.fashion_mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
@tf.function
def train_step(opt, data):
    with tf.GradientTape() as tape:
        total_loss, reconstruction_loss, kl_loss = loss(data)
        # if count % 100 == 0:
        #     tf.print(total_loss, reconstruction_loss, kl_loss)
    trainable_weights = (encoder.trainable_weights + decoder.trainable_weights)
    grads = tape.gradient(total_loss, trainable_weights)
    opt.apply_gradients(zip(grads, trainable_weights))
    return total_loss, reconstruction_loss, kl_loss

def train_model(epochs):
    for epoch in tf.range(1,epochs+1):
        accum_loss = 0.0
        count = 0
        for features in data_iter(x_train, batch_size):
            total_loss, reconstruction_loss, kl_loss = train_step(opt, features)
            accum_loss += total_loss.numpy()
            count += 1

        tf.print("epoch =",epoch,"loss = ",accum_loss / count, "reconstruction_loss = ", reconstruction_loss, "kl_loss =", kl_loss)

train_model(epochs)

def plot_latent(encoder, decoder):
    # display a n*n 2D manifold of digits
    n = 20
    digit_size = 28
    scale = 2.0
    figsize = 15
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.axis('off')
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(data_name + '_Latent_customTrain.png', dpi=300)
    plt.show()


plot_latent(encoder, decoder)

"""
## Display how the latent space clusters different digit classes
"""


def plot_label_clusters(encoder, decoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(data_name + '_LabelCluster_customTrain.png', dpi=300)
    plt.show()


(x_train, y_train), _ = keras.datasets.mnist.load_data() if data_name == 'mnist' else keras.datasets.fashion_mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

#plot_label_clusters(encoder, decoder, x_train, y_train)