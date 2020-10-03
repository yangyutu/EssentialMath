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
SMALL_SIZE = 25
MEDIUM_SIZE = 30
BIGGER_SIZE = 35
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
sns.set_style(style='white')
latent_dim = 2
inputs_decoder = 49
data_name = 'mnist'

if data_name not in ['fashion_mnist', 'mnist']:
    print('data_name should be fashion_mnist or mnist')

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        activation = tf.nn.relu
        self.dense1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        self.dense2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        self.dense3 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        self.flatten = tf.keras.layers.Flatten()
        self.mean = tf.keras.layers.Dense(latent_dim)
        self.std_dev = tf.keras.layers.Dense(latent_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.flatten(x)
        
        mean = self.mean(x)
        std_dev = tf.nn.softplus(self.std_dev(x))
        log_var = self.std_dev(x)
        # Reparametrization trick
        epsilon = tf.random.normal(tf.stack([tf.shape(x)[0], latent_dim]), name='epsilon')
        z = mean + tf.multiply(epsilon, std_dev)
        #z = mean + tf.exp(0.5 * log_var) * epsilon
        return mean, std_dev, z
    
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

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_std_dev, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + 2.0 * tf.math.log(z_std_dev) - tf.square(z_mean) - tf.square(z_std_dev)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }



# Link encoder and decoder
encoder = Encoder()
decoder = Decoder()
#mean_, std_dev, z = encoder(testData)
#output = decoder(z)

         
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data() if data_name == 'mnist' else keras.datasets.fashion_mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255 # last dim is channel

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)

"""
## Display a grid of sampled digits
"""



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
    plt.savefig(data_name + '_LatentV1.png', dpi=300)
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
    plt.savefig(data_name + '_LabelClusterV1.png', dpi=300)
    plt.show()


(x_train, y_train), _ = keras.datasets.mnist.load_data() if data_name == 'mnist' else keras.datasets.fashion_mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(encoder, decoder, x_train, y_train)