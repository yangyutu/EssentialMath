# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 01:50:10 2020

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

sns.set_style(style='white')

# Network parameters
tf.flags.DEFINE_float('learning_rate', .0005, 'Initial learning rate.')
tf.flags.DEFINE_integer('epochs', 20, 'Number of steps to run trainer.')
tf.flags.DEFINE_integer('batch_size', 128, 'Minibatch size')
tf.flags.DEFINE_integer('latent_dim', 2, 'Number of latent dimensions')
tf.flags.DEFINE_integer('test_image_number', 5, 'Number of test images to recover during training')
tf.flags.DEFINE_integer('inputs_decoder', 49, 'Size of decoder input layer')
tf.flags.DEFINE_string('dataset', 'mnist', 'Dataset name [mnist, fashion-mnist]')
tf.flags.DEFINE_string('logdir', './logs', 'Logs folder')
tf.flags.DEFINE_bool('plot_latent', True, 'Plot latent space')

FLAGS = tf.flags.FLAGS

# Define and create results folders
results_folder = os.path.join('Results', FLAGS.dataset)
[os.makedirs(os.path.join(results_folder, folder)) for folder in ['Test', 'Train']
    if not os.path.exists(os.path.join(results_folder, folder))]

# Empty log folder
try:
    if not len(os.listdir(FLAGS.logdir)) == 0:
        shutil.rmtree(FLAGS.logdir)
except:
    pass

# Get data
data = keras.datasets.mnist if FLAGS.dataset == 'mnist' else keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Create tf dataset
with tf.variable_scope("DataPipe"):
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.map(lambda x: tf.image.convert_image_dtype([x], dtype=tf.float32))
    dataset = dataset.batch(batch_size=FLAGS.batch_size).prefetch(FLAGS.batch_size)

    iterator = dataset.make_initializable_iterator()
    input_batch = iterator.get_next()
    input_batch = tf.reshape(input_batch, shape=[-1, 28, 28, 1])


def encoder(X):
    activation = tf.nn.relu
    with tf.variable_scope("Encoder"):
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.flatten(x)

        # Local latent variables
        mean_ = tf.layers.dense(x, units=FLAGS.latent_dim, name='mean')
        std_dev = tf.nn.softplus(tf.layers.dense(x, units=FLAGS.latent_dim), name='std_dev')  # softplus to force >0

        # Reparametrization trick
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], FLAGS.latent_dim]), name='epsilon')
        z = mean_ + tf.multiply(epsilon, std_dev)

        return z, mean_, std_dev


def decoder(z):
    activation = tf.nn.relu
    with tf.variable_scope("Decoder"):
        x = tf.layers.dense(z, units=FLAGS.inputs_decoder, activation=activation)
        x = tf.layers.dense(x, units=FLAGS.inputs_decoder, activation=activation)
        recovered_size = int(np.sqrt(FLAGS.inputs_decoder))
        x = tf.reshape(x, [-1, recovered_size, recovered_size, 1])

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=None)

        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28, 1])
        return img


# Link encoder and decoder
z, mean_, std_dev = encoder(input_batch)
output = decoder(z)

# Reshape input and output to flat vectors
flat_output = tf.reshape(output, [-1, 28 * 28])
flat_input = tf.reshape(input_batch, [-1, 28 * 28])

with tf.name_scope('loss'):
    img_loss = tf.reduce_sum(flat_input * -tf.log(flat_output) + (1 - flat_input) * -tf.log(1 - flat_output), 1)
    latent_loss = 0.5 * tf.reduce_sum(tf.square(mean_) + tf.square(std_dev) - tf.log(tf.square(std_dev)) - 1, 1)
    loss = tf.reduce_mean(img_loss + latent_loss)
    tf.summary.scalar('batch_loss', loss)

optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)


init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
gpu_options = tf.GPUOptions(allow_growth=True)

# Training loop
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(init_vars)
    merged_summary_op = tf.summary.merge_all()

    for epoch in range(FLAGS.epochs):
        sess.run(iterator.initializer)
        print('Actual epoch: {}'.format(epoch))

        flag = True  # Show only first batch of epoch

        while True:
            try:
                sess.run(optimizer)
                if flag:
                    # Get input and recover output images comparison
                    summ, target, output_ = sess.run([merged_summary_op, input_batch, output])
                    f, axarr = plt.subplots(FLAGS.test_image_number, 2)

                    for j in range(FLAGS.test_image_number):
                        for pos, im in enumerate([target, output_]):
                            axarr[j, pos].imshow(im[j].reshape((28, 28)), cmap='gray')
                            axarr[j, pos].axis('off')

                    plt.savefig(os.path.join(results_folder, 'Train/Epoch_{}').format(epoch))
                    plt.close(f)
                    flag = False
                    writer.add_summary(summ, epoch)

                    # Create artificial image from unit norm sample
                    artificial_image = sess.run(output, feed_dict={z: np.random.normal(0, 1, (1, FLAGS.latent_dim))})
                    plt.figure()
                    with sns.axes_style("white"):
                        plt.imshow(artificial_image[0].reshape((28, 28)), cmap='gray')
                    plt.savefig(os.path.join(results_folder, 'Test/{}'.format(epoch)))
                    plt.close()

                    # Create plot of latent space (only if latent dimensions are 2)
                    if FLAGS.latent_dim == 2 and FLAGS.plot_latent:
                        coords = sess.run(z, feed_dict={input_batch: test_images[..., np.newaxis]/255.})
                        colormap = ListedColormap(sns.color_palette(sns.hls_palette(10, l=.45 , s=.8)).as_hex())
                        plt.scatter(coords[:, 0], coords[:, 1], c=test_labels, cmap=colormap)

                        cbar = plt.colorbar()
                        if FLAGS.dataset == 'fashion-mnist':
                            cbar.ax.set_yticklabels(['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                                                     'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

                        # plt.axis('off')
                        plt.title('Latent space')
                        plt.savefig(os.path.join(results_folder, 'Test/Latent_{}'.format(epoch)))
                        plt.close()

            except tf.errors.OutOfRangeError:
                break

        # Create mesh grid of values
        values = np.arange(-3, 4, .5)
        xx, yy = np.meshgrid(values, values)
        input_holder = np.zeros((1, 2))
        # Matrix that will contain the grid of images
        container = np.zeros((28 * len(values), 28 * len(values)))

        for row in range(xx.shape[0]):
            for col in range(xx.shape[1]):
                input_holder[0, :] = [xx[row, col], yy[row, col]]
                artificial_image = sess.run(output, feed_dict={z: input_holder})
                container[row * 28: (row + 1) * 28, col * 28: (col + 1) * 28] = np.squeeze(artificial_image)

        plt.imshow(container, cmap='gray')
        plt.savefig(os.path.join(results_folder, 'Test/Space_{}'.format(epoch)))
        plt.close(  )