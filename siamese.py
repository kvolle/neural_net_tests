from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import random
import os

image_size = 28
# Suppress warnings
#old_v = tf.logging.get_verbosity()
#tf.logging.set_verbosity(tf.logging.ERROR)

#import helpers
import model
from siamese_tf_mnist import visualize
from skimage import transform

def noise(images):
    twisted = images
    for i in range(len(images)):
        angle = random.randint(0,360)
        twisted[i, :, :, :] = transform.rotate(images[i,:,:,:], angle)
    return twisted

# prepare data and tf.session
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
sess = tf.InteractiveSession()

# setup siamese network
network = model.siamese([1024, 1024, 2])
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(network.loss)

#saver = tf.train.Saver()
tf.initialize_all_variables().run()
"""
testing = network.o1.eval({network.x1: mnist.test.images})
np.savetxt("labels_prior.csv", mnist.test.labels, delimiter=",")
np.savetxt("output_prior.csv", testing, delimiter=",")
"""
writer = tf.summary.FileWriter("log/Kyle/",sess.graph)
for step in range(2000):
    long_x1, batch_y1 = mnist.train.next_batch(128)
    long_x2, batch_y2 = mnist.train.next_batch(128)
    batch_y = (batch_y1 == batch_y2)
    batch_x1 = long_x1.reshape(len(long_x1), image_size, image_size, 1)
    batch_x2 = long_x2.reshape(len(long_x1), image_size, image_size, 1)
    batch_x2 = noise(batch_x2)
    _, loss_v = sess.run([train_step, network.loss], feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y_: batch_y})

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()

    if step % 600 == 0:
        train_step = tf.train.GradientDescentOptimizer(0.0001*pow(2,step/600)).minimize(network.loss)
    if step % 10 == 0:
        print ('step %d: loss %.3f' % (step, loss_v))

    if (step + 1) % 2000 == 0:
        #saver.save(sess, './model')
        image_vector = mnist.test.images.reshape(len(mnist.test.images), image_size, image_size, 1)
        image_vector = image_vector[:1000,:,:,:]
        embed = network.o1.eval({network.x1: image_vector})
        embed.tofile('embed.txt')

#np.savetxt("labels.csv", mnist.test.labels, delimiter=",")
#np.savetxt("output.csv", embed, delimiter=",")
writer.close()

# visualize result
x_test = mnist.test.images.reshape([-1, 28, 28])
y_test = mnist.test.labels
visualize.visualize(embed, x_test, y_test)
