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

# prepare data and tf.session
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter('board_beginner')  # create writer
writer.add_graph(sess.graph)
# setup siamese network
network = model.siamese([1024, 1024, 2])
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(network.loss)
s1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer1')
s2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer2')

saver = tf.train.Saver(s1, s2)
tf.initialize_all_variables().run()
"""
testing = network.o1.eval({network.x1: mnist.test.images})
np.savetxt("labels_prior.csv", mnist.test.labels, delimiter=",")
np.savetxt("output_prior.csv", testing, delimiter=",")
"""
writer = tf.summary.FileWriter("log/Kyle/Classification/",sess.graph)
N = 100000
for step in range(N):
    long_x1, batch_y1 = mnist.train.next_batch(128)
    long_x2, batch_y2 = mnist.train.next_batch(128)
    batch_y = (batch_y1 == batch_y2)
    batch_x1 = long_x1.reshape(len(long_x1), image_size, image_size, 1)
    batch_x2 = long_x2.reshape(len(long_x1), image_size, image_size, 1)

    _, loss_v = sess.run([train_step, network.loss], feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y_: batch_y1})
    [acc1, acc2, acc3, acc4] = sess.run(network.acc, feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y_: batch_y1})
    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()
#    if step == 10:
#        train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(network.loss)
    if step % 100 == 0:
        print ('step %d: loss %.3f' % (step, loss_v))
        writer.add_summary(acc1, step)
        writer.add_summary(acc2, step)
        writer.add_summary(acc3, step)
        writer.add_summary(acc4, step)

    if (step + 1) % N == 0:
        saver.save(sess, './model/conv')
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
