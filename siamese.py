from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np
import os
import model
# Suppress warnings
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
sess = tf.InteractiveSession()

network = model.network([100, 2])
train_step = tf.train.GradientDescentOptimizer(0.1).minimize()
#saver = tf.train.Saver()
tf.initialize_all_variables().run()

for step in range(100):#(50000):
    batch_x1, batch_y1 = mnist.train.next_batch(128)
    batch_x2, batch_y2 = mnist.train.next_batch(128)
    #batch_y = (batch_y1 == batch_y2).astype('float')

    _, loss_v = sess.run([train_step, network.loss], feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y1: batch_y1,
                        network.y2: batch_y2})

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()

    if step % 10 == 0:
        print ('step %d: loss %.3f' % (step, loss_v))