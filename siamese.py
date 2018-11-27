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
def get_batch(Xdata_binary, Ydata_binary):
    n = Xdata_binary.shape[0]
    batch_size = 128
    batch = np.floor(np.random.rand(batch_size) * n).astype(int)
    batch_x = Xdata_binary[batch, :]
    batch_y = Ydata_binary[batch]
    return[batch_x, batch_y]
# prepare data and tf.session
classes_plus_1=3
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
#Xdata_binary = np.array([x for (x,y) in zip(mnist.train.images,mnist.train.labels) if y[9]==0])
#Ydata_binary = np.array([y[0:9] for y in mnist.train.labels if y[9]==0])
Xdata_binary = np.array([x for (x,y) in zip(mnist.train.images,mnist.train.labels) if y < classes_plus_1])
Ydata_binary = np.array([y for y in mnist.train.labels if y < classes_plus_1])

sess = tf.InteractiveSession()

# setup siamese network
network = model.siamese([1024, 1024, 2])
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(network.loss)
s1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer1')
s2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer2')

saver = tf.train.Saver(s1, s2)
tf.initialize_all_variables().run()
"""
testing = network.o1.eval({network.x1: mnist.test.images})
np.savetxt("labels_prior.csv", mnist.test.labels, delimiter=",")
np.savetxt("output_prior.csv", testing, delimiter=",")
"""

if tf.train.checkpoint_exists("./model/conv"):
    print("Model exists")
#    saver.restore(sess, "./model/conv")
else:
    print("Model not found")

vars = tf.trainable_variables()
vars_to_train=vars#[]
"""
for var in vars:
    if "siamese/layer1" not in var.name:
        if "siamese/layer2" not in var.name:
            vars_to_train.append(var)
            #print("Name: %s" % (var.name))
"""
train_step = tf.train.GradientDescentOptimizer(0.00002).minimize(network.loss,var_list=vars_to_train)

writer = tf.summary.FileWriter("log/Kyle/Classification/RR/Class/",sess.graph)
N = 100000
for step in range(N):
    long_x1, batch_y1 = get_batch(Xdata_binary, Ydata_binary)
    long_x2, batch_y2 = get_batch(Xdata_binary, Ydata_binary)
    batch_y = (batch_y1 == batch_y2)
    batch_x1 = long_x1.reshape(len(long_x1), image_size, image_size, 1)
    batch_x2 = long_x2.reshape(len(long_x1), image_size, image_size, 1)
    #if (step % 10) != 0:
    #    batch_x2 = noise(batch_x2)
    _, loss_v = sess.run([train_step, network.loss], feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y_: batch_y})

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()
#    if step % 600 == 0:
#        train_step = tf.train.GradientDescentOptimizer(0.0001*pow(2,step/600)).minimize(network.loss)
    if step % 100 == 0:
        print ('step %d: loss %.3f' % (step, loss_v))
    if step % 3 == 0:
        [sum1] = sess.run(network.acc, feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y_: batch_y})
        writer.add_summary(sum1, step)


    if (step + 1) % N == 0:
        #saver.save(sess, './model')
        iv_long = np.array([x for (x,y) in zip(mnist.test.images,mnist.test.labels) if y < classes_plus_1])
        image_vector = iv_long.reshape([len(iv_long), image_size, image_size, 1])
        #image_vector = mnist.test.images.reshape(len(mnist.test.images), image_size, image_size, 1)
        image_vector = image_vector[:1000,:,:,:]
        embed = network.o1.eval({network.x1: image_vector})
        embed.tofile('embed.txt')

#np.savetxt("labels.csv", mnist.test.labels, delimiter=",")
#np.savetxt("output.csv", embed, delimiter=",")
writer.close()

# visualize result
#x_test = mnist.test.images.reshape([-1, 28, 28])
#y_test = mnist.test.labels

x_long = np.array([x for (x,y) in zip(mnist.test.images,mnist.test.labels) if y < classes_plus_1])
y_test = np.array([y for y in mnist.test.labels if y < classes_plus_1])
x_test = x_long.reshape([len(y_test), image_size, image_size])
#y_test = np.array([y for y in mnist.test.labels if y<2])
#x_test = np.array([x for (x,y) in zip(mnist.test.images, mnist.test.labels) if y<2])
#x_test = x_test.reshape(len(y_test), image_size, image_size, 1)
visualize.visualize(embed, x_test, y_test)
