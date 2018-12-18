from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import random

image_size = 28

#import helpers
import model
import visualize
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
classes=7
mnist = input_data.read_data_sets('data/fashion', one_hot=False)

Xdata_binary = np.array([x for (x,y) in zip(mnist.train.images,mnist.train.labels) if y < classes])
Ydata_binary = np.array([y for y in mnist.train.labels if y < classes])

sess = tf.InteractiveSession()

# setup siamese network
network = model.siamese([1024, 1024, 2])
s1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer1')
s2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer2')

mod = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver = tf.train.Saver(mod, max_to_keep=15)
tf.initialize_all_variables().run()

if tf.train.checkpoint_exists("./model/Final"):
    print("Model exists")
    response = input("Load saved model? (Y/n)")
    if (response == 'Y') or (response == 'y'):
        saver.restore(sess, './model/Final')# Sloppy and dangerous
else:
    print("Model not found")

iv_long = np.array([x for (x,y) in zip(mnist.test.images,mnist.test.labels) if y < classes])
y_test = np.array([y for y in mnist.test.labels if y < classes])
image_vector = iv_long.reshape([len(iv_long), image_size, image_size, 1])
iv_long = iv_long[:1000,:]
image_vector = image_vector[:1000,:,:,:]
y_test = y_test[:1000]
embed = network.o1.eval({network.x1: image_vector})
embed.tofile('embed.txt')
image_vector = image_vector.reshape(1000, image_size, image_size)
visualize.save(embed, image_vector, y_test, 19)

