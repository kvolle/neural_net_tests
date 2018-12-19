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

def get_paired_batches(xdata, ydata):
    percentage_matching = 50
    [batch1_x, batch1_y] = get_batch(xdata, ydata)
    [raw2_x, raw2_y] = get_batch(xdata, ydata)
    max_y = max(raw2_y)

    separated =[[] for k in range(max_y+1)]
    for (x, y) in zip(raw2_x, raw2_y):
        separated[y].append(x)
    batch2_x = []
    batch2_y = []
    for (x, y) in zip(batch1_x, batch1_y):
        if len(batch2_y) != len(batch2_x):
            print("test")
        if np.random.random_integers(0, 99, 1) < percentage_matching:
            # get one from the corresponding separated
            batch2_x.append(separated[y][0])
            batch2_y.append(y)
            if len(separated[y]) > 1:
                del separated[y][0]
        else:
            ind = y
            while (ind == y):
                ind = np.random.random_integers(0, max_y)
            # get one from another section of separated
            batch2_x.append(separated[ind][0])
            batch2_y.append(ind)
            if len(separated[ind]) > 1:
                del separated[ind][0]
    batch2_x = np.asarray(batch2_x)
    batch2_y = np.asarray(batch2_y)
    return [batch1_x, batch1_y, batch2_x, batch2_y]

# prepare data and tf.session
classes=5
mnist = input_data.read_data_sets('data/fashion', one_hot=False)

Xdata_binary = np.array([x for (x,y) in zip(mnist.train.images,mnist.train.labels) if y < classes])
Ydata_binary = np.array([y for y in mnist.train.labels if y < classes])

sess = tf.InteractiveSession()

# setup siamese network
network = model.siamese([1024, 1024, 2])
s1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer1')
s2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer2')

#saver = tf.train.Saver(s1, s2, max_to_keep=15)
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
N = 100000#150000
for step in range(N):
    #long_x1, batch_y1 = get_batch(Xdata_binary, Ydata_binary)
    #long_x2, batch_y2 = get_batch(Xdata_binary, Ydata_binary)
    [long_x1, batch_y1, long_x2, batch_y2] = get_paired_batches(Xdata_binary, Ydata_binary)
    batch_y = (batch_y1 == batch_y2)
    batch_x1 = long_x1.reshape(len(long_x1), image_size, image_size, 1)
    batch_x2 = long_x2.reshape(len(long_x1), image_size, image_size, 1)

    _, loss_v = sess.run([train_step, network.loss], feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y_: batch_y})

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()
    if step % 100 == 0:
        [loss_sum] = sess.run([network.acc], feed_dict={
            network.x1: batch_x1,
            network.x2: batch_x2,
            network.y_: batch_y})
        writer.add_summary(loss_sum, step)
    if step == 200000:
        train_step = tf.train.GradientDescentOptimizer(0.0002).minimize(network.loss, var_list=vars_to_train)
    """
    if step % 3 == 0:
        [sum1] = sess.run(network.acc, feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y_: batch_y})
        writer.add_summary(sum1, step)
    """

    if step % 2000 == 0:
        saver.save(sess, './model/mod', global_step = step)
        iv_long = np.array([x for (x,y) in zip(mnist.test.images,mnist.test.labels) if y < classes])
        y_test = np.array([y for y in mnist.test.labels if y < classes])
        image_vector = iv_long.reshape([len(iv_long), image_size, image_size, 1])
        #image_vector = mnist.test.images.reshape(len(mnist.test.images), image_size, image_size, 1)
        iv_long = iv_long[:1000,:]
        image_vector = image_vector[:1000,:,:,:]
        y_test = y_test[:1000]
        embed = network.o1.eval({network.x1: image_vector})
        embed.tofile('embed.txt')
        image_vector = image_vector.reshape(1000, image_size, image_size)
        visualize.save(embed, image_vector, y_test, step)

writer.close()

# visualize result

x_long = np.array([x for (x,y) in zip(mnist.test.images, mnist.test.labels) if y < classes])
y_test = np.array([y for y in mnist.test.labels if y < classes])
x_test = x_long.reshape([len(y_test), image_size, image_size])
visualize.save(embed, x_test, y_test, N)
saver.save(sess,'./model/Final')
#visualize.visualize(embed, x_test, y_test)
