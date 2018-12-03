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

def get_batch(Xdata_sub, Ydata_sub):
    n = Xdata_sub.shape[0]
    batch_size = 128
    batch = np.floor(np.random.rand(batch_size) * n).astype(int)
    batch_x = Xdata_sub[batch, :]
    batch_y = Ydata_sub[batch]
    return[batch_x, batch_y]

# This is being done for each element of y
# but really can be done with only the number of classes and then mapped
def haltonize(data, dim):
    base = [17., 61.] # This will need to be generalized to higher dim
    skip = [43, 409] # This will need to be generalized to higher dim
    halton_data = np.empty([len(data), dim])
    temp = np.empty([1, dim])
    for x in range(len(data)):
        for y in range(dim):
            f = 1.0
            r = 0.0
            seed = (data[x] + 1) * skip[y]
            while seed > 0.:
                f = f/base[y]
                r = r + f*(np.mod(seed, base[y]))
                seed = np.floor(seed/base[y])
            temp[0, y] = r
        halton_data[x,:] = temp*100.
    return halton_data
# prepare data and tf.session
classes=5
outputs=2
mnist = input_data.read_data_sets('data/fashion', one_hot=False)

Xdata_sub = np.array([x for (x,y) in zip(mnist.train.images,mnist.train.labels) if y < classes])
Ydata_sub = np.array([y for y in mnist.train.labels if y < classes])
Ydata_sub = haltonize(Ydata_sub, outputs)

sess = tf.InteractiveSession()

# setup siamese network
network = model.siamese([1024, 1024, outputs])
#s1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer1')
#s2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer2')

#saver = tf.train.Saver(s1, s2, max_to_keep=15)
mod = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver = tf.train.Saver(mod, max_to_keep=15)
tf.initialize_all_variables().run()

if tf.train.checkpoint_exists("./halton_model/Final"):
    print("Model exists")
    saver.restore(sess, './halton_model/Final')# Sloppy and dangerous
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
N = 1000#150000
for step in range(N):
    long_x, batch_y = get_batch(Xdata_sub, Ydata_sub)

    batch_x = long_x.reshape(len(long_x), image_size, image_size, 1)


    _, loss_v = sess.run([train_step, network.loss], feed_dict={
                        network.x: batch_x,
                        network.y_: batch_y})

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()
    if step % 10 == 0:
        [loss_sum] = sess.run([network.acc], feed_dict={
            network.x: batch_x,
            network.y_: batch_y})
        writer.add_summary(loss_sum, step)
    """
    if step % 3 == 0:
        [sum1] = sess.run(network.acc, feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y_: batch_y})
        writer.add_summary(sum1, step)
    """

    if step % 1000 == 0:
        saver.save(sess, './halton_model/mod', global_step = step)
        iv_long = np.array([x for (x,y) in zip(mnist.test.images,mnist.test.labels) if y < classes + 1])
        y_test = np.array([y for y in mnist.test.labels if y < classes + 1])
        image_vector = iv_long.reshape([len(iv_long), image_size, image_size, 1])
        #image_vector = mnist.test.images.reshape(len(mnist.test.images), image_size, image_size, 1)
        iv_long = iv_long[:1000,:]
        image_vector = image_vector[:1000,:,:,:]
        y_test = y_test[:1000]
        embed = network.o1.eval({network.x: image_vector})
        embed.tofile('embed.txt')
        image_vector = image_vector.reshape(1000, image_size, image_size)
        visualize.save(embed, image_vector, y_test, step)

writer.close()

# visualize result

x_long = np.array([x for (x,y) in zip(mnist.test.images, mnist.test.labels) if y < classes + 1])
y_test = np.array([y for y in mnist.test.labels if y < classes + 1])
x_test = x_long.reshape([len(y_test), image_size, image_size])
visualize.save(embed, x_test, y_test, N)
#visualize.visualize(embed, x_test, y_test)

#Final save
saver.save(sess, './halton_model/Final')