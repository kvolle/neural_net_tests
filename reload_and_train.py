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

def get_batch(Xdata_binary, Ydata_binary):
    n = Xdata_binary.shape[0]
    batch_size = 128
    batch = np.floor(np.random.rand(batch_size) * n).astype(int)
    batch_x = Xdata_binary[batch, :]
    batch_y = Ydata_binary[batch]
    return[batch_x, batch_y]
def custom_loss():
    x = tf.constant(0., dtype=tf.float32)
    return tf.reduce_mean(x)
"""
    margin=5.0
    labels_t = tf.to_float(self.y_)
    labels_f = tf.subtract(1.0, labels_t, name="1-yi")
    distance2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
    distance2 = tf.reduce_sum(distance2, 1)
    distance = tf.sqrt(distance2 + 1e-6, name="Distance")
    same = tf.multiply(labels_t, distance2)
    margin_tensor = tf.constant(margin, dtype=tf.float32, name="Margin")
    diff = tf.multiply(labels_f, tf.pow(tf.maximum(0.0, tf.subtract(margin_tensor, distance)), 2.))
    loss = tf.reduce_mean(same)+tf.reduce_mean(diff)
    return loss
"""
# prepare data and tf.session
classes=5
outputs=2
N = 101#150000
mnist = input_data.read_data_sets('data/fashion', one_hot=False)

Xdata_binary = np.array([x for (x,y) in zip(mnist.train.images,mnist.train.labels) if y < classes])
Ydata_binary = np.array([y for y in mnist.train.labels if y < classes])



# setup siamese network
#network = model.siamese([1024, 1024, 2])
#s1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer1')
#s2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer2')
trained_graph = tf.Graph()
sess = tf.InteractiveSession(graph=trained_graph)
with trained_graph.as_default():
    default = tf.train.import_meta_graph('./halton_model/Final.meta', clear_devices=True)
    default.restore(sess, './halton_model/Final')# Sloppy and dangerous
    trained = tf.trainable_variables()

Siamese = tf.Graph()
sess2 = tf.Session(graph=Siamese)
halton = {}
for t in trained:
    halton[t.name] = t#t.value()
with Siamese.as_default():
    network = model.siamese([1024, 1024, outputs])
    writer = tf.summary.FileWriter("log/Reload/Trained/",sess2.graph)
    untrained = tf.trainable_variables()
    unassigned = {}
    for u in range(len(untrained)):
        unassigned[u] = untrained[u].name.split('/')
    tf.initialize_all_variables().run(session=sess2)
    # Go through the variables in the halton model
    # Look for two matches in untrained
    for name, value in halton.items():
        divided_name = name.split('/')
        matching_indices = []
        for index, siamese_name in unassigned.items():
            if len(siamese_name) == len(divided_name):
                # Must at least be the same length to match
                match_so_far = True
                for segment_halton, segment_siamese in zip(divided_name, siamese_name):
                    if segment_halton not in segment_siamese:
                        match_so_far = False
                if match_so_far :
                    matching_indices.append(index)
        test = value.eval(session=sess)
        for i in matching_indices:
            assignment_op = untrained[i].assign(tf.convert_to_tensor(test))
            sess2.run(assignment_op)
            test = untrained[i].eval(session=sess2)
            a = test.mean()

    train_step = tf.train.GradientDescentOptimizer(0.002).minimize(network.loss)
    test_weight = network.W_fc1.eval(session=sess2)
    mean_weight = test_weight.mean()
    for step in range(N):

        long_x1, batch_y1 = get_batch(Xdata_binary, Ydata_binary)
        long_x2, batch_y2 = get_batch(Xdata_binary, Ydata_binary)
        batch_y = (batch_y1 == batch_y2)
        batch_x1 = long_x1.reshape(len(long_x1), image_size, image_size, 1)
        batch_x2 = long_x2.reshape(len(long_x1), image_size, image_size, 1)

        _, loss_v = sess2.run([train_step, network.loss], feed_dict={
            network.x1: batch_x1,
            network.x2: batch_x2,
            network.y_: batch_y})
        print(str(step) + ': ' + str(loss_v))
        if step % 100 == 0:
            #saver.save(sess, './model/reload/mod', global_step=step)
            iv_long = np.array([x for (x, y) in zip(mnist.test.images, mnist.test.labels) if y < classes])
            y_test = np.array([y for y in mnist.test.labels if y < classes])
            image_vector = iv_long.reshape([len(iv_long), image_size, image_size, 1])
            # image_vector = mnist.test.images.reshape(len(mnist.test.images), image_size, image_size, 1)
            iv_long = iv_long[:1000, :]
            image_vector = image_vector[:1000, :, :, :]
            y_test = y_test[:1000]
            embed = network.o1.eval({network.x1: image_vector}, session=sess2)
            embed.tofile('embed.txt')
            image_vector = image_vector.reshape(1000, image_size, image_size)
            visualize.save(embed, image_vector, y_test, step)
    writer.close()