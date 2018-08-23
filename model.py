import tensorflow as tf

class siamese:

    # Create model
    def __init__(self, sizes):
        self.x1 = tf.placeholder(tf.float32, [None, 784])
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.layers = []
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1,sizes)
            scope.reuse_variables()
            self.o2 = self.network(self.x2,sizes)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.custom_loss()

    def network(self, input_layer, sizes):

        i = 0
        for x in sizes:
            self.layers.append(self.layer_generation(input_layer, x, "layer"+str(i)))
            i=i+1
            input_layer=self.layers[-1]
        return self.layers[-1]

    def layer_generation(self, input, layer_size, name):
        input_len = input.get_shape()[1]
        seed = tf.truncated_normal_initializer(stddev=0.01)
        w = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, layer_size], initializer=seed)
        b = tf.get_variable(name+'_b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[layer_size], dtype=tf.float32) )
        out = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, w), b), name=name+'_out')
        return out

    def custom_loss(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

"""
    def custom_loss(self):
        margin=5.0
        #different_class_examples = tf.subtract(tf.constant(1.0, dtype=tf.float32), self.y_, name="diff_class")
        different_class_examples = tf.subtract(1.0, self.y_, name="diff_class")
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")
        distance = tf.pow(tf.subtract(self.o1, self.o2), 2)
        distance = tf.reduce_sum(distance, 1)
        distance = tf.sqrt(distance+1e-6, name="Distance")
        same_class_losses = tf.multiply(self.y_, distance)
        margin_tensor = tf.constant(margin,dtype=tf.float32, name="Margin")
        diff_class_losses = tf.multiply(labels_f, tf.pow(tf.maximum(0.0, tf.subtract(margin_tensor, distance)), 2.))
        losses = tf.add(same_class_losses, diff_class_losses)
        loss = tf.reduce_mean(losses, name="loss")#losses, name="loss")
        return loss
"""