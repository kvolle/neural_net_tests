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
        self.y_ = tf.placeholder(tf.bool, [None])
        self.loss = self.custom_loss()

    def network(self, input_layer, sizes):
        i = 0
        input_layer_local = input_layer
        for x in sizes:
            self.layers.append(self.layer_generation(input_layer_local, x, "layer" + str(i)))
            i = i + 1
            input_layer_local = tf.nn.relu(self.layers[-1], name='out_'+str(i))
        return self.layers[-1]

    def layer_generation(self, input, layer_size, name):
        input_len = input.get_shape()[1]
        seed = tf.truncated_normal_initializer(stddev=0.01)
        w = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, layer_size], initializer=seed)
        b = tf.get_variable(name+'_b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[layer_size], dtype=tf.float32) )
        #out = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, w, name=name+'_mul'), b,name=name+'_add'), name=name+'_out')
        out = tf.nn.bias_add(tf.matmul(input, w, name=name + '_mul'), b, name=name + '_add')
        return out

    def conv2d(self, input_layer, W):
        return tf.nn.conv2d(input= input_layer,
                            filter=W,
                            strides=[1,1,1,1],
                            padding='SAME')

    def create_max_pool_layer(self, input):
        return  tf.nn.max_pool(value=input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    def activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def conv_layer(self, weights, name):
        with tf.variable_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal(shape=weights, stddev=0.1, dtype=tf.float32))
            conv = self.conv2d(images, kernel)
            bias = tf.Variable(tf.constant(1., shape=[weights[-1]], dtype=tf.float32))
            preactivation = tf.nn.bias_add(conv, bias)
            conv_relu = tf.nn.relu(preactivation, name=scope.name)
            self.activation_summary(conv_relu)
            h_pool = self.create_max_pool_layer(conv_relu)

    def custom_loss(self):
        margin=5.0
        labels_t = tf.to_float(self.y_)
        labels_f = tf.subtract(1.0, labels_t, name="1-yi")
        distance2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        distance2 = tf.reduce_sum(distance2, 1)
        distance = tf.sqrt(distance2+1e-6, name="Distance")
        same_class_losses = tf.multiply(labels_t, distance2)
        margin_tensor = tf.constant(margin,dtype=tf.float32, name="Margin")
        diff_class_losses = tf.multiply(labels_f, tf.pow(tf.maximum(0.0, tf.subtract(margin_tensor, distance)), 2.))
        losses = tf.add(same_class_losses, diff_class_losses)
        loss = tf.reduce_mean(losses, name="loss")#losses, name="loss")
        return loss

"""
    def custom_loss(self):
        margin = 5.0
        labels_t = tf.to_float(self.y_)
        labels_f = tf.subtract(1.0, labels_t, name="1-yi")          # labels_ = !labels;
        tf.write_file("test.csv",str(tf.reduce_mean(labels_f)),name="Debug")
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
