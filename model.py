import tensorflow as tf

class siamese:

    # Create model
    def __init__(self, sizes):
        self.keep_prob = 0.5 #tf.placeholder(tf.float32, name='dropout_prob')
        self.num_labels = 10
        self.x1 = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.x2 = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.layers = []
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1,sizes)
            scope.reuse_variables()
            self.o2 = self.network(self.x2,sizes)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.loss = self.custom_loss()
        self.acc = self.accuracy_summary()

    def network(self, input_layer, sizes):
        #i = 0
        input_layer_local = input_layer
        out_1 = self.conv_layer(input_layer_local, [5,5,1,32],'layer1')
        out_2 = self.conv_layer(out_1, [5, 5, 32, 64],'layer2')
        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(out_2, [-1, 7 * 7 * 64])
            #W_fc1 = self._create_weights([7 * 7 * 64, 1024])
            #b_fc1 = self._create_bias([1024])

            W_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=0.1, dtype=tf.float32))
            b_fc1 = tf.Variable(tf.constant(1., shape=[1024], dtype=tf.float32))
            local1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1, name=scope.name)
            #self._activation_summary(local1)

        with tf.variable_scope('local2_linear') as scope:
            W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, self.num_labels], stddev=0.1, dtype=tf.float32))
            b_fc2 = tf.Variable(tf.constant(1., shape=[self.num_labels], dtype=tf.float32))
            local1_drop = tf.nn.dropout(local1, self.keep_prob)
            local2 = tf.nn.bias_add(tf.matmul(local1_drop, W_fc2), b_fc2, name=scope.name)
            #self._activation_summary(local2)
        return local2
    """
        for x in sizes:
            self.layers.append(self.layer_generation(input_layer_local, x, "layer" + str(i)))
            i = i + 1
            input_layer_local = tf.nn.relu(self.layers[-1], name='out_'+str(i))
        return self.layers[-1]
    """
    def layer_generation(self, input, layer_size, name):
        input_len = input.get_shape()[1]
        seed = tf.truncated_normal_initializer(stddev=0.01)
        w = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, layer_size], initializer=seed)
        b = tf.get_variable(name+'_b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[layer_size], dtype=tf.float32) )
        #out = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, w, name=name+'_mul'), b,name=name+'_add'), name=name+'_out')
        out = tf.nn.bias_add(tf.matmul(input, w, name=name + '_mul'), b, name=name + '_add')
        return out

    def conv2d(self, input_layer, W):
        return tf.nn.conv2d(input=input_layer,
                            filter=W,
                            strides=[1, 1, 1, 1],
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

    def conv_layer(self, input_layer, weights, name):
        with tf.variable_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal(shape=weights, stddev=0.1, dtype=tf.float32))
            conv = self.conv2d(input_layer, kernel)
            bias = tf.Variable(tf.constant(1., shape=[weights[-1]], dtype=tf.float32))
            preactivation = tf.nn.bias_add(conv, bias)
            conv_relu = tf.nn.relu(preactivation, name=scope.name)
            self.activation_summary(conv_relu)
            h_pool = self.create_max_pool_layer(conv_relu)
        return h_pool

    def custom_loss(self):
        #distance = tf.pow(tf.subtract(self.y_, soft), 2.)
        #sum_step = tf.sqrt(tf.reduce_sum(distance, 1))
        #loss = tf.reduce_mean(sum_step, 0)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.o1))

        return loss
    def accuracy_summary(self):
        soft = tf.nn.softmax(self.o1)
#        correct = tf.cast(tf.argmax(self.o1, 1) == tf.argmax(self.y_, 1), dtype=tf.float32)
        distance = tf.pow(tf.subtract(self.y_, soft), 2.)
        sum_step = tf.sqrt(tf.reduce_sum(distance, 1))
        correct = tf.reduce_mean(sum_step,0)
        answer_guess = tf.argmax(soft, axis=1)
        answer_truth = tf.argmax(self.y_, axis=1)
        number_correct = tf.cast(tf.equal(answer_guess, answer_truth), dtype=tf.float64)
        #dist_test = tf.cast(tf.subtract(answer_truth, answer_guess), dtype=tf.float64)
        mean_correct = tf.reduce_mean(correct)
        distance = tf.summary.scalar("Distance", mean_correct)
        count = tf.summary.scalar("Correct",tf.reduce_mean(number_correct))
        #count = tf.summary.scalar("Wrongness", tf.reduce_mean(dist_test))
        guess_hist = tf.summary.histogram("Guess",answer_guess)
        answer_hist = tf.summary.histogram("Ground",answer_truth)
        test = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='siamese/layer1')
        t = test[0]
        first = t[:, :, 0, 16]
        test_image = tf.reshape(first, [1, 5, 5, 1])

        return [distance, count, guess_hist, answer_hist, tf.summary.image("pic", test_image)]
        #return tf.summary.scalar("Ave2", tf.reduce_mean(soft))
        """
        weight=1.5
        margin=2.0
        labels_t = tf.to_float(self.y_)
        labels_f = tf.subtract(1.0, labels_t, name="1-yi")
        distance2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        distance2 = tf.reduce_sum(distance2, 1)
        distance = tf.sqrt(distance2+1e-6, name="Distance")
        weight_tensor = tf.constant(weight, dtype=tf.float32, name="Weight")
        same_class_losses = tf.multiply(weight_tensor, tf.multiply(labels_t, distance2))
        margin_tensor = tf.constant(margin,dtype=tf.float32, name="Margin")
        diff_class_losses = tf.multiply(labels_f, tf.pow(tf.maximum(0.0, tf.subtract(margin_tensor, distance)), 2.))
        losses = tf.add(same_class_losses, diff_class_losses)
        loss = tf.reduce_mean(losses, name="loss")#losses, name="loss")
        return loss
        """
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
