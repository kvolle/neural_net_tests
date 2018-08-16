import tensorflow as tf

class network:

    # Create model
    def __init__(self, sizes):
        self.x1 = tf.placeholder(tf.float32, [None, 784])
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        # Create loss
        self.y1 = tf.placeholder(tf.float32, [None])
        self.y2 = tf.placeholder(tf.float32, [None])
        self.loss = self.custom_loss()
        previous = 784
        i = 0
        self.layers = []
        input_layer = tf.placeholder(tf.float32, shape=[None, 784])
        for x in sizes:
            self.layers.append(self.layer_generation(input_layer, x, "layer"+str(i)))
            i=i+1

    def layer_generation(self, input, layer_size, name):
        input_len = input.get_shape()[1]
        seed = tf.truncated_normal_initializer(stddev=0.01)
        w = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, layer_size], initializer=seed)
        b = tf.get_variable(name+'_b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[layer_size], dtype=tf.float32) )
        fc = tf.nn.bias_add(tf.matmul(input, w), b)
        return fc

    def custom_loss(self):
