import numpy as np

import tensorflow as tf


class cnn_brain(object):
    # input_image_shape  图片的形状比如：长*宽*高（通道数)，get_image_url 下载图片的路径
    def __init__(self, train_xx, train_yy,
                 chang, kuan
                 ):
        _, self.input_size = train_xx.shape
        _, self.out_size = train_yy.shape
        self.chang = chang
        self.kuan = kuan

        self.build_net()

    def weight_varibale(self, shape):
        #  修剪的普通分布
        return tf.truncated_normal(shape, stddev=0.2)

    def bias_varibale(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, input, filter):  # 滤波器
        # [1, 1, 1, 1] 定义小方块的步长 ， x 方向 1 y 方向 1 前后都是1
        return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, value):
        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def build_net(self):
        # --------第一层卷积---------
        self.input_x = tf.placeholder(tf.float32, [None, self.input_size])
        self.input_y = tf.placeholder(tf.float32, [None, self.out_size])
        self.keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(self.input_x, [-1, self.chang, self.kuan, 1])
        W_conv1 = self.weight_varibale([5, 5, 1, 32])  # patch 5*5 输入1 图片的高度  输出 32
        B_conv1 = self.bias_varibale([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + B_conv1)
        p_conv1 = self.max_pool_2x2(h_conv1)  # (?, 15, 10, 32)  #  图片长和宽减半
        # --------第二层卷积---------

        W_conv2 = self.weight_varibale([5, 5, 32, 128])  # 5*5*32   32是图片的高度 5*5 是图片的长和宽，128 是输出高度
        B_conv2 = self.bias_varibale([128])
        h_conv2 = tf.nn.relu(self.conv2d(p_conv1, W_conv2) + B_conv2)
        p_conv2 = self.max_pool_2x2(h_conv2)  # (?, 8, 5, 128)

        # dropout  将输出结果变成 (?, 1024)
        p_conv2_shape_2 = (round(round(self.chang /2) / 2)) * (round(round(self.kuan /2) / 2)) * 128
        print(p_conv2_shape_2)

        p_flat = tf.reshape(p_conv2, [-1, p_conv2_shape_2])  # (?, 5120)
        w_flat = self.weight_varibale([p_conv2_shape_2, 1024])
        b_flat = self.bias_varibale([1024])
        h_flat = tf.nn.relu(tf.matmul(p_flat, w_flat) + b_flat)  # (?, 1024)
        drop_flat = tf.nn.dropout(h_flat, self.keep_prob)

        w = self.weight_varibale([1024, self.out_size])
        b = self.bias_varibale([self.out_size])

        self.output_y = tf.nn.softmax(tf.matmul(drop_flat, w) + b)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.input_y * tf.log(self.output_y),
                                                      reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        init = tf.global_variables_initializer()
        self.sess = tf.Session();
        self.sess.run(init)

    def compute_accuracy(self, v_xs, v_ys):
        global output_y
        y_pre = self.sess.run(self.output_y, feed_dict={self.input_x: v_xs, self.keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = self.sess.run(accuracy, feed_dict={self.input_x: v_xs, self.input_y: v_ys, self.keep_prob: 1})
        return result
