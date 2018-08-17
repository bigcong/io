import numpy as np
import os

import tensorflow as tf
from PIL import Image
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

from 破解验证码.GET import GET


class brain(object):
    # input_image_shape  图片的形状比如：长*宽*高（通道数)，get_image_url 下载图片的路径
    def __init__(self, train_xx, train_yy):
        _, self.input_size = train_xx.shape
        _, self.out_size = train_yy.shape
        self.build_net()
        self.saver = tf.train.Saver()
        self.save_path = "logs/" + str(self.out_size) + ".ckpt"
        self.randow_pools = np.arange(97, 97 + 26)

    def build_net(self):
        self.input_x = tf.placeholder(tf.float32, [None, self.input_size])
        self.input_y = tf.placeholder(tf.float32, [None, self.out_size])

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        e1 = tf.layers.dense(self.input_x, 200, tf.nn.tanh, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='e1')

        drop_out_e1 = tf.nn.dropout(e1, 0.5)

        self.output_y = tf.layers.dense(drop_out_e1, self.out_size, tf.nn.softmax, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='e2')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_y, labels=self.input_y))
        self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def compute_accuracy(self, v_xs, v_ys):
        global output_y
        y_pre = self.sess.run(self.output_y, feed_dict={self.input_x: v_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = self.sess.run(accuracy, feed_dict={self.input_x: v_xs, self.input_y: v_ys})
        return result

    def train(self, train_xx, train_yy):
        try:

            self.saver.restore(self.sess, self.save_path)
            num = 10
        except:
            num = 100
            print("发生错误")
        for i in range(num):
            X_train, X_test, y_train, y_test = train_test_split(train_xx, train_yy, test_size=.2)

            self.sess.run(self.train_op, feed_dict={self.input_x: X_train, self.input_y: y_train})
            print(self.compute_accuracy(X_test, y_test))
        self.saver.save(self.sess, self.save_path)

    def test(self, lb):
        get = GET()
        get.spit()
        try:

            self.saver.restore(self.sess, self.save_path)
        except:
            print("发生错误")
        p = self.sess.run(self.output_y, feed_dict={self.input_x: get.dms})
        p_max = np.max(p, axis=1)
        print(p_max)
        pp = lb.inverse_transform(p)
        print(pp)
        for i, max in enumerate(p_max):
            if p_max[i] < 0.6:
                pp[i] = chr(np.random.choice(self.randow_pools))

        varcode = ''.join(pp)
        print(varcode)
        print(get.codeUUID)
        if get.viefiy(get.codeUUID, varcode):
            for index, (test_x, ppp, im) in enumerate(list(zip(get.dms, pp, get.ims))[0:10]):
                plt.subplot(2, 5, index + 1)
                plt.imshow(test_x.reshape(30, 20))
                im.save("data/" + str(ppp) + "_" + get.codeUUID + ".png")
                im.close()

                plt.title(ppp)
        else:
            for index, (test_x, ppp, im) in enumerate(list(zip(get.dms, pp, get.ims))[0:10]):
                plt.subplot(2, 5, index + 1)
                plt.imshow(test_x.reshape(30, 20))
                #im.save("wrong/" + str(ppp) + "_" + get.codeUUID + ".png")
                im.close()

                plt.title(ppp)
        # plt.show()
