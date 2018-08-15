import numpy as np
import os

import tensorflow as tf
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer


class c_brain(object):
    def __init__(self,
                 x_size=600,
                 y_size=10
                 ):
        input_x = tf.placeholder(tf.float32, [None, x_size])
        input_y = tf.placeholder(tf.float32, [None, y_size])
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        e1 = tf.layers.dense(input_x, 50, tf.nn.tanh, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='e1')
        dropout_e1 = tf.nn.dropout(e1, 0.5)
        output_y = tf.layers.dense(dropout_e1, 10, tf.nn.softmax, kernel_initializer=w_initializer,
                                   bias_initializer=b_initializer, name='e2')
        cross_entroy = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(output_y), reduction_indices=[1]))
        train_setp = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_train_data(self):
        xx = []
        yy = []
        path = 'data/'
        lists = os.listdir(path)  # 列出目录的下所有文件和文件夹保存到lists
        for i in lists:
            im = Image.open(path + i)
            im = im.convert("L")  # 转成灰色模式
            data = im.getdata()
            data = np.array(data) / 225  # 转换成矩阵

            yy.append(int(i.split("_")[0]))
            xx.append(np.array(data))
        yy = LabelBinarizer().fit_transform(yy)
        return xx, yy


if __name__ == '__main__':
    x = c_brain()
    xx, yy = x.get_train_data()
    print(yy.shape)

