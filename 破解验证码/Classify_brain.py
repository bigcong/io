import numpy as np
import os

import tensorflow as tf
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

x_size = 600
y_size = 3

input_x = tf.placeholder(tf.float32, [None, x_size])
input_y = tf.placeholder(tf.float32, [None, y_size])


def add_layer(inputs, in_size, out_size, activation_function=None):
    # 定一个矩阵in_size*out_size
    Weight = tf.Variable(tf.random_uniform([in_size, out_size]))
    # 定一个矩阵1*out_size 都是0.1 的矩阵
    Biase = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx = tf.matmul(inputs, Weight) + Biase
    Wx = tf.nn.dropout(Wx, 0.5)

    if activation_function is None:
        # 线性方程
        output = Wx
    else:
        output = activation_function(Wx)
    tf.summary.histogram('/outputs', output)
    return output


l1 = add_layer(input_x, x_size, 100, activation_function=tf.nn.tanh)

output_y = add_layer(l1, 100, y_size, activation_function=tf.nn.softmax)

cross_entroy = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(output_y), reduction_indices=[1]))

train_setp = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def get_train_data():
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
    return np.array(xx), yy


def compute_accuracy(v_xs, v_ys):
    global output_y
    y_pre = sess.run(output_y, feed_dict={input_x: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={input_x: v_xs, input_y: v_ys})
    return result





if __name__ == '__main__':
    xx, yy = get_train_data()
    print(xx.shape)
    print(yy.shape)

