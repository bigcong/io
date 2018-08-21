import os

import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def get_train_data():
    xx = []
    yy = []
    path = 'data/'
    lists = os.listdir(path)  # 列出目录的下所有文件和文件夹保存到lists
    for i in lists:
        im = Image.open(path + i)
        data = im.getdata()
        data = np.array(data) / 255.0  # 转换成矩阵

        yy.append(i.split("_")[0])
        xx.append(np.array(data))
    return np.array(xx), yy


def weight_varibale(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_varibale(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # [1, 1, 1, 1] 定义小方块的步长 ， x 方向 1 y 方向 1 前后都是1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


input_x = tf.placeholder(tf.float32, [None, 30 * 20])
input_y = tf.placeholder(tf.float32, [None, 35])

keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(input_x, [-1, 30, 20, 1])

W_conv1 = weight_varibale([5, 5, 1, 32])  # patch 5*5 输入1 图片的高度  输出 32
b_conv1 = bias_varibale([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # output size 15*10*32

# W_conv2 = weight_varibale([5, 5, 32, 128])  # patch 5*5 输入1 图片的高度  输出 32
# b_conv2 = bias_varibale([128])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)  # output size 5*8*128


# W_conv3 = weight_varibale([5, 5, 128, 256])  # patch 5*5 输入1 图片的高度  输出 32
# b_conv3 = bias_varibale([256])
# h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# h_pool3 = max_pool_2x2(h_conv3)  # output size 3*4*256


W_fc1 = weight_varibale([15 * 10 * 32, 1024])
b_fc1 = bias_varibale([1024])
h_pool2_flat = tf.reshape(h_pool1, [-1, 15 * 10 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 防止过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_varibale([1024, 35])
b_fc2 = bias_varibale([35])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={input_x: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={input_x: v_xs, input_y: v_ys, keep_prob: 1})
    return result


x_x = []
y_y = []
y1_y1 = []

with  tf.Session() as sess:
    init = tf.global_variables_initializer()

    sess.run(init)
    xx, yy = get_train_data()

    lb = LabelBinarizer();
    yy = lb.fit_transform(yy)

    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=.9)
        x_x.append(i)



        loss = sess.run(cross_entropy, feed_dict={input_x: X_train, input_y: y_train, keep_prob: 0.5})



        accuracy = compute_accuracy(X_test, y_test)
        print(loss, accuracy)
        y_y.append(loss)
        y1_y1.append(accuracy)

