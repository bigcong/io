import os

import numpy as np
from PIL import Image

from sklearn import datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from 破解验证码.GET import GET

input_x = tf.placeholder(tf.float32, [None, 600])
input_y = tf.placeholder(tf.float32, [None, 8])

w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

e1 = tf.layers.dense(input_x, 50, tf.nn.tanh, kernel_initializer=w_initializer,
                     bias_initializer=b_initializer, name='e1')

drop_out_e1 = tf.nn.dropout(e1, 0.5)

output_y = tf.layers.dense(drop_out_e1, 8, tf.nn.softmax, kernel_initializer=w_initializer,
                           bias_initializer=b_initializer, name='e2')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_y, labels=input_y))
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)


def compute_accuracy(v_xs, v_ys):
    global output_y
    y_pre = sess.run(output_y, feed_dict={input_x: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={input_x: v_xs, input_y: v_ys})
    return result


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


if __name__ == '__main__':
    xx, yy = get_train_data()
    print(xx[0])
    print(yy[0])

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, "logs/save_net.ckpt")
    x = GET()
    x.plot()
    x.spit()

    p = sess.run(output_y, feed_dict={input_x: x.dms})
    pp = np.argmax(p, 1)
    print(pp)
