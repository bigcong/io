import os
import requests

import numpy as np
from PIL import Image

from sklearn import datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from 破解验证码.GET import GET
import matplotlib.pyplot as plt

input_x = tf.placeholder(tf.float32, [None, 600])
input_y = tf.placeholder(tf.float32, [None, 9])

w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

e1 = tf.layers.dense(input_x, 100, tf.nn.tanh, kernel_initializer=w_initializer,
                     bias_initializer=b_initializer, name='e1')

drop_out_e1 = tf.nn.dropout(e1, 0.5)

output_y = tf.layers.dense(drop_out_e1, 9, tf.nn.softmax, kernel_initializer=w_initializer,
                           bias_initializer=b_initializer, name='e2')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_y, labels=input_y))
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)


def compute_accuracy(sess, v_xs, v_ys):
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


def get_test_data():
    xx = []
    yy = []
    path = 'GET/'
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


saver = tf.train.Saver()


def train():
    with tf.Session() as sess:
        train_xx, train_yy = get_train_data()

        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, "logs/save_net.ckpt")

        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(train_xx, train_yy, test_size=.2)

            sess.run(train_op, feed_dict={input_x: X_train, input_y: y_train})
            print(compute_accuracy(sess, X_test, y_test))

        saver.save(sess, "logs/save_net.ckpt")


def test():
    get = GET()
    get.spit()

    with tf.Session() as sess:
        test_xx = get.dms
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, "logs/save_net.ckpt")

        p = sess.run(output_y, feed_dict={input_x: test_xx})
        pp = np.argmax(p, 1) + 1
        if viefiy(get.codeUUID, ''.join([str(xx) for xx in pp])):
            for index, (test_x, ppp, im) in enumerate(list(zip(test_xx, pp, get.ims))[0:10]):
                plt.subplot(2, 5, index + 1)
                plt.imshow(test_x.reshape(30, 20))
                im.save("data/" + str(ppp) + "_" + get.codeUUID + ".png")
                plt.title(ppp)
        else:
            for index, (test_x, ppp, im) in enumerate(list(zip(test_xx, pp, get.ims))[0:10]):
                plt.subplot(2, 5, index + 1)
                plt.imshow(test_x.reshape(30, 20))
                im.save("wrong/" + str(ppp) + "_" + get.codeUUID + ".png")
                plt.title(ppp)
        # plt.show()


def viefiy(codeId, varcode):
    url = "http://59.110.157.9/polarisex/user/loginGAFirst?email=18613868034&pwd=8123c3fc72f458ba6633c172d9c68ea2&vercode=" + varcode + "&source=1&codeid=" + codeId + "&local=zh_TW"
    print(url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}
    json = requests.post(url, headers=headers).json()
    print(json)

    if json['status'] == 200:
        return True
    else:
        return False


if __name__ == '__main__':
    for i in range(100):
        train()
        #test()
