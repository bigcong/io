import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.examples.tutorials.mnist import input_data

from 破解验证码.数字设别 import get_train_data


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


input_x = tf.placeholder(tf.float32, [None, 30 * 20]) / 255.
input_y = tf.placeholder(tf.float32, [None, 35])

keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(input_x, [-1, 30, 20, 1])

W_conv1 = weight_varibale([5, 5, 1, 32])  # patch 5*5 输入1 图片的高度  输出 32
b_conv1 = bias_varibale([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32
# output size 7*7*64

W_fc1 = weight_varibale([10 * 15 * 32, 1024])
b_fc1 = bias_varibale([1024])
h_pool2_flat = tf.reshape(h_pool1, [-1, 10 * 15 * 32])
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


with  tf.Session() as sess:
    init = tf.global_variables_initializer()

    sess.run(init)
    xx, yy = get_train_data()
    lb = LabelBinarizer();
    yy = lb.fit_transform(yy)

    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=.9)

        sess.run(train_step, feed_dict={input_x: X_train, input_y: y_train, keep_prob: 0.5})
        print(sess.run(cross_entropy, feed_dict={input_x: X_train, input_y: y_train, keep_prob: 0.5}),
              compute_accuracy(X_test, y_test))
