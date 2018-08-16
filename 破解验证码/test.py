import tensorflow as tf

input_y = tf.placeholder(tf.float32, [None, 8])

with tf.Session() as sess:


    saver.restore(sess, "logs/save_net.ckpt")
