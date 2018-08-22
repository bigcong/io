import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential({
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# })
model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#  epochs 训练回合 ， batch_size 每回合多少次， 验证占比 validation_split
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
