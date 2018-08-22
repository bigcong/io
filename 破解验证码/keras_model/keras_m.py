import os

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from 破解验证码.数字设别 import get_train_data


def create_and_save(checkpoint_path="logs/1.ckpt"):
    xx, yy = get_train_data()
    lb = LabelBinarizer();
    yy = lb.fit_transform(yy)

    _, zeros_shape = yy.shape
    yy = np.argmax(yy, 1)

    X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=.9)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),  # 防着overfiting
        tf.keras.layers.Dense(35, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # 训练模型
    model.fit(X_train, y_train, epochs=10,
              validation_data=(X_test, y_test),
              callbacks=[cp_callback])
    # 评价模型
    model.evaluate(X_test, y_test)
    model.summary()

    return model


if __name__ == '__main__':


    model = create_and_save()
    checkpoint_path = "logs/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback



