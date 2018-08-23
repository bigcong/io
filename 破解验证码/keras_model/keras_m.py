import os
import pathlib
import requests

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import numpy as np

from 破解验证码.GET import GET
from 破解验证码.数字设别 import get_train_data


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(600,)),  # 保存和存储model,需要定义input_shape
        tf.keras.layers.Dropout(0.2),  # 防着overfiting
        tf.keras.layers.Dense(35, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def get_data():
    xx, yy = get_train_data()
    lb = LabelEncoder();
    yy = lb.fit_transform(yy)

    X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=.9)
    return X_train, X_test, y_train, y_test, lb


def train_and_save(checkpoint_path="logs/cp.ckpt"):
    X_train, X_test, y_train, y_test, _ = get_data()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     period=10)
    try:
        model = tf.keras.models.load_model("logs/cp.h5")
    except:
        model = create_model()

    # 训练模型
    model.fit(X_train, y_train, epochs=10,
              validation_data=(X_test, y_test), validation_split=0.5,
              callbacks=[cp_callback])
    # 评价模型

    model.summary()
    model.save("logs/cp.h5")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, lb = get_data()
    model = tf.keras.models.load_model("logs/cp.h5")
    for i in range(100):
        if i % 50 == 0:
            train_and_save()

        get = GET()
        get.getImage()
        get.spit()
        p_varcode = "".join(lb.inverse_transform(np.argmax(model.predict(get.dms), 1)))
        if (get.viefiy(get.codeUUID, p_varcode)):
            get.im.save("/Users/cc/cc/io/破解验证码/right/" + p_varcode + ".png")
        else:
            get.im.save("/Users/cc/cc/io/破解验证码/wrong/" + p_varcode + ".png")

# Create checkpoint callback
