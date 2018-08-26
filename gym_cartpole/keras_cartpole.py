import os
import pathlib
import requests

import tensorflow as tf
from gym.envs.classic_control import CartPoleEnv
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import numpy as np

from 破解验证码.GET import GET
from 破解验证码.数字设别 import get_train_data


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(600,)),  # 保存和存储model,需要定义input_shape
        tf.keras.layers.Dropout(0.2),  # 防着overfiting
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
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


def chose_action(model, x_input=None, epsilon=0.9):
    if np.random.uniform() < epsilon and x_input != None:
        # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
        actions_value = model.predict(x_input)
        action = np.argmax(actions_value)
    else:
        action = np.random.randint(0, 2)  # 随机选择
    return action


def get():
    env = CartPoleEnv()

    for i_episode in range(10000):
        observation = env.reset()
        action = chose_action(model=model)
        while True:
            observation_, reward, done, info = env.step(action)
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            ransition = np.hstack((observation, [action, reward], observation_))
            print()


if __name__ == '__main__':

    env = CartPoleEnv()
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            print(reward)
            ransition = np.hstack((observation, [action, reward], observation_))
            print(ransition)


            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()

# Create checkpoint callback
