"""
Deep Q network,
Using:
Tensorflow: 1.0
gym: 0.7.3
"""
import numpy as np

from gym.envs.classic_control import CartPoleEnv
import tensorflow as tf


def save():
    env = CartPoleEnv()

    total_steps = 0
    memory = []

    memory_counter = 0
    for i_episode in range(100):

        observation = env.reset()
        while True:
            env.render()
            action = env.action_space.sample()

            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            transition = np.hstack((observation, [action, reward], observation_))
            memory.append(transition)

            if done:
                break

            observation = observation_
            total_steps += 1
    memory = np.array(memory)
    np.save("memory.npy", memory)

    env.close()


def load():
    b = np.load("memory.npy")
    return b


def create_model():
    try:
        model = tf.keras.models.load_model("logs/cp.h5")

    except:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2, activation=tf.nn.relu, input_shape=(4,)),  # 保存和存储model,需要定义input_shape
            tf.keras.layers.Dropout(0.5),  # 防着overfiting
            # tf.keras.layers.Dense(2, activation=tf.nn.relu)
        ])

        # loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])

    return model


def get_data(memory=None, model=None):
    if memory is None:
        memory = load()
    if model is None:
        model = create_model();
    s_, s = memory[:, -4:], memory[:, :4]
    q_next, q_eval = model.predict(s_), model.predict(s)
    q_target = q_eval.copy()
    reward = memory[:, 5]
    actions = memory[:, 4].astype(int)
    index, _ = q_target.shape
    q_target[np.arange(index), actions] = reward + 0.9 * np.max(q_next, axis=1)
    return s, q_target


def go():
    env = CartPoleEnv()

    total_steps = 0
    memory = []
    model = create_model()

    epsilon = 0.9
    memory_counter = 1000
    for i_episode in range(1000):

        observation = env.reset()
        ep_r = 0

        while True:
            env.render()

            if np.random.uniform() < epsilon:
                actions_value = model.predict(np.array([observation]))
                action = np.argmax(actions_value)
            else:
                action = env.action_space.sample()

            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            transition = np.hstack((observation, [action, reward], observation_))
            memory.append(transition)
            if len(memory) > memory_counter:
                xx, yy = get_data(np.array(memory), model);
                print(xx.shape)
                model.fit(xx, yy, epochs=10)
                epsilon = epsilon + 0.00001
                memory = []
                # memory_counter = memory_counter + 5
            ep_r = ep_r + reward

            if done:
                # print(ep_r)

                break

            observation = observation_
            total_steps += 1

    model.save("logs/cp.h5")
    model.summary()
    env.close()


if __name__ == '__main__':
    go()
