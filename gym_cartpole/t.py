import numpy as np

from gym.envs.classic_control import CartPoleEnv
from gym.spaces import Discrete


def choose_action(epsilon, observation):
    # 统一 observation 的 shape (1, size_of_observation)
    observation = observation[np.newaxis, :]

    if np.random.uniform() < 0.9:
        # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
    else:
        action = np.random.randint(0, self.n_actions)  # 随机选择
    return action


def go2():
    env = CartPoleEnv()
    episode_step_counter = 0
    for i_episode in range(10000):
        action = env.reset()

        step_counter = 0;
        while True:
            env.render()
            # 随机选择一个action
            # 获取环境给予的奖励
            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            print(reward)

            step_counter = step_counter + 1

            if done:
                episode_step_counter += step_counter
                # print("第{}回合，坚持了{}步".format(i_episode, step_counter))
                print("平均步数:{}".format(episode_step_counter / (i_episode + 1)))

                break

    env.close()


if __name__ == '__main__':
    batch_index = np.arange(32, dtype=np.int32)
    print(batch_index)





    batch_memory = np.random.uniform(0, 2, 32 * 10).reshape(32, 10)
    reward = batch_memory[:, 5]
    print(reward)
    eval_act_index = batch_memory[:, 4].astype(int)
    print(eval_act_index)




    q_target = np.random.uniform(0, 2, 64).reshape(32, 2)

    q_target[batch_index, eval_act_index] = 1
    print(q_target)
