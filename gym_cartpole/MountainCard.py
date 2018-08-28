import time

from gym.envs.classic_control import MountainCarEnv

from gym_cartpole.RL_brain import DeepQNetwork

env = MountainCarEnv()
RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.0001, )

print(env.action_space)
total_steps = 0
for i_episode in range(10):

    observation = env.reset()
    reward_counter = 0
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        # 位置和速度
        position, velocity = observation_
        reward = abs(position - (-0.5))

        RL.store_transition(observation, action, reward, observation_)
        if total_steps > 500:
            RL.learn()
        reward_counter += reward
        reward_counter += reward

        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(reward_counter, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break
        total_steps += 1
        observation = observation_
RL.plot_cost()
