import time

from gym.envs.classic_control import AcrobotEnv

from gym_cartpole.RL_brain import DeepQNetwork

env = AcrobotEnv()
RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001, )

print(env.action_space)
total_steps = 0
for i_episode in range(10):

    observation = env.reset()
    reward_counter = 0
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        cos1, sin1, cos2, cos1, rad1, rad2 = observation_

        reward = 2 - cos1 - cos2 + reward
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()
        reward_counter += reward

        if done:
            print("奖励：", reward_counter)
            break
        total_steps += total_steps
        observation = observation_
