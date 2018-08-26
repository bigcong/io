from gym.envs.classic_control import CartPoleEnv

from gym_cartpole.tt import DeepQNetwork

RL = DeepQNetwork()

total_steps = 0
env = CartPoleEnv()

for i_episode in range(10000):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('回合: ', i_episode,
                  'ep_r: ', round(ep_r, 2))
            break

        observation = observation_
        total_steps += 1

# RL.plot_cost()
