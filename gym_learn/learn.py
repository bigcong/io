import gym
from gym.envs.classic_control import CartPoleEnv

env = CartPoleEnv()
env = env.unwrapped # 不做这个会有很多限制


print(env.action_space) # 查看这个环境中可用的 action 有多少个
print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)
