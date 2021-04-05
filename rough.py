import numpy as np
import tensorflow as tf
import gym

env = gym.make('CarRacing-v0')
obs = env.reset()

# obs, reward, done, info = env.step(env.action_space.sample())
# print(obs.shape)
# (96, 96, 3)

for i in range(100):

    obs, reward, done, info = env.step(env.action_space.sample())
    print(env.action_space.sample().size)
