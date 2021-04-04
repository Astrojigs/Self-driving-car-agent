import numpy as np
import tensorflow as tf
import gym

env = gym.make('CarRacing-v0')
obs = env.reset()
for step in range(1000):
    action = env.step(env.action_space.sample())
    env.render()

env.close()
