import numpy as np
import tensorflow as tf
import gym
from collections import deque
import random

class DeepQNetwork:
    def __init__(self, env, lr, epsilon, gamma, epsilon_decay):

        # Variables
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.counter = 0
        # Learning rate
        self.lr = lr

        self.env = env
        # Action space
        self.action_space = self.env.action_space
        self.num_action_space = self.action_space.n
        # Observation space
        self.observation_space = self.env.observation_space
        self.num_observation_space = self.observation_space.shape # = (96, 96, 3)

        #Important things
        replay_buffer = deque(maxlen=500000)
        self.batch_size = 64

        self.model = self.initialize_model()

    def initialize_model(self):
        input_layer = tf.keras.layers.Input(shape=self.num_observation_space) #(96, 96, 3)
        middle_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2),
                                                activation='relu', padding='same')(input_layer) #(48, 48, 64)

        third_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), strides=(2,2),
                                                activation='relu', padding='same')(middle_layer) #(24, 24, 128)
        flatten = tf.keras.layers.Flatten()(third_layer)
        dense_layer = tf.keras.layers.Dense(256,activation='relu')(flatten)

        # Output
        output_layer = tf.keras.layers.Dense(3, activation=tf.keras.activations.linear)(dense_layer)

        model = tf.keras.Model(inputs=[input_layer], outputs = [output_layer])

        model.compile(tf.keras.optimizers.Adam(lr=self.lr), loss='mse')

        return model

    def get_action(self, state):
        # Epsilon greedy policy
        if random.randrange(self.num_action_space) > self.epsilon:
            # Use action recieved from the Model
            print(f"From 'get_action()' method || model.predict(states).numpy() = {self.model.predict(state).numpy()} \n should be [x y z] shape")
            return self.model.predict(state).numpy()

        else:
            # return a random action_space
            return self.action_space.sample()

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_counter(self):
        self.counter+=1
        step_size = 5
        self.counter = self.counter % step_size

    def get_attributes_from_sample(self, sample):
        states = np.squeeze(np.squeeze(np.array([i[0] for i in sample])))
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        next_states = np.squeeze(np.array([i[3] for i in sample]))
        done_list = np.array([i[4] for i in sample])
        return states, actions, rewards, next_states, done_list

    def update_model(self):
        #replay_buffer size check
        if len(self.replay_buffer) < self.batch_size or self.counter != 0:
            return

        # Early Stopping
        if np.mean(self.rewards_list[-10:]) > 180:
            return

        #take a random sample
        random_sample = random.sample(self.replay_buffer, self.batch_size)

        # Extract the attributes from the sample
        states, actions, rewards, next_states, done_list = self.get_attributes_from_sample(random_sample)
