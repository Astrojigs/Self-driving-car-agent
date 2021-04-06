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
        self.epsilon_min = 0.01
        self.gamma = gamma
        self.counter = 0
        self.rewards_list = []
        # Learning rate
        self.lr = lr

        self.env = env
        # Action space
        self.action_space = self.env.action_space
        self.num_action_space = self.action_space.shape
        # Observation space
        self.observation_space = self.env.observation_space
        self.num_observation_space = self.observation_space.shape # = (96, 96, 3)

        #Important things
        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = 64

        self.model = self.initialize_model()

    def initialize_model(self):
        input_layer = tf.keras.layers.Input(shape=self.num_observation_space) #(96, 96, 3)
        middle_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2),
                                                activation='relu', padding='same')(input_layer) #(48, 48, 64)

        third_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), strides=(2,2),
                                                activation='relu', padding='same')(middle_layer) #(24, 24, 128)
        fourth_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(4,4), strides=(4,4),
                                                activation='relu', padding='same')(third_layer) #(6, 6, 128)
        flatten = tf.keras.layers.Flatten()(fourth_layer)
        dense_layer = tf.keras.layers.Dense(512, activation='relu')(flatten)
        dense_layer2 = tf.keras.layers.Dense(256, activation='relu')(dense_layer)

        # Output
        output_layer = tf.keras.layers.Dense(self.action_space.sample().size,
                                             activation=tf.keras.activations.linear)(dense_layer2)
        # If linear activation function does not work then try tanh which will make the o/p vary from -1 to 1

        model = tf.keras.Model(inputs = [input_layer], outputs = [output_layer])

        model.compile(tf.keras.optimizers.Adam(lr=self.lr), loss='mse')

        return model

    def get_action(self, state):
        # Epsilon greedy policy
        if np.random.rand() > self.epsilon:
            # Use action recieved from the Model
            print(f"From 'get_action()' method || model.predict(states).numpy() = {self.model.predict(state)} \n should be [x y z] shape")
            return self.model.predict(state)[0]

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

        targets = np.tile(rewards, (self.action_space.sample().size, 1)).transpose() + (self.gamma * self.model.predict_on_batch(next_states)) * (1 - done_list)[:, None]
        # targets.shape = (64, 3)

        target_vec = self.model.predict_on_batch(states)
        # print(f'target_vec = {target_vec.shape}')
        indexes = np.array([i for i in range(self.batch_size)])

        target_vec = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def learn(self, num_episodes = 2000):
        for episode in range(num_episodes):
            #reset the environment
            state = self.env.reset()

            reward_for_episode = 0
            num_steps = 5000
            # print(state.shape) = (96, 96, 3)
            state = state.reshape((1,) + self.num_observation_space)

            #what to do in every step
            for step in range(num_steps):
                print(step)
                # Get the action
                received_action = self.get_action(state)
                # print(received_action) = [0.2579827  0.8255989  0.21661848]

                # Implement the action and the the next_states and rewards
                next_state, reward, done, info = env.step(received_action)

                # Render the actions
                self.env.render()

                # Reshape the next_state and put it in replay buffer
                next_state = next_state.reshape((1,) + self.num_observation_space)
                # Store the experience in replay_buffer
                self.add_to_replay_buffer(state, received_action, reward, next_state, done)

                # Add rewards
                reward_for_episode+=reward
                # Change the state
                state = next_state

                # Update the model
                self.update_counter()
                self.update_model()

                if done:
                    break

            self.rewards_list.append(reward_for_episode)

            # Decay the epsilon after each completion
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            last_reward_mean = np.mean(self.rewards_list[-100:])
            if last_reward_mean > 200:
                print("DQN Training Complete...")
                break

            # self.model.save('Cardriver.h5', overwrite=True)
            print(f"Episode: {episode} \n Reward: {reward_for_episode} \n Average Reward: {last_reward_mean} \n Epsilon: {self.epsilon}")

    def save(self, name):
        self.model.save(name)

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 2000

    '''Use this when training model'''
    model = DeepQNetwork(env, lr, epsilon, gamma, epsilon_decay)
    model.learn(training_episodes)

    model.save('Cardriver.h5', overwrite=True)
