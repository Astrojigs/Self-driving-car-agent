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

    def normalize_obs(self, state):
        if 100<np.max(state)<=255:
            return state/np.max(state)
        return state

    def initialize_model(self):
        input_layer = tf.keras.layers.Input(shape=self.num_observation_space) #(96, 96, 3)
        middle_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2),
                                                activation='relu', padding='same')(input_layer) #(48, 48, 64)

        third_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), strides=(2,2),
                                                activation='relu', padding='same')(middle_layer) #(24, 24, 128)
        fourth_layer = tf.keras.layers.Conv2D(filters=256, kernel_size=(4,4), strides=(4,4),
                                                activation='relu', padding='same')(third_layer) #(6, 6, 128)
        flatten = tf.keras.layers.Flatten()(fourth_layer)
        dense_layer = tf.keras.layers.Dense(512, activation='relu')(flatten)
        dense_layer2 = tf.keras.layers.Dense(256, activation='relu')(dense_layer)

        # Output
        output_layer = tf.keras.layers.Dense(4,
                                             activation=tf.keras.activations.softmax)(dense_layer2)
        # o/p = [L R A B]

        model = tf.keras.Model(inputs = [input_layer], outputs = [output_layer])

        model.compile(tf.keras.optimizers.Adam(lr=self.lr), loss='categorical_crossentropy')

        model.summary()

        return model

    def get_action(self, state):
        # Epsilon greedy policy
        state = self.normalize_obs(state)

        if np.random.rand() > self.epsilon:
            # Define the action
            action = np.zeros((self.action_space.sample().size,))
            # print(f"Debug code: get_action() ||| action before assignment = {action}")
            # Receiving the probabilities from the model
            action_proba = self.model.predict(state)[0]
            print(f"action probabilities received from the model = action_proba = {action_proba}")

            # Turn left
            if action_proba[0] == np.max(action_proba):
                action[0] = -1
            # Turn Right
            elif action_proba[1] == np.max(action_proba):
                action[0] = +1
                # Accelerate
            elif action_proba[2] == np.max(action_proba):
                action[1] = +1
            # Brake
            elif action_proba[3] == np.max(action_proba):
                action[2] = 0.8

                # print(f"Assigned action = {action}")
            else:
                return np.zeros((3,))

            return action



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

    def discretize_action(self, action):
        # received actions will be batch_size*[s a b]
        '''steer left if s < -0.5:
        steer right if s> +0.5
        accelerate if a>0.5
        brake if b>0.5'''
        discrete_action = np.zeros((4,))

        # Steering
        if action[0] < -0.5:
            # Steer left
            discrete_action[0] = 1
        else:
            discrete_action[0] = 0

        if action[0] > 0.5:
            # Steer right
            discrete_action[1] = +1
        else:
            discrete_action[1] = 0

        # Accelerate
        if action[1] > 0.5:
            discrete_action[2] = +1
        else:
            discrete_action[2] = 0


        # Brake
        if action[2] > 0.5:
            discrete_action[3] = 0.8
        else:
            discrete_action[3] = 0
        # print(f"from discrete_action() ||| discrete_action = {discrete_action}") = [0.  1.  1.  0.8]
        return discrete_action
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
        targets = rewards + self.gamma * np.argmax(self.model.predict_on_batch(next_states), axis=1) * (1 - done_list)
        # print(targets.shape) # = (64,)

        # Discretize the action
        discretized_actions = np.random.randint(0, 1, (self.batch_size, 4))
        for i in range(self.batch_size):
            discretized_actions[i] = self.discretize_action(actions[i])
        # print(discretized_actions.shape) = (64, 4)
        target_vec = self.model.predict_on_batch(states)
        # print(f"target_vec.shape = {target_vec.shape}") = (64, 4)
        # target_vec = [L R A B]
        # print(f'target_vec = {target_vec.shape}')
        indexes = np.array([i for i in range(self.batch_size)])
        # replace maximum value of target_vec with the target value
        target_vec[[indexes], [np.argmax(discretized_actions, axis=1)]] = targets
        # print(target_vec)

        '''The problem is that the target_vec uses the actions [s a b] and I've '''
        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def learn(self, num_episodes = 2000):
        for episode in range(num_episodes):
            #reset the environment
            state = self.env.reset()
            self.normalize_obs(state)
            reward_for_episode = 0
            num_steps = 1000
            # print(state.shape) = (96, 96, 3)
            state = state.reshape((1,) + self.num_observation_space)

            #what to do in every step
            for step in range(num_steps):
                # Get the action
                received_action = self.get_action(state)
                # print(received_action) = [0.2579827  0.8255989  0.21661848]

                # Implement the action and the the next_states and rewards
                next_state, reward, done, info = env.step(received_action)

                self.normalize_obs(next_state)
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
