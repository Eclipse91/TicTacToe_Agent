import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
from keras.models import load_model


class QLearningAgent:
    def __init__(self, state_size, action_size, path_to_model):
        '''
        Initialize the Q-learning agent.
        Parameters:
            state_size (tuple): The dimensions of the state space.
            action_size (int): The number of possible actions.
            gamma: This is the discount factor, denoted by (gamma), and it determines the importance of future rewards. 
                A value of 0 means that only immediate rewards are considered, while a value approaching 1 considers 
                future rewards with greater weight.
            epsilon: This is the exploration rate, determining the likelihood that the agent will explore rather than 
                exploit the environment. It starts high to encourage exploration initially and decays over time as the 
                agent learns more about the environment.
            epsilon_min: This sets the minimum exploration rate. Once epsilon decays to this value or below, the agent 
                will stop decaying epsilon further, ensuring that it doesn't stop exploring entirely.
            epsilon_decay: This is the rate at which epsilon decreases over time. It's multiplied by epsilon each time 
                its update is triggered, leading to a gradual decrease in exploration as the agent learns more about 
                the environment.
            learning_rate: This is the step size or the rate at which the agent updates its Q-values (or neural network 
                weights in the case of DQN) based on the observed rewards and transitions. It determines how much the 
                agent relies on new information compared to its existing knowledge.
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.2 # 0.1 #
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.01
        self.model = self.load_saved_model(path_to_model) if path_to_model else self._build_model()

    def _build_model(self):
        '''
        Build the neural network model.
        Returns:
            keras.Model: The compiled neural network model.
        '''        
        model = keras.Sequential([
                keras.layers.Flatten(input_shape=(3,3)),  # Flatten the 3x3 board
                keras.layers.Dense(27, activation=tf.nn.relu),  # Dense layer with 27 neurons and ReLU activation
                # keras.layers.Dense(18, activation=tf.nn.relu),  # Dense layer with 18 neurons and ReLU activation
                keras.layers.Dense(9, activation=tf.nn.softmax)  # Output layer with 9 neurons for classification
            ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(loss='mse', optimizer=optimizer)

        return model

    def remember(self, state, action, reward, next_state, done):
        '''
        Remember a transition tuple in the replay memory.
        Parameters:
            state (numpy.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (numpy.ndarray): Next state.
            done (bool): Whether the episode has terminated.
        '''
        state = np.array(state).reshape(3, 3)
        next_state = np.array(next_state).reshape(3, 3)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        '''
        Choose an action based on the epsilon-greedy policy.
        Parameters:
            state (numpy.ndarray): Current state.
        Returns:
            int: Action to take.
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.expand_dims(state, axis=0),verbose=0)
        return np.argmax(act_values[0])  # Return action with highest Q-value

    def replay(self, batch_size):
        '''
        Experience replay and neural network training.
        Parameters:
            batch_size (int): Number of samples to train on in each batch.
        '''
        minibatch = random.sample(self.memory, batch_size)
        
        # Extracting individual components from minibatch
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Predict Q-values for current and next states
        current_q_values = self.model.predict(states,verbose=0)
        next_q_values = self.model.predict(next_states,verbose=0)

        # Calculate targets using vectorized operations
        targets = current_q_values.copy()
        targets[np.arange(batch_size), actions] = rewards + (1 - dones) * self.gamma * np.amax(next_q_values, axis=1)

        # Train the model using the states and targets
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        '''
        Save the model to the specified path. If an error occurs during saving, a fallback path ('model_file.h5') is attempted.        
        Parameters:
            path (str): The file path where the model will be saved.
        '''
        try:
            self.model.save(path)
        except (TypeError, IOError, FileNotFoundError, OSError, ValueError, NotImplementedError) as e:
            print(f"An error occurred while saving the model: {e}")
            try: 
                self.model.save('model_file.h5')
            except Exception as e:
                print(f'Issue saving the file: {e}')

    def load_saved_model(self, path):
        '''
        Load a saved model from the specified path.
        Parameters:
            path (str): The file path from which to load the model.
        Returns:
            model: The loaded Keras model.
        '''
        try:
            model = load_model(path)
            return model
        except (IOError, FileNotFoundError, OSError, ValueError) as e:
            raise f"An error occurred while loading the model from path. Set MODEL_FILE_PATH to "" if you don't want to load an existing model"