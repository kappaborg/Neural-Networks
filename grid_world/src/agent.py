import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        """Neural Net for Deep-Q learning Model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse',
                     optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = state.reshape(1, -1)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                next_state = next_state.reshape(1, -1)
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state, verbose=0)[0]
                )
            
            state = state.reshape(1, -1)
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            states[i] = state
            targets[i] = target_f[0]
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
    
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name) 