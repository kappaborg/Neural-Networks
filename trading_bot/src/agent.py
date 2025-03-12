import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

# Enable mixed precision for better performance on M1/M2 Macs
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except:
    print("Mixed precision not supported on this device")

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Enhanced Hyperparameters
        self.memory = deque(maxlen=10000)  # Increased memory size
        self.gamma = 0.99  # Increased discount rate for better long-term rewards
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05  # Increased minimum exploration
        self.epsilon_decay = 0.997  # Slower decay for better exploration
        self.learning_rate = 0.00025  # Reduced learning rate for stability
        self.update_target_frequency = 5  # More frequent target updates
        self.batch_normalization = True
        
        # Prioritized Experience Replay parameters
        self.priority_scale = 0.7  # How much prioritization to use (0 = none, 1 = full)
        self.priority_offset = 0.1  # Small offset to ensure all experiences get some priority
        
        # Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
        # Training counter and metrics
        self.train_step = 0
        self.loss_history = []
        
        # Performance optimization
        self.training_batch = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        self.batch_counter = 0
        self.batch_size_threshold = 32  # When to trigger a training update
    
    def _build_model(self):
        """Enhanced Neural Net for Deep-Q learning Model"""
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        
        # First layer with batch normalization
        x = tf.keras.layers.Dense(128)(inputs)
        if self.batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Second layer
        x = tf.keras.layers.Dense(128)(x)
        if self.batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # LSTM layer for temporal dependencies
        x = tf.keras.layers.Reshape((1, 128))(x)
        x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
        
        # Value and Advantage streams (Dueling DQN)
        value_stream = tf.keras.layers.Dense(32, activation='relu')(x)
        value = tf.keras.layers.Dense(1)(value_stream)
        
        advantage_stream = tf.keras.layers.Dense(32, activation='relu')(x)
        advantage = tf.keras.layers.Dense(self.action_size)(advantage_stream)
        
        # Combine value and advantage (Dueling architecture)
        outputs = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use legacy optimizer for M1/M2 Macs with better parameters
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Huber loss for better stability
        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=optimizer,
            metrics=['mae']
        )
        
        return model
    
    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.001  # Small coefficient for soft update
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
        
        self.target_model.set_weights(target_weights)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience with priority"""
        # Calculate priority (TD error as proxy)
        target = reward
        if not done:
            target += self.gamma * np.amax(
                self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
            )
        current_q = self.model.predict(state.reshape(1, -1), verbose=0)[0][action]
        priority = (abs(target - current_q) + self.priority_offset) ** self.priority_scale
        
        self.memory.append((state, action, reward, next_state, done, priority))
        
        # Batch processing for better performance
        self.training_batch['states'].append(state)
        self.training_batch['actions'].append(action)
        self.training_batch['rewards'].append(reward)
        self.training_batch['next_states'].append(next_state)
        self.training_batch['dones'].append(done)
        
        self.batch_counter += 1
        if self.batch_counter >= self.batch_size_threshold:
            self.train_on_batch()
    
    def train_on_batch(self):
        """Process accumulated batch"""
        if self.batch_counter < self.batch_size_threshold:
            return
        
        states = np.array(self.training_batch['states'])
        next_states = np.array(self.training_batch['next_states'])
        
        # Predict Q-values for efficiency (single batch prediction)
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_counter):
            if self.training_batch['dones'][i]:
                current_q_values[i][self.training_batch['actions'][i]] = self.training_batch['rewards'][i]
            else:
                current_q_values[i][self.training_batch['actions'][i]] = (
                    self.training_batch['rewards'][i] + 
                    self.gamma * np.amax(next_q_values[i])
                )
        
        # Train in single batch
        history = self.model.fit(
            states, 
            current_q_values, 
            epochs=1, 
            verbose=0,
            batch_size=32
        )
        self.loss_history.append(history.history['loss'][0])
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.update_target_frequency == 0:
            self.update_target_network()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Clear batch
        self.training_batch = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        self.batch_counter = 0
    
    def act(self, state):
        """Choose action using epsilon-greedy with noise for exploration"""
        if np.random.rand() <= self.epsilon:
            # Add some directed noise to random actions
            action = random.randrange(self.action_size)
            if np.random.rand() < 0.3:  # 30% of random actions are smart random
                # Predict and add noise
                act_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
                noise = np.random.normal(0, 0.5, self.action_size)
                action = np.argmax(act_values + noise)
            return action
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Sample from prioritized experience replay"""
        if len(self.memory) < batch_size:
            return
        
        # Get priorities for all memories
        priorities = np.array([memory[5] for memory in self.memory])
        probabilities = priorities / np.sum(priorities)
        
        # Sample based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        minibatch = [self.memory[idx] for idx in indices]
        
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        
        for i, (state, action, reward, next_state, done, _) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # Double DQN update
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Get actions from main network for next states
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                # Use action from main network, but Q-value from target network
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * next_q_values[i][next_actions[i]]
        
        # Train with importance sampling weights
        self.model.fit(states, current_q_values, epochs=1, verbose=0, batch_size=batch_size)
    
    def save(self, filepath):
        """Save the model and training state"""
        try:
            save_dir = os.path.dirname(filepath)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model
            self.model.save(filepath)
            
            # Save training state
            state_path = os.path.join(save_dir, 'training_state.npz')
            np.savez(
                state_path,
                epsilon=self.epsilon,
                train_step=self.train_step,
                loss_history=np.array(self.loss_history)
            )
            print(f"Model and training state saved to {save_dir}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load(self, filepath):
        """Load the model and training state"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(filepath)
            self.target_model = tf.keras.models.load_model(filepath)
            
            # Load training state
            state_path = os.path.join(os.path.dirname(filepath), 'training_state.npz')
            if os.path.exists(state_path):
                training_state = np.load(state_path)
                self.epsilon = training_state['epsilon']
                self.train_step = training_state['train_step']
                self.loss_history = training_state['loss_history'].tolist()
            
            print(f"Model and training state loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {str(e)}") 