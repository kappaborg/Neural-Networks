import numpy as np
import tensorflow as tf
from collections import deque
import random

class AdvancedGridWorldAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_counter = 0
        
        # Additional features
        self.priority_memory = []  # For prioritized experience replay
        self.episode_history = []  # For tracking episode performance
        self.current_state_history = deque(maxlen=4)  # For temporal patterns
        
    def _build_model(self):
        """Neural Net for Deep-Q learning Model with advanced architecture"""
        # Input layers
        grid_input = tf.keras.layers.Input(shape=(self.state_size - 3,))  # Grid state
        env_input = tf.keras.layers.Input(shape=(3,))  # Environmental conditions
        
        # Process grid state
        grid = tf.keras.layers.Reshape((8, -1))(grid_input)  # Reshape to 2D grid
        
        # Convolutional layers for spatial patterns
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu')(grid)
        conv2 = tf.keras.layers.Conv1D(32, 3, activation='relu')(conv1)
        
        # Attention mechanism
        attention = tf.keras.layers.Attention()([conv2, conv2])
        
        # Flatten and combine with environmental conditions
        flat = tf.keras.layers.Flatten()(attention)
        combined = tf.keras.layers.Concatenate()([flat, env_input])
        
        # Dense layers with residual connections
        dense1 = tf.keras.layers.Dense(256, activation='relu')(combined)
        dense2 = tf.keras.layers.Dense(256, activation='relu')(dense1)
        residual = tf.keras.layers.Add()([dense1, dense2])
        
        # Dueling DQN architecture
        value_stream = tf.keras.layers.Dense(128, activation='relu')(residual)
        value = tf.keras.layers.Dense(1)(value_stream)
        
        advantage_stream = tf.keras.layers.Dense(128, activation='relu')(residual)
        advantage = tf.keras.layers.Dense(self.action_size)(advantage_stream)
        
        # Combine value and advantage
        q_values = tf.keras.layers.Add()([
            value,
            tf.keras.layers.Subtract()([
                advantage,
                tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1, keepdims=True))(advantage)
            ])
        ])
        
        model = tf.keras.Model(inputs=[grid_input, env_input], outputs=q_values)
        model.compile(loss='huber_loss',
                     optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience with priority"""
        # Calculate TD error for priority
        current_q = self.model.predict([
            state[:-3].reshape(1, -1),
            state[-3:].reshape(1, -1)
        ], verbose=0)[0]
        
        if not done:
            next_q = self.target_model.predict([
                next_state[:-3].reshape(1, -1),
                next_state[-3:].reshape(1, -1)
            ], verbose=0)[0]
            target_q = reward + self.gamma * np.max(next_q)
        else:
            target_q = reward
            
        td_error = abs(target_q - current_q[action])
        
        # Store experience with priority
        self.priority_memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'priority': td_error
        })
        
        # Keep memory within size limit
        if len(self.priority_memory) > 10000:
            self.priority_memory = sorted(
                self.priority_memory,
                key=lambda x: x['priority'],
                reverse=True
            )[:10000]
    
    def act(self, state):
        """Choose action using epsilon-greedy policy with additional strategies"""
        if np.random.rand() <= self.epsilon:
            # Smart exploration: bias towards unexplored actions
            if len(self.episode_history) > 0:
                action_counts = np.zeros(self.action_size)
                for history in self.episode_history[-100:]:  # Look at last 100 episodes
                    for action in history['actions']:
                        action_counts[action] += 1
                
                # Probability distribution favoring less-taken actions
                p = 1.0 / (action_counts + 1)
                p = p / np.sum(p)
                return np.random.choice(self.action_size, p=p)
            else:
                return random.randrange(self.action_size)
        
        # Split state into grid and environmental conditions
        grid_state = state[:-3].reshape(1, -1)
        env_state = state[-3:].reshape(1, -1)
        
        act_values = self.model.predict([grid_state, env_state], verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train on a batch of experiences using prioritized experience replay"""
        if len(self.priority_memory) < batch_size:
            return
        
        # Sort experiences by priority and select batch
        sorted_memory = sorted(
            self.priority_memory,
            key=lambda x: x['priority'],
            reverse=True
        )
        
        minibatch = sorted_memory[:batch_size]
        
        grid_states = np.zeros((batch_size, self.state_size - 3))
        env_states = np.zeros((batch_size, 3))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, experience in enumerate(minibatch):
            state = experience['state']
            action = experience['action']
            reward = experience['reward']
            next_state = experience['next_state']
            done = experience['done']
            
            # Split states
            grid_states[i] = state[:-3]
            env_states[i] = state[-3:]
            
            # Calculate target
            if not done:
                next_grid_state = next_state[:-3].reshape(1, -1)
                next_env_state = next_state[-3:].reshape(1, -1)
                
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(
                        [next_grid_state, next_env_state],
                        verbose=0
                    )[0]
                )
            else:
                target = reward
            
            target_f = self.model.predict(
                [grid_states[i:i+1], env_states[i:i+1]],
                verbose=0
            )[0]
            target_f[action] = target
            targets[i] = target_f
        
        # Train the model
        self.model.fit(
            [grid_states, env_states],
            targets,
            epochs=1,
            verbose=0
        )
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target network
        self.update_target_counter += 1
        if self.update_target_counter % 100 == 0:
            self.target_model.set_weights(self.model.get_weights())
    
    def update_episode_history(self, episode_data):
        """Update episode history with performance data"""
        self.episode_history.append(episode_data)
        if len(self.episode_history) > 1000:  # Keep last 1000 episodes
            self.episode_history.pop(0)
    
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name) 