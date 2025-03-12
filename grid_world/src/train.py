import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import json

# Import local modules
from .environment import GridWorldEnv
from .agent import DQNAgent

class TrainingMonitor:
    def __init__(self, project_name, save_dir='checkpoints'):
        self.project_name = project_name
        self.save_dir = save_dir
        self.start_time = time.time()
        self.checkpoint_dir = os.path.join(save_dir, project_name, 
                                         time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'timing': []
        }
        
    def log_episode(self, episode, total_episodes, reward, steps, metrics=None):
        elapsed_time = time.time() - self.start_time
        remaining_time = (elapsed_time / (episode + 1)) * (total_episodes - episode - 1)
        
        self.history['episodes'].append(episode)
        self.history['rewards'].append(reward)
        self.history['steps'].append(steps)
        self.history['timing'].append(elapsed_time)
        
        if metrics is None:
            metrics = {}
            
        print(f"\n{self.project_name} - Episode {episode+1}/{total_episodes}")
        print(f"Reward: {reward:.2f}")
        print(f"Steps: {steps}")
        print(f"Time elapsed: {elapsed_time/3600:.2f} hours")
        print(f"Estimated time remaining: {remaining_time/3600:.2f} hours")
        for key, value in metrics.items():
            print(f"{key}: {value}")

def train_agent(episodes=1000, batch_size=32):
    # Initialize environment and agent
    env = GridWorldEnv(size=10)
    state_size = env.observation_space
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)
    monitor = TrainingMonitor('GridWorld')
    
    rewards_history = []
    steps_history = []
    
    print("Starting Grid World training...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for time_step in range(500):  # Maximum 500 steps per episode
            # Agent selects action
            action = agent.act(state)
            
            # Environment step
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            if done:
                break
        
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        # Calculate metrics
        metrics = {
            'epsilon': agent.epsilon,
            'memory_size': len(agent.memory)
        }
        
        # Log progress
        monitor.log_episode(episode, episodes, total_reward, steps, metrics)
        
        # Print grid state every 100 episodes
        if episode % 100 == 0:
            env.render()
    
    return rewards_history, steps_history

def plot_results(rewards, steps):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot steps
    ax2.plot(steps)
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train the agent
    rewards, steps = train_agent()
    
    # Plot results
    plot_results(rewards, steps)
    
    # Save the trained model
    agent.save("grid_world_model.h5") 