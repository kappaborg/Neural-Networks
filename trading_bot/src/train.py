import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import time

# Import local modules
from .environment import TradingEnv
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
            'returns': [],
            'portfolio_values': [],
            'timing': []
        }
        
    def log_episode(self, episode, total_episodes, returns, portfolio_value, metrics=None):
        elapsed_time = time.time() - self.start_time
        remaining_time = (elapsed_time / (episode + 1)) * (total_episodes - episode - 1)
        
        self.history['episodes'].append(episode)
        self.history['returns'].append(returns)
        self.history['portfolio_values'].append(portfolio_value)
        self.history['timing'].append(elapsed_time)
        
        if metrics is None:
            metrics = {}
            
        print(f"\nTrading Bot - Episode {episode+1}/{total_episodes}")
        print(f"Returns: ${returns:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Time elapsed: {elapsed_time/3600:.2f} hours")
        print(f"Estimated time remaining: {remaining_time/3600:.2f} hours")
        for key, value in metrics.items():
            print(f"{key}: {value}")

def train_agent(symbol='MSFT', episodes=100, batch_size=32, data_source='yfinance'):
    """Train the trading agent"""
    try:
        print(f"\nStarting training with {data_source} data for {symbol}...")
        
        # Setup dates
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Initialize environment and agent
        env = TradingEnv(symbol, start_date, end_date, data_source=data_source)
        state_size = env.observation_space
        action_size = env.action_space
        agent = DQNAgent(state_size, action_size)
        monitor = TrainingMonitor('TradingBot')
        
        returns_history = []
        portfolio_values = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Agent selects action
                action = agent.act(state)
                
                # Environment step
                next_state, reward, done = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Update state and total reward
                state = next_state
                total_reward += reward
                
                # Train agent
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
            
            # Log progress
            returns_history.append(total_reward)
            portfolio_values.append(env.current_value)
            
            # Calculate metrics
            metrics = {
                'epsilon': agent.epsilon,
                'memory_size': len(agent.memory),
                'avg_return': np.mean(returns_history[-10:]) if returns_history else 0
            }
            
            # Log progress
            monitor.log_episode(episode, episodes, total_reward, env.current_value, metrics)
            
            # Save model periodically
            if (episode + 1) % 10 == 0:
                agent.save(os.path.join(monitor.checkpoint_dir, f'model_episode_{episode+1}.h5'))
                
            # Render environment periodically
            if (episode + 1) % 20 == 0:
                env.render()
        
        print("\nTraining completed!")
        return returns_history, portfolio_values
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        return [], []

def plot_results(returns, portfolio_values):
    """Plot training results"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot returns
        ax1.plot(returns, label='Episode Returns')
        ax1.set_title('Trading Returns per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Returns ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot portfolio value
        ax2.plot(portfolio_values, label='Portfolio Value', color='green')
        ax2.set_title('Portfolio Value over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting results: {str(e)}")

if __name__ == "__main__":
    # Train the agent
    returns, portfolio_values = train_agent()
    
    # Plot results
    plot_results(returns, portfolio_values) 