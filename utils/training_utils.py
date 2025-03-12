import time
import os
import json
from datetime import datetime
import tensorflow as tf

class TrainingMonitor:
    def __init__(self, project_name, save_dir='checkpoints'):
        self.project_name = project_name
        self.save_dir = save_dir
        self.start_time = time.time()
        self.checkpoint_dir = os.path.join(save_dir, project_name, 
                                         datetime.now().strftime('%Y%m%d_%H%M%S'))
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
            
        # Log progress
        print(f"\n{self.project_name} - Episode {episode+1}/{total_episodes}")
        print(f"Reward: {reward:.2f}")
        print(f"Steps: {steps}")
        print(f"Time elapsed: {elapsed_time/3600:.2f} hours")
        print(f"Estimated time remaining: {remaining_time/3600:.2f} hours")
        for key, value in metrics.items():
            print(f"{key}: {value}")
            
    def save_checkpoint(self, model, episode, metrics):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_episode_{episode}')
        
        # Save model weights
        model.save_weights(checkpoint_path + '.h5')
        
        # Save training history and metrics
        history_path = checkpoint_path + '_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'episode': episode,
                'history': self.history,
                'metrics': metrics
            }, f)
            
    def load_checkpoint(self, model, episode):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_episode_{episode}')
        
        # Load model weights
        model.load_weights(checkpoint_path + '.h5')
        
        # Load training history
        history_path = checkpoint_path + '_history.json'
        with open(history_path, 'r') as f:
            checkpoint_data = json.load(f)
            self.history = checkpoint_data['history']
            
        return checkpoint_data['metrics'] 