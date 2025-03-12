import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_agent, plot_results

if __name__ == "__main__":
    print("Starting Grid World training...")
    rewards, steps = train_agent(
        episodes=1000,
        batch_size=32
    )
    
    # Plot the results
    plot_results(rewards, steps) 