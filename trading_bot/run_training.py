import os
import sys
import argparse

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_agent, plot_results

def main():
    parser = argparse.ArgumentParser(description='Train the Trading Bot')
    parser.add_argument('--symbol', type=str, default='MSFT',
                      help='Stock symbol to trade (default: MSFT)')
    parser.add_argument('--episodes', type=int, default=100,
                      help='Number of episodes to train (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--data-source', type=str, default='yfinance',
                      choices=['yfinance', 'tradingview'],
                      help='Data source to use (default: yfinance)')
    
    args = parser.parse_args()
    
    try:
        print("\nStarting Trading Bot training...")
        print(f"Symbol: {args.symbol}")
        print(f"Episodes: {args.episodes}")
        print(f"Batch size: {args.batch_size}")
        print(f"Data source: {args.data_source}")
        
        # Train the agent
        returns, portfolio_values = train_agent(
            symbol=args.symbol,
            episodes=args.episodes,
            batch_size=args.batch_size,
            data_source=args.data_source
        )
        
        # Plot the results
        if returns and portfolio_values:
            plot_results(returns, portfolio_values)
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 