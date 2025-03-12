import os
import argparse
from datetime import datetime, timedelta
from src.data_loader import DataLoader
from src.train import train_agent
import json

def main():
    parser = argparse.ArgumentParser(description='Train Trading Bot with comprehensive data')
    parser.add_argument('--symbols', type=str, nargs='+',
                      help='Symbols to train on (e.g., BTC/USDT AAPL)')
    parser.add_argument('--start-date', type=str,
                      help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                      help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--sources', type=str, nargs='+',
                      choices=['yfinance', 'ccxt', 'tradingview'],
                      default=['yfinance', 'ccxt', 'tradingview'],
                      help='Data sources to use')
    parser.add_argument('--episodes', type=int, default=200,
                      help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--sequence-length', type=int, default=60,
                      help='Sequence length for training data')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # If no symbols specified, use default set
    if not args.symbols:
        args.symbols = ['BTC/USDT', 'ETH/USDT', 'AAPL', 'MSFT']
    
    # Set up output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training configuration
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\nStarting comprehensive training process...")
    print(f"Symbols: {args.symbols}")
    print(f"Data sources: {args.sources}")
    print(f"Training period: {args.start_date} to {args.end_date}")
    
    # Train for each symbol
    for symbol in args.symbols:
        try:
            print(f"\nProcessing {symbol}...")
            
            # Get training data
            data = data_loader.get_training_data(
                symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                sources=args.sources
            )
            
            if data is not None:
                print(f"Successfully loaded data for {symbol}")
                print(f"Data shape: {data.shape}")
                print("Features available:", list(data.columns))
                
                # Prepare sequences for training
                features, targets = data_loader.prepare_training_dataset(
                    data,
                    sequence_length=args.sequence_length
                )
                
                print(f"Prepared {len(features)} training sequences")
                
                # Train the agent
                print(f"\nTraining agent for {symbol}...")
                returns, portfolio_values = train_agent(
                    symbol=symbol,
                    episodes=args.episodes,
                    batch_size=args.batch_size,
                    data_source='prepared_data',
                    prepared_data=(features, targets)
                )
                
                # Save training results
                symbol_dir = os.path.join(output_dir, symbol.replace('/', '_'))
                os.makedirs(symbol_dir, exist_ok=True)
                
                results = {
                    'symbol': symbol,
                    'returns': returns,
                    'portfolio_values': portfolio_values,
                    'training_config': config
                }
                
                with open(os.path.join(symbol_dir, 'training_results.json'), 'w') as f:
                    json.dump(results, f, indent=4)
                
                print(f"Training completed for {symbol}")
                print(f"Results saved to {symbol_dir}")
                
            else:
                print(f"No data available for {symbol}, skipping...")
                
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    print("\nTraining process completed!")
    print(f"All models and results saved to {output_dir}")

if __name__ == "__main__":
    main() 