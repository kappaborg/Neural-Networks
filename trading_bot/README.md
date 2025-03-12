# Trading Bot Documentation

A sophisticated trading bot that combines deep reinforcement learning with technical analysis to provide trading recommendations for cryptocurrencies, stocks, and commodities.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Available Assets](#available-assets)
- [Usage Scenarios](#usage-scenarios)
- [Training the Model](#training-the-model)
- [Analyzing Trades](#analyzing-trades)
- [Model Architecture](#model-architecture)
- [Technical Indicators](#technical-indicators)

## Features

- Deep Q-Learning (DQN) with experience replay
- Integration with TradingView signals
- Real-time market data analysis
- Portfolio management
- Multiple asset classes support
- Confidence-based recommendations
- Technical indicator analysis
- Model persistence and state management
- Detailed performance metrics
- Customizable trading strategies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading_bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Available Assets

### Cryptocurrencies
- BTC/USDT (Bitcoin)
- ETH/USDT (Ethereum)
- BNB/USDT (Binance Coin)
- ADA/USDT (Cardano)
- SOL/USDT (Solana)
- DOT/USDT (Polkadot)
- DOGE/USDT (Dogecoin)
- AVAX/USDT (Avalanche)
- MATIC/USDT (Polygon)
- LINK/USDT (Chainlink)

### Stocks
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- AMZN (Amazon)
- NVDA (NVIDIA)
- META (Meta)
- TSLA (Tesla)
- JPM (JPMorgan)
- V (Visa)
- WMT (Walmart)

### Commodities
- GC=F (Gold)
- SI=F (Silver)
- CL=F (Crude Oil)
- NG=F (Natural Gas)
- PL=F (Platinum)

## Usage Scenarios

### 1. Training the Model

#### Basic Training
```bash
python run_training.py
```

#### Custom Training Parameters
```bash
python run_training.py --symbol MSFT --episodes 200 --batch-size 64
```

#### Training with Different Data Sources
```bash
# Using Yahoo Finance
python run_training.py --data-source yfinance --symbol AAPL

# Using TradingView
python run_training.py --data-source tradingview --symbol BTC/USDT
```

### 2. Analyzing Trades

#### List Available Assets
```bash
python analyze_trades.py --list-assets
```

#### Analyze Single Asset
```bash
# Basic analysis
python analyze_trades.py --symbol BTC/USDT

# Analysis with saved model
python analyze_trades.py --model checkpoints/TradingBot/latest/model.h5 --symbol ETH/USDT

# Save analysis to file
python analyze_trades.py --symbol AAPL --output analysis/apple_analysis.json
```

#### Portfolio Analysis
```bash
# Analyze entire portfolio
python analyze_trades.py --portfolio

# Analyze portfolio with saved model
python analyze_trades.py --model checkpoints/TradingBot/latest/model.h5 --portfolio

# Save portfolio analysis
python analyze_trades.py --portfolio --output analysis/portfolio_analysis.json
```

### 3. Real-time Trading Signals

#### Get Immediate Recommendations
```bash
# For cryptocurrency
python analyze_trades.py --symbol BTC/USDT

# For stocks
python analyze_trades.py --symbol AAPL

# For commodities
python analyze_trades.py --symbol GC=F
```

### 4. Model Management

#### Save Model State
```bash
# Models are automatically saved during training
# Default location: checkpoints/TradingBot/<timestamp>/model_episode_<N>.h5
```

#### Load Saved Model
```bash
python analyze_trades.py --model checkpoints/TradingBot/<timestamp>/model_episode_100.h5 --symbol MSFT
```

## Model Architecture

The trading bot uses a sophisticated DQN architecture:
- Input Layer: 10 features (OHLCV + technical indicators)
- Hidden Layers:
  - Dense Layer (128 units) with Batch Normalization
  - Dense Layer (128 units) with Batch Normalization
  - LSTM Layer (64 units)
  - Dueling DQN architecture (Value and Advantage streams)
- Output Layer: 3 actions (Buy, Sell, Hold)

### Advanced Features
- Prioritized Experience Replay
- Double DQN implementation
- Soft target network updates
- Mixed precision training (for M1/M2 Macs)
- Batch processing optimization

## Technical Indicators

The bot analyzes the following technical indicators:
1. Trend Indicators
   - Simple Moving Average (SMA20, SMA50, SMA200)
   - Exponential Moving Average (EMA12, EMA26)
   - MACD (Moving Average Convergence Divergence)

2. Momentum Indicators
   - Relative Strength Index (RSI)
   - MACD Histogram

3. Volatility Indicators
   - Bollinger Bands (Upper and Lower)

4. Additional TradingView Signals
   - Oscillator Signals
   - Moving Average Consensus
   - Overall Recommendation

## Confidence Metrics

The bot provides confidence scores based on:
1. Technical Analysis
   - RSI levels (oversold/overbought)
   - MACD crossovers
   - Moving average alignments
   - Oscillator consensus
   
2. Model Predictions
   - Action probabilities
   - Historical accuracy
   - Prediction stability

3. Combined Metrics
   - Weighted recommendation scores
   - Multi-source consensus
   - Historical performance

## Output Formats

### Single Asset Analysis
```json
{
    "symbol": "BTC/USDT",
    "timestamp": "2024-03-15 12:34:56",
    "recommendations": {
        "tradingview": {
            "action": "BUY",
            "confidence": 0.85,
            "details": {...}
        },
        "model": {
            "action": "BUY",
            "confidence": 0.78,
            "details": {...}
        }
    },
    "final_recommendation": {
        "action": "BUY",
        "confidence": 0.815,
        "action_scores": {...}
    }
}
```

### Portfolio Analysis
```json
{
    "timestamp": "2024-03-15 12:34:56",
    "analysis": [...],
    "summary": {
        "total_assets": 25,
        "recommendations": {
            "BUY": 10,
            "SELL": 5,
            "HOLD": 10
        },
        "high_confidence_signals": [...],
        "average_confidence": 0.75
    }
}
```

## Best Practices

1. Training
   - Start with default parameters
   - Use at least 100 episodes for initial training
   - Monitor loss and reward curves
   - Save models periodically

2. Analysis
   - Use both model and TradingView signals
   - Focus on high confidence signals (>0.7)
   - Consider multiple timeframes
   - Validate signals across different indicators

3. Portfolio Management
   - Diversify across asset classes
   - Monitor high confidence signals
   - Regular portfolio rebalancing
   - Risk management based on confidence scores 