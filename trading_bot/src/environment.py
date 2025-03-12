import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import ta  # Technical Analysis library

class TradingEnv:
    def __init__(self, symbol, start_date, end_date, initial_balance=10000, data_source='yfinance'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.data_source = data_source
        
        # Download data
        self.data = self._get_data()
        self.reset()
        
        # Action space: 0: hold, 1: buy, 2: sell
        self.action_space = 3
        # State space: OHLCV + technical indicators
        self.observation_space = 10  
        
    def _get_data(self):
        """Download and prepare data with error handling"""
        try:
            df = None
            if self.data_source == 'yfinance':
                df = self._get_yfinance_data()
            elif self.data_source == 'tradingview':
                df = self._get_tradingview_data()
            
            if df is None or df.empty:
                print("Falling back to dummy data for testing")
                return self._get_dummy_data()
            
            # Calculate technical indicators using the ta library
            df = self._add_technical_indicators(df)
            
            # Forward fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            if len(df) < 2:
                raise Exception("Insufficient data points")
                
            return df
            
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            return self._get_dummy_data()
    
    def _get_yfinance_data(self):
        """Get data from Yahoo Finance"""
        try:
            # Add extra days to account for holidays/weekends
            start = (datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            df = yf.download(self.symbol, start=start, end=self.end_date, progress=False)
            if not df.empty:
                print(f"Successfully downloaded data for {self.symbol} from Yahoo Finance")
                return df
        except Exception as e:
            print(f"Yahoo Finance download failed: {str(e)}")
        return None
    
    def _get_tradingview_data(self):
        """Get data from TradingView"""
        try:
            # Note: This requires the tradingview-ta package
            from tradingview_ta import TA_Handler, Interval
            
            handler = TA_Handler(
                symbol=self.symbol,
                screener="america" if ":" not in self.symbol else "crypto",
                exchange="NASDAQ" if ":" not in self.symbol else "BINANCE",
                interval=Interval.INTERVAL_1_DAY
            )
            
            analysis = handler.get_analysis()
            indicators = analysis.indicators
            
            # Create DataFrame with available data
            current_date = datetime.now().strftime('%Y-%m-%d')
            df = pd.DataFrame([indicators], index=[current_date])
            
            # Rename columns to match our format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            print(f"Successfully got TradingView data for {self.symbol}")
            return df
            
        except Exception as e:
            print(f"TradingView data fetch failed: {str(e)}")
            return None
    
    def _add_technical_indicators(self, df):
        """Add technical indicators using the ta library"""
        try:
            # Trend indicators
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            
            # Momentum indicators
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['MACD'] = ta.trend.macd_diff(df['Close'])
            
            # Volatility indicators
            df['BB_UPPER'] = ta.volatility.bollinger_hband(df['Close'])
            df['BB_LOWER'] = ta.volatility.bollinger_lband(df['Close'])
            
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def _get_dummy_data(self):
        """Create dummy data for testing"""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        df = pd.DataFrame(index=dates)
        df['Open'] = np.random.randn(len(dates)).cumsum() + 100
        df['High'] = df['Open'] + abs(np.random.randn(len(dates)))
        df['Low'] = df['Open'] - abs(np.random.randn(len(dates)))
        df['Close'] = df['Open'] + np.random.randn(len(dates))
        df['Volume'] = np.random.randint(1000000, 10000000, len(dates))
        df = self._add_technical_indicators(df)
        return df
    
    def reset(self):
        """Reset the trading environment"""
        self.current_step = 20  # Start after warmup period for indicators
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_value = self.balance
        return self._get_state()
    
    def _get_state(self):
        """Get current state of the environment"""
        current_data = self.data.iloc[self.current_step]
        
        state = np.array([
            current_data['Open'],
            current_data['High'],
            current_data['Low'],
            current_data['Close'],
            current_data['Volume'],
            current_data['SMA_20'],
            current_data['EMA_12'],
            current_data['EMA_26'],
            current_data['RSI'],
            current_data['MACD']
        ])
        
        # Normalize state
        return state / np.max(np.abs(state) + 1e-8)  # Add small epsilon to avoid division by zero
    
    def step(self, action):
        """Take a step in the environment"""
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute trade
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            self.shares_held += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        new_price = self.data.iloc[self.current_step]['Close']
        new_value = self.balance + self.shares_held * new_price
        reward = new_value - self.current_value
        self.current_value = new_value
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done
    
    def render(self):
        """Print current portfolio status"""
        current_price = self.data.iloc[self.current_step]['Close']
        total_value = self.balance + self.shares_held * current_price
        print(f"\nStep: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares held: {self.shares_held}")
        print(f"Current share price: ${current_price:.2f}")
        print(f"Total value: ${total_value:.2f}")
        print("------------------------") 