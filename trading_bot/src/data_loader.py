import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ccxt
import requests
import ta
from tradingview_ta import TA_Handler, Interval
import time
import os
import json

class DataLoader:
    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize exchange connections
        self.binance = ccxt.binance()
        self.coinbase = ccxt.coinbase()
        
        # Define timeframes
        self.timeframes = {
            'short': '1d',
            'medium': '1w',
            'long': '1M'
        }
        
    def get_training_data(self, symbol, start_date=None, end_date=None, sources=None):
        """
        Get comprehensive training data from multiple sources
        
        Parameters:
        - symbol: Trading symbol (e.g., 'BTC/USDT' or 'AAPL')
        - start_date: Start date for historical data
        - end_date: End date for historical data
        - sources: List of data sources to use ['yfinance', 'ccxt', 'tradingview']
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if sources is None:
            sources = ['yfinance', 'ccxt', 'tradingview']
            
        cache_file = os.path.join(self.cache_dir, f"{symbol.replace('/', '_')}_{start_date}_{end_date}.csv")
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            print(f"Loading cached data for {symbol}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        combined_data = None
        
        for source in sources:
            try:
                if source == 'yfinance':
                    data = self._get_yfinance_data(symbol, start_date, end_date)
                elif source == 'ccxt':
                    data = self._get_ccxt_data(symbol, start_date, end_date)
                elif source == 'tradingview':
                    data = self._get_tradingview_data(symbol)
                
                if data is not None:
                    if combined_data is None:
                        combined_data = data
                    else:
                        # Merge data while handling conflicts
                        combined_data = self._merge_data_sources(combined_data, data)
            except Exception as e:
                print(f"Error fetching data from {source}: {str(e)}")
        
        if combined_data is not None:
            # Add technical indicators
            combined_data = self._add_technical_indicators(combined_data)
            
            # Add market sentiment data
            combined_data = self._add_market_sentiment(combined_data, symbol)
            
            # Cache the data
            combined_data.to_csv(cache_file)
            
            return combined_data
        else:
            raise Exception(f"Could not fetch data for {symbol} from any source")
    
    def _get_yfinance_data(self, symbol, start_date, end_date):
        """Get data from Yahoo Finance"""
        try:
            # Convert crypto symbol format if needed
            if '/' in symbol:
                symbol = symbol.replace('/', '-')
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty:
                print(f"Successfully downloaded Yahoo Finance data for {symbol}")
                return data
        except Exception as e:
            print(f"Yahoo Finance download failed: {str(e)}")
        return None
    
    def _get_ccxt_data(self, symbol, start_date, end_date):
        """Get data from cryptocurrency exchanges"""
        try:
            if '/' not in symbol:  # Only for crypto pairs
                return None
                
            # Try Binance first, then Coinbase
            for exchange in [self.binance, self.coinbase]:
                try:
                    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                    ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since)
                    
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    print(f"Successfully downloaded {exchange.id} data for {symbol}")
                    return df
                except:
                    continue
                    
        except Exception as e:
            print(f"CCXT download failed: {str(e)}")
        return None
    
    def _get_tradingview_data(self, symbol):
        """Get data from TradingView"""
        try:
            # Determine if it's crypto or stock
            is_crypto = '/' in symbol
            exchange = "BINANCE" if is_crypto else "NASDAQ"
            screener = "crypto" if is_crypto else "america"
            
            handler = TA_Handler(
                symbol=symbol.replace('/', ''),
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_DAY
            )
            
            analysis = handler.get_analysis()
            indicators = analysis.indicators
            
            # Create DataFrame with indicators
            df = pd.DataFrame([indicators])
            df.index = [datetime.now()]
            
            # Rename columns to match our format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            print(f"Successfully got TradingView data for {symbol}")
            return df
            
        except Exception as e:
            print(f"TradingView data fetch failed: {str(e)}")
            return None
    
    def _merge_data_sources(self, df1, df2):
        """Merge data from different sources with conflict resolution"""
        # Merge on index (timestamp)
        merged = pd.concat([df1, df2], axis=1, join='outer')
        
        # Resolve conflicts by taking average of non-null values
        for col in df1.columns.intersection(df2.columns):
            merged[col] = merged[[f"{col}_x", f"{col}_y"]].mean(axis=1)
            
        return merged
    
    def _add_technical_indicators(self, df):
        """Add comprehensive technical indicators"""
        try:
            # Trend Indicators
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            
            # MACD
            df['MACD'] = ta.trend.macd_diff(df['Close'])
            df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
            
            # Momentum Indicators
            df['RSI'] = ta.momentum.rsi(df['Close'])
            df['Stoch_RSI'] = ta.momentum.stochrsi(df['Close'])
            
            # Volatility Indicators
            df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
            df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # Volume Indicators
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['ADI'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
            
            return df
        except Exception as e:
            print(f"Error adding technical indicators: {str(e)}")
            return df
    
    def _add_market_sentiment(self, df, symbol):
        """Add market sentiment data"""
        try:
            # Add fear and greed index for crypto
            if '/' in symbol:
                fear_greed = self._get_crypto_fear_greed()
                if fear_greed is not None:
                    df['Fear_Greed_Index'] = fear_greed
            
            # Add market volatility index
            vix = self._get_vix_data()
            if vix is not None:
                df['VIX'] = vix
            
            return df
        except Exception as e:
            print(f"Error adding market sentiment: {str(e)}")
            return df
    
    def _get_crypto_fear_greed(self):
        """Get Crypto Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/?limit=0"
            response = requests.get(url)
            data = response.json()
            return int(data['data'][0]['value'])
        except:
            return None
    
    def _get_vix_data(self):
        """Get VIX (Volatility Index) data"""
        try:
            vix = yf.download('^VIX', period='1d')['Close'][-1]
            return vix
        except:
            return None
    
    def get_batch_training_data(self, symbols, start_date=None, end_date=None, sources=None):
        """Get training data for multiple symbols"""
        all_data = {}
        for symbol in symbols:
            try:
                data = self.get_training_data(symbol, start_date, end_date, sources)
                all_data[symbol] = data
                # Add delay to avoid rate limits
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
        return all_data
    
    def prepare_training_dataset(self, data, sequence_length=60):
        """Prepare data for training"""
        features = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            # Get sequence of data
            sequence = data.iloc[i:i+sequence_length]
            
            # Calculate returns
            future_return = (data.iloc[i+sequence_length]['Close'] - 
                           data.iloc[i+sequence_length-1]['Close']) / data.iloc[i+sequence_length-1]['Close']
            
            # Create target (0: Hold, 1: Buy, 2: Sell)
            if future_return > 0.01:  # 1% threshold for buy
                target = 1
            elif future_return < -0.01:  # -1% threshold for sell
                target = 2
            else:
                target = 0
            
            features.append(sequence.values)
            targets.append(target)
        
        return np.array(features), np.array(targets) 