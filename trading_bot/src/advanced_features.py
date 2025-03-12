import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib
import requests
import json

class AdvancedFeatures:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def calculate_advanced_indicators(self, df):
        """Calculate advanced technical indicators using TA-Lib"""
        # Trend Indicators
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['AROON_UP'], df['AROON_DOWN'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
        
        # Momentum Indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
        
        # Volatility Indicators
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Volume Indicators
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return df
    
    def add_market_regime_features(self, df):
        """Add market regime identification features"""
        # Volatility regime
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
        df['regime_volatility'] = pd.qcut(df['volatility'], q=3, labels=['low', 'medium', 'high'])
        
        # Trend regime
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
        
        return df
    
    def add_risk_metrics(self, df):
        """Add risk-related metrics"""
        # Calculate Value at Risk (VaR)
        returns = df['Close'].pct_change()
        df['VaR_95'] = returns.rolling(window=252).quantile(0.05)
        df['VaR_99'] = returns.rolling(window=252).quantile(0.01)
        
        # Calculate Conditional VaR (CVaR)
        df['CVaR_95'] = returns[returns <= df['VaR_95']].rolling(window=252).mean()
        
        return df
    
    def add_sentiment_features(self, df, symbol):
        """Add sentiment analysis features"""
        # Example of news sentiment (you would need actual API keys)
        try:
            news_sentiment = self._get_news_sentiment(symbol)
            df['news_sentiment'] = news_sentiment
        except:
            df['news_sentiment'] = 0
            
        # Social media sentiment (example)
        df['social_sentiment'] = self._get_social_sentiment(symbol)
        
        return df
    
    def _get_news_sentiment(self, symbol):
        """Get news sentiment from external API (example)"""
        # This is a placeholder - you would need to implement actual API calls
        return np.random.normal(0, 1)
    
    def _get_social_sentiment(self, symbol):
        """Get social media sentiment (example)"""
        # This is a placeholder - you would need to implement actual API calls
        return np.random.normal(0, 1)
    
    def add_order_book_features(self, df):
        """Add order book related features"""
        # Simulate order book features (in real implementation, you'd use actual order book data)
        df['bid_ask_spread'] = df['High'] - df['Low']
        df['market_depth'] = df['Volume'].rolling(window=5).mean()
        
        return df
    
    def add_correlation_features(self, df, market_index='SPY'):
        """Add correlation with market index"""
        # Download market index data
        market_data = yf.download(market_index, start=df.index[0], end=df.index[-1])['Close']
        
        # Calculate rolling correlation
        df['market_correlation'] = df['Close'].rolling(window=20).corr(market_data)
        
        return df

class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_optimal_position_size(self, current_price, volatility, portfolio_value):
        """Calculate optimal position size using Kelly Criterion"""
        edge = self._estimate_edge(current_price, volatility)
        win_rate = self._estimate_win_rate(current_price, volatility)
        
        kelly_fraction = (edge * win_rate - (1 - win_rate)) / edge
        kelly_fraction = max(0, min(kelly_fraction, 0.2))  # Cap at 20%
        
        return kelly_fraction * portfolio_value / current_price
    
    def _estimate_edge(self, current_price, volatility):
        """Estimate edge based on volatility and price action"""
        return volatility * 2  # Simplified example
    
    def _estimate_win_rate(self, current_price, volatility):
        """Estimate win rate based on historical data"""
        return 0.55  # Simplified example

class RiskManager:
    def __init__(self, max_drawdown=0.2, var_limit=0.05):
        self.max_drawdown = max_drawdown
        self.var_limit = var_limit
        
    def check_risk_limits(self, current_drawdown, var):
        """Check if position meets risk limits"""
        if current_drawdown > self.max_drawdown:
            return False
        if abs(var) > self.var_limit:
            return False
        return True
    
    def calculate_position_limits(self, portfolio_value, volatility):
        """Calculate position limits based on volatility"""
        max_position = portfolio_value * (0.1 / volatility)  # Higher volatility = smaller position
        return max_position 