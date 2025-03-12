import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from tradingview_ta import TA_Handler, Interval
import json

class TradingAnalyzer:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        # Define supported assets
        self.supported_assets = {
            'crypto': [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT'
            ],
            'stocks': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
                'META', 'TSLA', 'JPM', 'V', 'WMT'
            ],
            'commodities': [
                'GC=F',  # Gold
                'SI=F',  # Silver
                'CL=F',  # Crude Oil
                'NG=F',  # Natural Gas
                'PL=F'   # Platinum
            ]
        }
    
    def load_model(self, model_path):
        """Load a saved trading model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            
            # Load training state if available
            state_path = os.path.join(os.path.dirname(model_path), 'training_state.npz')
            if os.path.exists(state_path):
                self.training_state = np.load(state_path)
                print("Training state loaded")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def get_tradingview_signals(self, symbol, exchange="BINANCE", screener="crypto"):
        """Get trading signals from TradingView"""
        try:
            handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_DAY
            )
            analysis = handler.get_analysis()
            
            # Get recommendation
            recommendation = analysis.summary['RECOMMENDATION']
            
            # Get indicators
            indicators = analysis.indicators
            
            # Create detailed analysis
            analysis_details = {
                'recommendation': recommendation,
                'rsi': indicators['RSI'],
                'macd': {
                    'macd': indicators['MACD.macd'],
                    'signal': indicators['MACD.signal']
                },
                'ma': {
                    'sma20': indicators['SMA20'],
                    'sma50': indicators['SMA50'],
                    'sma200': indicators['SMA200']
                },
                'oscillators': analysis.oscillators,
                'moving_averages': analysis.moving_averages
            }
            
            return analysis_details
            
        except Exception as e:
            print(f"Error getting TradingView signals: {str(e)}")
            return None
    
    def get_recommendation(self, symbol, data_source='both'):
        """Get trading recommendation for a symbol"""
        try:
            recommendations = {}
            
            # Get TradingView signals
            if data_source in ['tradingview', 'both']:
                # Determine if it's crypto or stock
                is_crypto = '/' in symbol
                exchange = "BINANCE" if is_crypto else "NASDAQ"
                screener = "crypto" if is_crypto else "america"
                
                tv_signals = self.get_tradingview_signals(
                    symbol.replace('/', ''), 
                    exchange=exchange,
                    screener=screener
                )
                if tv_signals:
                    recommendations['tradingview'] = {
                        'action': tv_signals['recommendation'],
                        'confidence': self._calculate_confidence(tv_signals),
                        'details': tv_signals
                    }
            
            # Get model prediction if model is loaded
            if self.model and data_source in ['model', 'both']:
                # Get latest market data
                market_data = self._get_latest_market_data(symbol)
                if market_data is not None:
                    state = self._prepare_state(market_data)
                    action = self._predict_action(state)
                    recommendations['model'] = {
                        'action': self._action_to_string(action),
                        'confidence': self._calculate_model_confidence(state),
                        'details': {
                            'state': state.tolist(),
                            'raw_prediction': self.model.predict(state.reshape(1, -1), verbose=0).tolist()
                        }
                    }
            
            # Combine recommendations
            final_recommendation = self._combine_recommendations(recommendations)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'recommendations': recommendations,
                'final_recommendation': final_recommendation
            }
            
        except Exception as e:
            print(f"Error getting recommendation: {str(e)}")
            return None
    
    def _calculate_confidence(self, signals):
        """Calculate confidence score from TradingView signals"""
        try:
            # Calculate confidence based on multiple indicators
            confidence = 0
            
            # RSI confidence
            rsi = signals['rsi']
            if rsi < 30:
                confidence += 0.2  # Oversold
            elif rsi > 70:
                confidence += 0.2  # Overbought
            
            # MACD confidence
            macd = signals['macd']['macd']
            macd_signal = signals['macd']['signal']
            if abs(macd - macd_signal) > 1:
                confidence += 0.2
            
            # Moving average confidence
            ma_data = signals['ma']
            if ma_data['sma20'] > ma_data['sma50']:
                confidence += 0.2
            if ma_data['sma50'] > ma_data['sma200']:
                confidence += 0.2
            
            # Add oscillator and MA consensus
            oscillator_buy = signals['oscillators']['BUY']
            oscillator_sell = signals['oscillators']['SELL']
            ma_buy = signals['moving_averages']['BUY']
            ma_sell = signals['moving_averages']['SELL']
            
            total_signals = oscillator_buy + oscillator_sell + ma_buy + ma_sell
            if total_signals > 0:
                confidence += 0.2 * (max(oscillator_buy, oscillator_sell) + 
                                   max(ma_buy, ma_sell)) / total_signals
            
            return min(confidence, 1.0)
            
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _calculate_model_confidence(self, state):
        """Calculate model's confidence in its prediction"""
        try:
            predictions = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            # Calculate softmax probabilities
            exp_preds = np.exp(predictions - np.max(predictions))
            probabilities = exp_preds / exp_preds.sum()
            # Return the highest probability as confidence
            return float(np.max(probabilities))
        except Exception as e:
            print(f"Error calculating model confidence: {str(e)}")
            return 0.5
    
    def _action_to_string(self, action):
        """Convert numeric action to string"""
        actions = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return actions.get(action, 'HOLD')
    
    def _predict_action(self, state):
        """Get model's action prediction"""
        predictions = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.argmax(predictions)
    
    def _combine_recommendations(self, recommendations):
        """Combine different recommendations into a final decision"""
        try:
            final_confidence = 0
            action_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'NEUTRAL': 0}
            
            for source, rec in recommendations.items():
                action = rec['action']
                confidence = rec['confidence']
                
                # Normalize TradingView recommendations
                if action in ['STRONG_BUY', 'BUY']:
                    action = 'BUY'
                elif action in ['STRONG_SELL', 'SELL']:
                    action = 'SELL'
                elif action in ['NEUTRAL']:
                    action = 'HOLD'
                
                action_scores[action] += confidence
                final_confidence += confidence
            
            # Get action with highest score
            final_action = max(action_scores.items(), key=lambda x: x[1])[0]
            
            # Calculate final confidence
            if len(recommendations) > 0:
                final_confidence /= len(recommendations)
            
            return {
                'action': final_action,
                'confidence': final_confidence,
                'action_scores': action_scores
            }
            
        except Exception as e:
            print(f"Error combining recommendations: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.5, 'action_scores': action_scores}
    
    def _get_latest_market_data(self, symbol):
        """Get latest market data for the symbol"""
        try:
            # For crypto
            if '/' in symbol:
                return self._get_latest_crypto_data(symbol)
            # For stocks
            else:
                return self._get_latest_stock_data(symbol)
        except Exception as e:
            print(f"Error getting market data: {str(e)}")
            return None
    
    def _prepare_state(self, market_data):
        """Prepare market data as state input for the model"""
        try:
            # Extract relevant features
            state = np.array([
                market_data['Open'],
                market_data['High'],
                market_data['Low'],
                market_data['Close'],
                market_data['Volume'],
                market_data['SMA20'],
                market_data['EMA12'],
                market_data['EMA26'],
                market_data['RSI'],
                market_data['MACD']
            ])
            
            # Normalize state
            return state / (np.max(np.abs(state)) + 1e-8)
            
        except Exception as e:
            print(f"Error preparing state: {str(e)}")
            return None
    
    def list_available_assets(self):
        """Return list of supported assets"""
        return self.supported_assets
    
    def save_recommendation(self, recommendation, filepath):
        """Save recommendation to a file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(recommendation, f, indent=4)
            print(f"Recommendation saved to {filepath}")
        except Exception as e:
            print(f"Error saving recommendation: {str(e)}")
    
    def analyze_portfolio(self, symbols):
        """Analyze multiple assets and provide portfolio recommendations"""
        portfolio_analysis = []
        
        for symbol in symbols:
            recommendation = self.get_recommendation(symbol)
            if recommendation:
                portfolio_analysis.append(recommendation)
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis': portfolio_analysis,
            'summary': self._generate_portfolio_summary(portfolio_analysis)
        }
    
    def _generate_portfolio_summary(self, portfolio_analysis):
        """Generate summary of portfolio analysis"""
        summary = {
            'total_assets': len(portfolio_analysis),
            'recommendations': {
                'BUY': 0,
                'SELL': 0,
                'HOLD': 0
            },
            'high_confidence_signals': [],
            'average_confidence': 0
        }
        
        for analysis in portfolio_analysis:
            action = analysis['final_recommendation']['action']
            confidence = analysis['final_recommendation']['confidence']
            
            # Count recommendations
            summary['recommendations'][action] += 1
            
            # Track high confidence signals
            if confidence > 0.7:
                summary['high_confidence_signals'].append({
                    'symbol': analysis['symbol'],
                    'action': action,
                    'confidence': confidence
                })
            
            summary['average_confidence'] += confidence
        
        if len(portfolio_analysis) > 0:
            summary['average_confidence'] /= len(portfolio_analysis)
        
        return summary 