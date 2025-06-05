#!/usr/bin/env python3
"""
CHART PATTERN ML MODEL
Cache data ile chart pattern recognition ve prediction
"""

import pandas as pd
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ChartPatternML:
    """Chart Pattern ML System"""
    
    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Pattern definitions
        self.patterns = {
            'bullish': ['bull_flag', 'ascending_triangle', 'cup_handle', 'golden_cross'],
            'bearish': ['bear_flag', 'descending_triangle', 'head_shoulders', 'death_cross'],
            'neutral': ['sideways', 'consolidation', 'range_bound']
        }
        
        # Load all available data
        self.load_cache_data()
        
    def load_cache_data(self):
        """Load all cache data"""
        print("📊 Loading cache data...")
        
        self.all_data = {}
        cache_files = glob.glob(f"{self.cache_dir}/*.csv")
        
        for file_path in cache_files:
            if 'USDT' in file_path or any(stock in file_path for stock in ['AAPL', 'MSFT']):
                symbol = os.path.basename(file_path).split('_')[0]
                try:
                    df = pd.read_csv(file_path)
                    if len(df) > 100:  # Valid data check
                        self.all_data[symbol] = df
                        print(f"  ✅ {symbol}: {len(df)} records")
                except Exception as e:
                    print(f"  ❌ {symbol}: {str(e)}")
        
        print(f"📊 Loaded {len(self.all_data)} symbols")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        df = df.copy()
        
        # Price features
        df['price_change'] = df['Close'].pct_change()
        df['volatility'] = df['Close'].rolling(20).std()
        df['price_momentum'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_price_trend'] = df['Volume'] * df['price_change']
        
        # Technical pattern features
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_cross'] = np.where(df['sma_5'] > df['sma_10'], 1, -1)
        
        # Bollinger Bands squeeze
        df['bb_squeeze'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
        
        # MACD features
        df['macd_signal_diff'] = df['MACD'] - df['MACD_Signal']
        df['macd_histogram'] = df['macd_signal_diff'].diff()
        
        # Support/Resistance levels
        df['resistance_touch'] = self._find_resistance_touches(df)
        df['support_touch'] = self._find_support_touches(df)
        
        # Chart patterns
        df['ascending_triangle'] = self._detect_ascending_triangle(df)
        df['descending_triangle'] = self._detect_descending_triangle(df)
        df['bull_flag'] = self._detect_bull_flag(df)
        df['bear_flag'] = self._detect_bear_flag(df)
        
        return df
    
    def _find_resistance_touches(self, df: pd.DataFrame, window=20) -> pd.Series:
        """Find resistance level touches"""
        resistance = df['High'].rolling(window).max()
        touches = (df['High'] >= resistance * 0.995).astype(int)
        return touches
    
    def _find_support_touches(self, df: pd.DataFrame, window=20) -> pd.Series:
        """Find support level touches"""
        support = df['Low'].rolling(window).min()
        touches = (df['Low'] <= support * 1.005).astype(int)
        return touches
    
    def _detect_ascending_triangle(self, df: pd.DataFrame, window=20) -> pd.Series:
        """Detect ascending triangle pattern"""
        # Simplified detection - rising lows, flat highs
        rising_lows = df['Low'].rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] > 0 if len(x) == window else False, raw=False)
        flat_highs = df['High'].rolling(window).apply(lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) < 0.001 if len(x) == window else False, raw=False)
        return ((rising_lows == True) & (flat_highs == True)).fillna(0).astype(int)
    
    def _detect_descending_triangle(self, df: pd.DataFrame, window=20) -> pd.Series:
        """Detect descending triangle pattern"""
        # Simplified detection - falling highs, flat lows
        falling_highs = df['High'].rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] < 0 if len(x) == window else False, raw=False)
        flat_lows = df['Low'].rolling(window).apply(lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) < 0.001 if len(x) == window else False, raw=False)
        return ((falling_highs == True) & (flat_lows == True)).fillna(0).astype(int)
    
    def _detect_bull_flag(self, df: pd.DataFrame, window=10) -> pd.Series:
        """Detect bull flag pattern"""
        # Strong uptrend followed by consolidation
        strong_up = df['Close'].pct_change(window) > 0.1
        consolidation = df['Close'].rolling(window).std() / df['Close'] < 0.05
        return (strong_up.shift(window) & consolidation).fillna(0).astype(int)
    
    def _detect_bear_flag(self, df: pd.DataFrame, window=10) -> pd.Series:
        """Detect bear flag pattern"""
        # Strong downtrend followed by consolidation
        strong_down = df['Close'].pct_change(window) < -0.1
        consolidation = df['Close'].rolling(window).std() / df['Close'] < 0.05
        return (strong_down.shift(window) & consolidation).fillna(0).astype(int)
    
    def create_prediction_labels(self, df: pd.DataFrame, forward_days=5) -> pd.Series:
        """Create prediction labels"""
        future_return = df['Close'].shift(-forward_days) / df['Close'] - 1
        
        labels = np.where(future_return > 0.02, 2,      # Strong Buy
                 np.where(future_return > 0.005, 1,     # Buy
                 np.where(future_return < -0.02, -2,    # Strong Sell
                 np.where(future_return < -0.005, -1,   # Sell
                         0))))                          # Hold
        
        return pd.Series(labels, index=df.index, name='prediction_label')
    
    def prepare_training_data(self):
        """Prepare training data from cache"""
        print("🔧 Preparing training data...")
        
        all_features = []
        all_labels = []
        
        for symbol, df in self.all_data.items():
            try:
                # Feature engineering
                df_features = self.engineer_features(df)
                
                # Create labels
                labels = self.create_prediction_labels(df_features)
                
                # Select feature columns
                feature_cols = [col for col in df_features.columns if col not in [
                    'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'BB_Middle', 'BB_Upper', 'BB_Lower'
                ]]
                
                # Remove rows with NaN
                valid_mask = ~(df_features[feature_cols].isna().any(axis=1) | labels.isna())
                if valid_mask.sum() > 50:  # Enough valid data
                    all_features.append(df_features.loc[valid_mask, feature_cols])
                    all_labels.append(labels[valid_mask])
                    
                    if not self.feature_columns:
                        self.feature_columns = feature_cols
                
            except Exception as e:
                print(f"  ❌ Error processing {symbol}: {str(e)}")
                continue
        
        if all_features:
            self.X = pd.concat(all_features, ignore_index=True)
            self.y = pd.concat(all_labels, ignore_index=True)
            print(f"  ✅ Training data: {len(self.X)} samples, {len(self.feature_columns)} features")
            return True
        else:
            print("  ❌ No valid training data")
            return False
    
    def train_models(self):
        """Train multiple ML models"""
        if not hasattr(self, 'X') or len(self.X) == 0:
            print("❌ No training data available")
            return
        
        print("🤖 Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # 1. Random Forest
        print("  🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        self.models['random_forest'] = rf_model
        print(f"    ✅ Random Forest Accuracy: {rf_accuracy:.3f}")
        
        # 2. LSTM Model (for time series)
        print("  🧠 Training LSTM...")
        try:
            lstm_model = self._build_lstm_model(X_train_scaled.shape[1])
            
            # Reshape for LSTM (samples, timesteps, features)
            X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
            
            # Convert labels to categorical
            y_train_cat = tf.keras.utils.to_categorical(y_train + 2, 5)  # -2 to 2 -> 0 to 4
            y_test_cat = tf.keras.utils.to_categorical(y_test + 2, 5)
            
            lstm_model.fit(X_train_lstm, y_train_cat, epochs=50, batch_size=32, 
                          validation_split=0.2, verbose=0)
            
            lstm_pred = lstm_model.predict(X_test_lstm, verbose=0)
            lstm_pred_classes = np.argmax(lstm_pred, axis=1) - 2  # Convert back to -2 to 2
            lstm_accuracy = accuracy_score(y_test, lstm_pred_classes)
            
            self.models['lstm'] = lstm_model
            print(f"    ✅ LSTM Accuracy: {lstm_accuracy:.3f}")
            
        except Exception as e:
            print(f"    ❌ LSTM training failed: {str(e)}")
        
        # 3. Gradient Boosting for numerical prediction
        print("  📈 Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_mse = np.mean((y_test - gb_pred) ** 2)
        
        self.models['gradient_boosting'] = gb_model
        print(f"    ✅ Gradient Boosting MSE: {gb_mse:.3f}")
        
        print("🎉 Model training completed!")
    
    def _build_lstm_model(self, input_features):
        """Build LSTM model"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(1, input_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(5, activation='softmax')  # 5 classes: -2, -1, 0, 1, 2
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def predict_from_chart(self, chart_data: pd.DataFrame) -> Dict:
        """Predict from chart data"""
        if not self.models:
            return {'error': 'Models not trained'}
        
        try:
            # Feature engineering
            chart_features = self.engineer_features(chart_data)
            
            # Get latest features
            latest_features = chart_features[self.feature_columns].iloc[-1:].fillna(0)
            
            # Scale features
            latest_scaled = self.scalers['standard'].transform(latest_features)
            
            predictions = {}
            
            # Random Forest prediction
            if 'random_forest' in self.models:
                rf_pred = self.models['random_forest'].predict(latest_scaled)[0]
                rf_proba = self.models['random_forest'].predict_proba(latest_scaled)[0]
                predictions['random_forest'] = {
                    'signal': rf_pred,
                    'confidence': max(rf_proba),
                    'probabilities': rf_proba.tolist()
                }
            
            # LSTM prediction
            if 'lstm' in self.models:
                lstm_input = latest_scaled.reshape(1, 1, latest_scaled.shape[1])
                lstm_pred = self.models['lstm'].predict(lstm_input, verbose=0)
                lstm_signal = np.argmax(lstm_pred[0]) - 2
                predictions['lstm'] = {
                    'signal': lstm_signal,
                    'confidence': max(lstm_pred[0]),
                    'probabilities': lstm_pred[0].tolist()
                }
            
            # Gradient Boosting prediction
            if 'gradient_boosting' in self.models:
                gb_pred = self.models['gradient_boosting'].predict(latest_scaled)[0]
                predictions['gradient_boosting'] = {
                    'signal': round(gb_pred),
                    'confidence': min(abs(gb_pred), 1.0),
                    'raw_value': gb_pred
                }
            
            # Ensemble prediction
            ensemble_signal = self._ensemble_prediction(predictions)
            
            return {
                'ensemble': ensemble_signal,
                'individual_models': predictions,
                'feature_importance': self._get_feature_importance()
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _ensemble_prediction(self, predictions: Dict) -> Dict:
        """Ensemble prediction from multiple models"""
        signals = []
        confidences = []
        
        for model_name, pred in predictions.items():
            signals.append(pred['signal'])
            confidences.append(pred['confidence'])
        
        if not signals:
            return {'signal': 0, 'confidence': 0.0, 'action': 'HOLD'}
        
        # Weighted average based on confidence
        weights = np.array(confidences) / sum(confidences)
        ensemble_signal = np.average(signals, weights=weights)
        ensemble_confidence = np.mean(confidences)
        
        # Convert to action
        if ensemble_signal >= 1.5:
            action = 'STRONG_BUY'
        elif ensemble_signal >= 0.5:
            action = 'BUY'
        elif ensemble_signal <= -1.5:
            action = 'STRONG_SELL'
        elif ensemble_signal <= -0.5:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'signal': ensemble_signal,
            'confidence': ensemble_confidence,
            'action': action,
            'individual_signals': signals,
            'model_count': len(signals)
        }
    
    def _get_feature_importance(self) -> Dict:
        """Get feature importance from Random Forest"""
        if 'random_forest' not in self.models:
            return {}
        
        importance = self.models['random_forest'].feature_importances_
        feature_imp = dict(zip(self.feature_columns, importance))
        
        # Sort by importance
        return dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))
    
    def save_models(self, save_dir='models'):
        """Save trained models"""
        import joblib
        
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == 'lstm':
                model.save(f"{save_dir}/{model_name}_model.h5")
            else:
                joblib.dump(model, f"{save_dir}/{model_name}_model.pkl")
        
        # Save scalers
        joblib.dump(self.scalers, f"{save_dir}/scalers.pkl")
        joblib.dump(self.feature_columns, f"{save_dir}/feature_columns.pkl")
        
        print(f"💾 Models saved to {save_dir}/")

def train_chart_pattern_ml():
    """Train chart pattern ML models"""
    print("🚀 CHART PATTERN ML TRAINING")
    print("="*50)
    
    # Initialize ML system
    ml_system = ChartPatternML()
    
    # Prepare training data
    if ml_system.prepare_training_data():
        # Train models
        ml_system.train_models()
        
        # Save models
        ml_system.save_models()
        
        print("\n🎉 Chart Pattern ML training completed!")
        return ml_system
    else:
        print("❌ Training failed - no valid data")
        return None

if __name__ == "__main__":
    train_chart_pattern_ml() 