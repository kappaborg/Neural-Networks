#!/usr/bin/env python3
"""
YOLO INTEGRATED TRADING SYSTEM
Advanced trading system combining YOLO symbol detection with cache data analysis

Pipeline:
1. YOLO → Detect symbol with bounding box
2. Symbol → Load corresponding cache data
3. Chart Pattern + Historical Data → AI Prediction
4. Real-time visualization with high accuracy
"""

import cv2
import numpy as np
import pandas as pd
import time
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import custom modules
from yolo_symbol_detector import YOLOSymbolDetector, SymbolDetection
from chart_pattern_ml import ChartPatternML
from ultra_fast_ocr import UltraFastOCR

@dataclass
class IntegratedPrediction:
    """Integrated prediction result"""
    symbol: str
    symbol_confidence: float
    price: Optional[float]
    price_confidence: float
    trading_signal: str
    signal_confidence: float
    prediction_source: str  # 'yolo_ml', 'fallback_ocr', 'cache_only'
    data_points: int
    chart_patterns: List[str]
    technical_indicators: Dict[str, float]
    processing_time_ms: float
    bbox: Optional[Tuple[int, int, int, int]]

class YOLOIntegratedTradingSystem:
    """YOLO-powered trading system with cache data integration"""
    
    def __init__(self):
        print("🚀 YOLO INTEGRATED TRADING SYSTEM INITIALIZING...")
        
        # Initialize components
        self.yolo_detector = YOLOSymbolDetector()
        self.ml_system = None
        try:
            self.ml_system = ChartPatternML()
        except:
            print("⚠️ ChartPatternML not available")
        
        self.ultra_ocr = UltraFastOCR()  # Fallback
        
        # Load cache data
        self.cache_data = self._load_cache_data()
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'yolo_successful': 0,
            'ocr_fallback_used': 0,
            'cache_data_used': 0,
            'avg_processing_time': 0.0,
            'symbol_accuracy': {},
            'prediction_accuracy': {}
        }
        
        print(f"✅ YOLO INTEGRATED SYSTEM READY!")
        print(f"   • YOLO Model: {'✅ Active' if self.yolo_detector.model else '❌ Fallback'}")
        print(f"   • Cache Data: {len(self.cache_data)} symbols loaded")
        print(f"   • ML Models: {'✅ Active' if self.ml_system else '❌ Disabled'}")
    
    def _load_cache_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical cache data for all symbols"""
        cache_data = {}
        
        try:
            cache_dir = "data/cache"
            if not os.path.exists(cache_dir):
                print("⚠️ Cache directory not found")
                return {}
            
            csv_files = glob.glob(os.path.join(cache_dir, "*.csv"))
            
            for file_path in csv_files:
                try:
                    filename = os.path.basename(file_path)
                    symbol = filename.split('_')[0]  # Extract symbol from filename
                    
                    df = pd.read_csv(file_path)
                    
                    # Standardize columns
                    column_mapping = {
                        'timestamp': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }
                    df = df.rename(columns=column_mapping)
                    
                    # Convert timestamp
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    cache_data[symbol] = df
                    print(f"  ✅ {symbol}: {len(df)} records loaded")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {filename}: {str(e)}")
            
            print(f"📊 Loaded {len(cache_data)} symbols from cache")
            return cache_data
            
        except Exception as e:
            print(f"❌ Cache loading error: {str(e)}")
            return {}
    
    def analyze_frame_integrated(self, frame: np.ndarray) -> IntegratedPrediction:
        """Complete integrated analysis: YOLO → Cache Data → Prediction"""
        start_time = time.time()
        
        try:
            # Phase 1: YOLO Symbol Detection (Primary)
            yolo_detection = self.yolo_detector.get_best_symbol(frame)
            
            symbol = 'UNKNOWN'
            symbol_confidence = 0.0
            bbox = None
            prediction_source = 'fallback_ocr'
            
            if yolo_detection and yolo_detection.confidence > 0.6:
                # YOLO successful
                symbol = yolo_detection.symbol
                symbol_confidence = yolo_detection.confidence
                bbox = yolo_detection.bbox
                prediction_source = 'yolo_ml'
                self.performance_stats['yolo_successful'] += 1
                
                print(f"🎯 YOLO Detection: {symbol} ({symbol_confidence:.2f}) at {bbox}")
            else:
                # Fallback to OCR
                ocr_result = self.ultra_ocr.extract_combo_fast(frame)
                symbol = ocr_result['symbol']
                symbol_confidence = ocr_result['symbol_confidence']
                prediction_source = 'fallback_ocr'
                self.performance_stats['ocr_fallback_used'] += 1
                
                print(f"🔄 OCR Fallback: {symbol} ({symbol_confidence:.2f})")
            
            # Phase 2: Price Extraction (Always try)
            price_result = self.ultra_ocr.extract_price_fast(frame)
            price = float(price_result.text) if price_result.text and price_result.text.replace('.', '').replace(',', '').isdigit() else None
            price_confidence = price_result.confidence
            
            # Phase 3: Cache Data Analysis
            data_points = 0
            chart_patterns = []
            technical_indicators = {}
            trading_signal = 'HOLD'
            signal_confidence = 0.3
            
            if symbol in self.cache_data:
                # Use real cache data for prediction
                symbol_data = self.cache_data[symbol]
                data_points = len(symbol_data)
                
                # Get latest 100 records for analysis
                recent_data = symbol_data.tail(100).copy()
                
                # Calculate technical indicators
                technical_indicators = self._calculate_technical_indicators(recent_data)
                
                # Detect chart patterns
                chart_patterns = self._detect_chart_patterns(recent_data)
                
                # Generate trading signal
                signal_result = self._generate_trading_signal(recent_data, technical_indicators, chart_patterns)
                trading_signal = signal_result['signal']
                signal_confidence = signal_result['confidence']
                
                prediction_source = 'yolo_ml' if prediction_source == 'yolo_ml' else 'cache_data'
                self.performance_stats['cache_data_used'] += 1
                
                print(f"📊 Cache Analysis: {data_points} records, Signal: {trading_signal} ({signal_confidence:.2f})")
            
            # Phase 4: ML Enhancement (if available)
            if self.ml_system and symbol in self.cache_data:
                try:
                    ml_prediction = self.ml_system.predict_from_chart(self.cache_data[symbol].tail(30))
                    if ml_prediction and ml_prediction.get('confidence', 0) > signal_confidence:
                        trading_signal = ml_prediction['prediction']
                        signal_confidence = ml_prediction['confidence']
                        print(f"🤖 ML Enhancement: {trading_signal} ({signal_confidence:.2f})")
                except Exception as e:
                    print(f"⚠️ ML prediction error: {str(e)}")
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_performance_stats(symbol, processing_time, prediction_source)
            
            return IntegratedPrediction(
                symbol=symbol,
                symbol_confidence=symbol_confidence,
                price=price,
                price_confidence=price_confidence,
                trading_signal=trading_signal,
                signal_confidence=signal_confidence,
                prediction_source=prediction_source,
                data_points=data_points,
                chart_patterns=chart_patterns,
                technical_indicators=technical_indicators,
                processing_time_ms=processing_time,
                bbox=bbox
            )
            
        except Exception as e:
            print(f"❌ Integrated analysis error: {str(e)}")
            return IntegratedPrediction(
                symbol='ERROR',
                symbol_confidence=0.0,
                price=None,
                price_confidence=0.0,
                trading_signal='HOLD',
                signal_confidence=0.0,
                prediction_source='error',
                data_points=0,
                chart_patterns=[],
                technical_indicators={},
                processing_time_ms=(time.time() - start_time) * 1000,
                bbox=None
            )
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators from cache data"""
        try:
            indicators = {}
            
            if len(df) < 20:
                return indicators
            
            # Simple Moving Averages
            indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else indicators['sma_20']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Price momentum
            indicators['price_change_1d'] = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            indicators['price_change_7d'] = ((df['close'].iloc[-1] - df['close'].iloc[-8]) / df['close'].iloc[-8]) * 100 if len(df) >= 8 else 0
            
            # Volume trend
            indicators['volume_avg'] = df['volume'].rolling(10).mean().iloc[-1] if 'volume' in df.columns else 0
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_avg'] if indicators['volume_avg'] > 0 and 'volume' in df.columns else 1
            
            return indicators
            
        except Exception as e:
            print(f"⚠️ Technical indicators error: {str(e)}")
            return {}
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect chart patterns in price data"""
        try:
            patterns = []
            
            if len(df) < 10:
                return patterns
            
            # Simple pattern detection
            recent_prices = df['close'].tail(10).values
            
            # Trend detection
            if recent_prices[-1] > recent_prices[0]:
                if all(recent_prices[i] >= recent_prices[i-1] for i in range(1, len(recent_prices))):
                    patterns.append('STRONG_UPTREND')
                else:
                    patterns.append('UPTREND')
            elif recent_prices[-1] < recent_prices[0]:
                if all(recent_prices[i] <= recent_prices[i-1] for i in range(1, len(recent_prices))):
                    patterns.append('STRONG_DOWNTREND')
                else:
                    patterns.append('DOWNTREND')
            else:
                patterns.append('SIDEWAYS')
            
            # Volatility
            price_std = np.std(recent_prices)
            price_mean = np.mean(recent_prices)
            volatility = (price_std / price_mean) * 100
            
            if volatility > 5:
                patterns.append('HIGH_VOLATILITY')
            elif volatility < 1:
                patterns.append('LOW_VOLATILITY')
            
            return patterns
            
        except Exception as e:
            print(f"⚠️ Pattern detection error: {str(e)}")
            return []
    
    def _generate_trading_signal(self, df: pd.DataFrame, indicators: Dict, patterns: List[str]) -> Dict:
        """Generate trading signal based on cache data analysis"""
        try:
            signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            # Technical indicators analysis
            if indicators:
                current_price = df['close'].iloc[-1]
                
                # SMA analysis
                if current_price > indicators.get('sma_20', current_price):
                    signal_scores['BUY'] += 0.2
                else:
                    signal_scores['SELL'] += 0.2
                
                if indicators.get('sma_20', 0) > indicators.get('sma_50', 0):
                    signal_scores['BUY'] += 0.15
                else:
                    signal_scores['SELL'] += 0.15
                
                # RSI analysis
                rsi = indicators.get('rsi', 50)
                if rsi < 30:  # Oversold
                    signal_scores['BUY'] += 0.3
                elif rsi > 70:  # Overbought
                    signal_scores['SELL'] += 0.3
                else:
                    signal_scores['HOLD'] += 0.2
                
                # Momentum analysis
                price_change = indicators.get('price_change_1d', 0)
                if price_change > 3:
                    signal_scores['BUY'] += 0.2
                elif price_change < -3:
                    signal_scores['SELL'] += 0.2
                else:
                    signal_scores['HOLD'] += 0.1
                
                # Volume analysis
                volume_ratio = indicators.get('volume_ratio', 1)
                if volume_ratio > 1.5:  # High volume
                    # Amplify existing trend
                    if signal_scores['BUY'] > signal_scores['SELL']:
                        signal_scores['BUY'] += 0.15
                    elif signal_scores['SELL'] > signal_scores['BUY']:
                        signal_scores['SELL'] += 0.15
            
            # Pattern analysis
            for pattern in patterns:
                if pattern in ['STRONG_UPTREND', 'UPTREND']:
                    signal_scores['BUY'] += 0.2
                elif pattern in ['STRONG_DOWNTREND', 'DOWNTREND']:
                    signal_scores['SELL'] += 0.2
                elif pattern == 'SIDEWAYS':
                    signal_scores['HOLD'] += 0.15
                elif pattern == 'HIGH_VOLATILITY':
                    # Reduce confidence in volatile conditions
                    for key in signal_scores:
                        signal_scores[key] *= 0.8
            
            # Determine final signal
            max_score = max(signal_scores.values())
            final_signal = max(signal_scores, key=signal_scores.get)
            
            # Calculate confidence (normalize to 0-1)
            confidence = min(0.95, max_score)
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'scores': signal_scores
            }
            
        except Exception as e:
            print(f"⚠️ Signal generation error: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0.3, 'scores': {}}
    
    def _update_performance_stats(self, symbol: str, processing_time: float, prediction_source: str):
        """Update performance statistics"""
        self.performance_stats['total_predictions'] += 1
        
        # Update average processing time
        total = self.performance_stats['total_predictions']
        current_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total
        
        # Update symbol accuracy tracking
        if symbol not in self.performance_stats['symbol_accuracy']:
            self.performance_stats['symbol_accuracy'][symbol] = {'detections': 0, 'confidence_sum': 0}
        
        self.performance_stats['symbol_accuracy'][symbol]['detections'] += 1
    
    def draw_integrated_overlay(self, frame: np.ndarray, prediction: IntegratedPrediction) -> np.ndarray:
        """Draw comprehensive integrated analysis overlay"""
        try:
            overlay = frame.copy()
            h, w = overlay.shape[:2]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Title
            cv2.putText(overlay, "🎯 YOLO INTEGRATED TRADING SYSTEM", (10, 35),
                       font, 0.8, (0, 255, 255), 2)
            
            # YOLO Detection Visualization
            if prediction.bbox:
                x1, y1, x2, y2 = prediction.bbox
                # Draw YOLO bounding box
                color = (0, 255, 0) if prediction.symbol_confidence > 0.7 else (0, 255, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                cv2.putText(overlay, f"YOLO: {prediction.symbol}", (x1, y1-10),
                           font, 0.6, color, 2)
            
            # Symbol Information
            symbol_color = (255, 255, 0) if prediction.symbol != 'UNKNOWN' else (128, 128, 128)
            cv2.putText(overlay, f"SYMBOL: {prediction.symbol}", (10, 75),
                       font, 0.7, symbol_color, 2)
            cv2.putText(overlay, f"Source: {prediction.prediction_source.upper()}", (10, 105),
                       font, 0.5, symbol_color, 1)
            cv2.putText(overlay, f"Confidence: {prediction.symbol_confidence:.1%}", (10, 125),
                       font, 0.5, symbol_color, 1)
            
            # Price Information
            if prediction.price:
                cv2.putText(overlay, f"PRICE: ${prediction.price:.2f}", (10, 155),
                           font, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"Price Conf: {prediction.price_confidence:.1%}", (10, 175),
                           font, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(overlay, "PRICE: N/A", (10, 155),
                           font, 0.7, (128, 128, 128), 2)
            
            # Trading Signal (Main Feature)
            signal_colors = {
                'BUY': (0, 255, 0),
                'SELL': (0, 0, 255),
                'HOLD': (0, 255, 255)
            }
            signal_color = signal_colors.get(prediction.trading_signal, (255, 255, 255))
            
            # Signal background
            cv2.rectangle(overlay, (5, 195), (450, 275), (0, 0, 0), -1)
            cv2.rectangle(overlay, (5, 195), (450, 275), signal_color, 3)
            
            cv2.putText(overlay, f"📈 SIGNAL: {prediction.trading_signal}", (15, 225),
                       font, 0.9, signal_color, 2)
            cv2.putText(overlay, f"Confidence: {prediction.signal_confidence:.1%}", (15, 250),
                       font, 0.6, signal_color, 1)
            cv2.putText(overlay, f"Data Points: {prediction.data_points}", (15, 270),
                       font, 0.5, signal_color, 1)
            
            # Technical Indicators (Right Side)
            right_x = w - 400
            y_pos = 75
            
            cv2.putText(overlay, "📊 TECHNICAL ANALYSIS", (right_x, y_pos),
                       font, 0.6, (255, 255, 255), 1)
            y_pos += 25
            
            for key, value in list(prediction.technical_indicators.items())[:6]:
                if isinstance(value, (int, float)):
                    cv2.putText(overlay, f"{key.upper()}: {value:.2f}", (right_x, y_pos),
                               font, 0.4, (200, 200, 200), 1)
                    y_pos += 20
            
            # Chart Patterns
            if prediction.chart_patterns:
                y_pos += 10
                cv2.putText(overlay, "📈 PATTERNS:", (right_x, y_pos),
                           font, 0.5, (180, 180, 180), 1)
                y_pos += 20
                
                for pattern in prediction.chart_patterns[:3]:
                    cv2.putText(overlay, f"• {pattern}", (right_x, y_pos),
                               font, 0.4, (150, 150, 150), 1)
                    y_pos += 18
            
            # Performance Info
            perf_y = h - 100
            cv2.putText(overlay, f"⚡ Processing: {prediction.processing_time_ms:.1f}ms", (10, perf_y),
                       font, 0.5, (100, 255, 100), 1)
            cv2.putText(overlay, f"📊 Total Predictions: {self.performance_stats['total_predictions']}", (10, perf_y + 20),
                       font, 0.4, (150, 150, 150), 1)
            cv2.putText(overlay, f"🎯 YOLO Success: {self.performance_stats['yolo_successful']}", (10, perf_y + 40),
                       font, 0.4, (150, 150, 150), 1)
            
            # Status indicator
            status_color = (0, 255, 0) if prediction.symbol != 'UNKNOWN' else (255, 0, 0)
            cv2.circle(overlay, (w - 30, 30), 10, status_color, -1)
            
            # Timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            cv2.putText(overlay, f"Time: {timestamp}", (w - 150, h - 20),
                       font, 0.4, (100, 100, 100), 1)
            
            return overlay
            
        except Exception as e:
            print(f"❌ Overlay error: {str(e)}")
            return frame
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        success_rate = 0.0
        if self.performance_stats['total_predictions'] > 0:
            success_rate = (self.performance_stats['yolo_successful'] + self.performance_stats['cache_data_used']) / self.performance_stats['total_predictions']
        
        return {
            'total_predictions': self.performance_stats['total_predictions'],
            'yolo_success_rate': self.performance_stats['yolo_successful'] / max(self.performance_stats['total_predictions'], 1),
            'cache_data_usage': self.performance_stats['cache_data_used'] / max(self.performance_stats['total_predictions'], 1),
            'ocr_fallback_rate': self.performance_stats['ocr_fallback_used'] / max(self.performance_stats['total_predictions'], 1),
            'avg_processing_time_ms': self.performance_stats['avg_processing_time'],
            'predictions_per_minute': 60000 / max(self.performance_stats['avg_processing_time'], 1),
            'symbol_accuracy': self.performance_stats['symbol_accuracy']
        }

def run_yolo_integrated_demo():
    """Run YOLO Integrated Trading System demo"""
    print("🎯 YOLO INTEGRATED TRADING SYSTEM DEMO")
    print("="*70)
    
    system = YOLOIntegratedTradingSystem()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available")
        return
    
    frame_count = 0
    last_prediction = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every 15 frames for performance
            if frame_count % 15 == 0:
                print(f"\n🎯 Frame {frame_count} YOLO Integrated Analysis:")
                
                prediction = system.analyze_frame_integrated(frame)
                last_prediction = prediction
                
                print(f"   📊 Symbol: {prediction.symbol} ({prediction.symbol_confidence:.1%}) - {prediction.prediction_source}")
                print(f"   💰 Price: ${prediction.price if prediction.price else 'N/A'} ({prediction.price_confidence:.1%})")
                print(f"   📈 Signal: {prediction.trading_signal} ({prediction.signal_confidence:.1%})")
                print(f"   📊 Data: {prediction.data_points} points, {len(prediction.chart_patterns)} patterns")
                print(f"   ⚡ Time: {prediction.processing_time_ms:.1f}ms")
            
            # Always show overlay with last prediction
            if last_prediction:
                vis_frame = system.draw_integrated_overlay(frame, last_prediction)
            else:
                vis_frame = frame.copy()
                cv2.putText(vis_frame, "🎯 YOLO INTEGRATED SYSTEM - INITIALIZING...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('YOLO Integrated Trading System', vis_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("📊 Performance Report:")
                report = system.get_performance_report()
                for key, value in report.items():
                    if isinstance(value, float):
                        print(f"   • {key}: {value:.3f}")
                    else:
                        print(f"   • {key}: {value}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        final_report = system.get_performance_report()
        print(f"\n🎉 FINAL YOLO INTEGRATED REPORT:")
        print(f"   📊 Total Predictions: {final_report['total_predictions']}")
        print(f"   🎯 YOLO Success Rate: {final_report['yolo_success_rate']:.1%}")
        print(f"   📊 Cache Data Usage: {final_report['cache_data_usage']:.1%}")
        print(f"   ⚡ Avg Processing: {final_report['avg_processing_time_ms']:.1f}ms")

if __name__ == "__main__":
    run_yolo_integrated_demo() 