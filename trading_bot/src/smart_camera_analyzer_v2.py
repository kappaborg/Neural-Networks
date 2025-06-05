"""
SMART CAMERA ANALYZER V2
Enhanced with working chart classifier + trading analysis
"""

import cv2
import numpy as np
import tensorflow as tf
import pytesseract
import os
import json
import glob
from datetime import datetime
import pandas as pd
import ta

class SmartCameraAnalyzerV2:
    """
    Enhanced camera analyzer with:
    - Working chart classifier (77K params, 95%+ accuracy)
    - Symbol recognition
    - Trading signal analysis
    - Real-time processing
    - REAL CACHE DATA INTEGRATION
    """
    
    def __init__(self):
        self.chart_classifier = None
        self.confidence_threshold = 0.6  # For chart detection
        
        # Load available symbols from cache directory
        self.available_symbols = self.load_available_symbols()
        self.symbol_patterns = self.create_symbol_patterns()
        
        self.current_symbol = None
        self.analysis_history = []
        
        # Cache for stable display
        self.last_analysis = None
        self.analysis_display_count = 0
        
        print("🚀 Smart Camera Analyzer V2 Initialized")
        print(f"📊 Available symbols: {len(self.available_symbols)}")
        print(f"🔍 Symbol patterns: {self.symbol_patterns}")
        self.load_chart_classifier()
        
    def load_available_symbols(self):
        """Load available symbols from cache directory"""
        try:
            cache_dir = "data/cache"
            if not os.path.exists(cache_dir):
                print("⚠️ Cache directory not found! Using default symbols.")
                return ['BTC', 'ETH', 'SOL', 'ADA']
            
            symbols = set()
            csv_files = glob.glob(os.path.join(cache_dir, "*.csv"))
            
            for file_path in csv_files:
                filename = os.path.basename(file_path)
                # Extract symbol from filename (e.g., BTC_USDT_2024-01-22_2025-06-04.csv -> BTC)
                symbol = filename.split('_')[0]
                symbols.add(symbol)
            
            available_symbols = sorted(list(symbols))
            print(f"✅ Found {len(available_symbols)} symbols in cache: {available_symbols}")
            return available_symbols
            
        except Exception as e:
            print(f"❌ Error loading symbols: {str(e)}")
            return ['BTC', 'ETH', 'SOL', 'ADA']  # Fallback
    
    def create_symbol_patterns(self):
        """Create symbol patterns for OCR detection"""
        patterns = []
        
        # Add base symbols
        for symbol in self.available_symbols:
            patterns.append(symbol)
            
            # Add USDT pairs for crypto symbols
            if symbol not in ['AAPL', 'MSFT', 'NVDA', 'TSLA']:  # These are stocks
                patterns.append(f"{symbol}/USDT")
                patterns.append(f"{symbol}USDT")
                patterns.append(f"{symbol}-USDT")
        
        # Add common additional patterns
        patterns.extend(['USDT', 'USD', 'NVDA', 'TSLA'])
        
        return list(set(patterns))  # Remove duplicates
        
    def load_chart_classifier(self):
        """Load the working ultra simple chart classifier"""
        try:
            model_path = "models/ultra_simple_chart.h5"
            if os.path.exists(model_path):
                self.chart_classifier = tf.keras.models.load_model(model_path)
                print("✅ Chart classifier loaded!")
                print(f"   • Parameters: {self.chart_classifier.count_params():,}")
                print(f"   • Model: Ultra Simple Binary Classifier")
            else:
                print("⚠️ Chart classifier not found! Chart detection disabled.")
        except Exception as e:
            print(f"❌ Error loading chart classifier: {str(e)}")
    
    def detect_chart(self, frame):
        """Detect if frame contains a chart using ML model"""
        if self.chart_classifier is None:
            return {'is_chart': False, 'confidence': 0.0, 'method': 'No Model'}
        
        try:
            # Preprocess frame for model
            img_resized = cv2.resize(frame, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predict
            prediction = self.chart_classifier.predict(img_batch, verbose=0)[0][0]
            
            is_chart = prediction > 0.5
            confidence = prediction if is_chart else (1 - prediction)
            
            return {
                'is_chart': is_chart,
                'confidence': confidence,
                'prediction_raw': prediction,
                'method': 'ML Model',
                'threshold': 0.5
            }
            
        except Exception as e:
            print(f"❌ Chart detection error: {str(e)}")
            return {'is_chart': False, 'confidence': 0.0, 'method': 'Error'}
    
    def recognize_symbol(self, frame):
        """Recognize trading symbol from chart"""
        try:
            # Convert to grayscale for OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance for text recognition
            enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
            
            # Apply threshold
            _, thresh = cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY)
            
            # OCR configuration for symbols
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ/0123456789$-'
            
            # Extract text
            text = pytesseract.image_to_string(thresh, config=config)
            
            # Find known symbols
            detected_symbols = []
            for pattern in self.symbol_patterns:
                if pattern in text.upper():
                    # Extract base symbol
                    base_symbol = pattern.split('/')[0].split('-')[0].replace('USDT', '')
                    if base_symbol in self.available_symbols:
                        detected_symbols.append(base_symbol)
            
            # Get most likely symbol
            if detected_symbols:
                symbol = detected_symbols[0]  # Take first match
                confidence = min(0.9, len(detected_symbols) * 0.3)  # Simple confidence
                return {
                    'symbol': symbol,
                    'confidence': confidence,
                    'all_detected': detected_symbols,
                    'raw_text': text.strip()
                }
            else:
                return {
                    'symbol': 'UNKNOWN',
                    'confidence': 0.0,
                    'all_detected': [],
                    'raw_text': text.strip()
                }
                
        except Exception as e:
            print(f"❌ Symbol recognition error: {str(e)}")
            return {'symbol': 'ERROR', 'confidence': 0.0}
    
    def load_market_data(self, symbol='BTC'):
        """Load REAL market data from cache"""
        try:
            # Find the data file for this symbol
            cache_dir = "data/cache"
            possible_files = [
                f"{symbol}_USDT_*.csv",
                f"{symbol}_*.csv"
            ]
            
            data_file = None
            for pattern in possible_files:
                files = glob.glob(os.path.join(cache_dir, pattern))
                if files:
                    data_file = files[0]  # Take first match
                    break
            
            if not data_file:
                print(f"⚠️ No data file found for {symbol}")
                return None
            
            print(f"📊 Loading real data: {os.path.basename(data_file)}")
            
            # Load the CSV data
            df = pd.read_csv(data_file)
            
            # Rename columns to match expected format
            column_mapping = {
                'timestamp': 'date',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp to datetime if needed
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Take last 100 rows for analysis
            df_recent = df.tail(100).copy()
            
            print(f"✅ Loaded {len(df_recent)} recent data points for {symbol}")
            print(f"   • Date range: {df_recent['date'].min()} to {df_recent['date'].max()}")
            print(f"   • Current price: ${df_recent['close'].iloc[-1]:.2f}")
            
            return df_recent
            
        except Exception as e:
            print(f"❌ Error loading market data for {symbol}: {str(e)}")
            return None
    
    def analyze_trading_signals(self, symbol='BTC'):
        """Analyze trading signals using REAL technical indicators from cache data"""
        try:
            # Load real market data
            df = self.load_market_data(symbol)
            if df is None:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No real data available'}
            
            # Use existing indicators if available, otherwise calculate
            has_indicators = all(col in df.columns for col in ['RSI', 'MACD', 'SMA_20', 'SMA_50'])
            
            if has_indicators:
                print(f"✅ Using pre-calculated indicators for {symbol}")
                # Use existing indicators from cache
                df['sma_20'] = df['SMA_20']
                df['sma_50'] = df['SMA_50'] 
                df['rsi'] = df['RSI']
                df['macd'] = df['MACD']
            else:
                print(f"📊 Calculating fresh indicators for {symbol}")
                # Calculate fresh indicators
                df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
                df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
                df['macd'] = ta.trend.macd_diff(df['close'])
            
            # Get latest values
            latest = df.iloc[-1]
            price = latest['close']
            sma_20 = latest['sma_20'] if not pd.isna(latest['sma_20']) else price
            sma_50 = latest['sma_50'] if not pd.isna(latest['sma_50']) else price
            rsi = latest['rsi'] if not pd.isna(latest['rsi']) else 50
            macd = latest['macd'] if not pd.isna(latest['macd']) else 0
            
            # Trading logic with real data
            signals = []
            confidence_factors = []
            
            # Price vs Moving Averages
            if price > sma_20 and sma_20 > sma_50:
                signals.append('BUY')
                confidence_factors.append(0.35)
            elif price < sma_20 and sma_20 < sma_50:
                signals.append('SELL')
                confidence_factors.append(0.35)
            else:
                signals.append('HOLD')
                confidence_factors.append(0.15)
            
            # RSI Analysis
            if rsi < 30:  # Oversold
                signals.append('BUY')
                confidence_factors.append(0.30)
            elif rsi > 70:  # Overbought
                signals.append('SELL')
                confidence_factors.append(0.30)
            else:
                signals.append('HOLD')
                confidence_factors.append(0.10)
            
            # MACD Analysis
            if macd > 0:
                signals.append('BUY')
                confidence_factors.append(0.25)
            elif macd < 0:
                signals.append('SELL')
                confidence_factors.append(0.25)
            else:
                signals.append('HOLD')
                confidence_factors.append(0.10)
            
            # Determine final signal
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            hold_count = signals.count('HOLD')
            
            if buy_count > sell_count and buy_count > hold_count:
                final_signal = 'BUY'
                confidence = min(0.95, sum(confidence_factors) * (buy_count / len(signals)))
            elif sell_count > buy_count and sell_count > hold_count:
                final_signal = 'SELL'
                confidence = min(0.95, sum(confidence_factors) * (sell_count / len(signals)))
            else:
                final_signal = 'HOLD'
                confidence = sum(confidence_factors) / len(signals)
            
            # Generate reason with real data
            reasons = []
            if price > sma_20:
                reasons.append(f"Price ${price:.2f} above SMA20 ${sma_20:.2f}")
            if rsi < 30:
                reasons.append(f"RSI oversold at {rsi:.1f}")
            elif rsi > 70:
                reasons.append(f"RSI overbought at {rsi:.1f}")
            if macd > 0:
                reasons.append(f"MACD bullish {macd:.4f}")
            elif macd < 0:
                reasons.append(f"MACD bearish {macd:.4f}")
            
            reason = "; ".join(reasons) if reasons else "Neutral indicators"
            
            # Calculate price change
            if len(df) >= 2:
                prev_price = df['close'].iloc[-2]
                price_change = ((price - prev_price) / prev_price) * 100
                price_change_text = f"+{price_change:.2f}%" if price_change > 0 else f"{price_change:.2f}%"
            else:
                price_change_text = "0.00%"
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'reason': reason,
                'price': price,
                'price_change': price_change_text,
                'rsi': rsi,
                'macd': macd,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'data_source': 'REAL_CACHE_DATA',
                'data_points': len(df),
                'indicators': {
                    'price': f"${price:.2f}",
                    'change': price_change_text,
                    'rsi': f"{rsi:.1f}",
                    'macd': f"{macd:.4f}"
                }
            }
            
        except Exception as e:
            print(f"❌ Trading analysis error for {symbol}: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Analysis error: {str(e)}'}
    
    def analyze_frame(self, frame):
        """Complete frame analysis: chart detection + symbol + trading"""
        try:
            analysis_result = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'chart_detection': {'is_chart': False, 'confidence': 0.0},
                'symbol_recognition': {'symbol': 'NONE', 'confidence': 0.0},
                'trading_analysis': {'signal': 'HOLD', 'confidence': 0.0},
                'final_status': 'ANALYZING'
            }
            
            # Step 1: Chart Detection
            chart_result = self.detect_chart(frame)
            analysis_result['chart_detection'] = chart_result
            
            if not chart_result['is_chart'] or chart_result['confidence'] < self.confidence_threshold:
                analysis_result['final_status'] = 'NO_CHART_DETECTED'
                return analysis_result
            
            # Step 2: Symbol Recognition (only if chart detected)
            symbol_result = self.recognize_symbol(frame)
            analysis_result['symbol_recognition'] = symbol_result
            
            detected_symbol = symbol_result.get('symbol', 'UNKNOWN')
            if detected_symbol == 'UNKNOWN':
                detected_symbol = 'BTC'  # Default fallback
            
            # Step 3: Trading Analysis (only if chart detected)
            trading_result = self.analyze_trading_signals(detected_symbol)
            analysis_result['trading_analysis'] = trading_result
            
            analysis_result['final_status'] = 'ANALYSIS_COMPLETE'
            
            # Store in history
            self.analysis_history.append(analysis_result)
            
            # Keep only last 10 analyses
            if len(self.analysis_history) > 10:
                self.analysis_history = self.analysis_history[-10:]
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Frame analysis error: {str(e)}")
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'final_status': 'ERROR',
                'error': str(e)
            }
    
    def draw_analysis_overlay(self, frame, analysis):
        """Draw analysis results on frame with stable display - Enhanced for REAL DATA"""
        try:
            overlay = frame.copy()
            h, w = overlay.shape[:2]
            
            # Create semi-transparent background for better readability
            background = np.zeros((h, w, 3), dtype=np.uint8)
            background[:] = (0, 0, 0)  # Black background
            
            # Chart detection status
            chart_det = analysis.get('chart_detection', {})
            is_chart = chart_det.get('is_chart', False)
            chart_conf = chart_det.get('confidence', 0.0)
            
            # Define text properties for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            if is_chart:
                # Success - chart detected
                status_color = (0, 255, 0)  # Green
                
                # Chart detection
                cv2.putText(overlay, f"CHART DETECTED", (10, 40),
                           font, 0.9, status_color, thickness)
                cv2.putText(overlay, f"Confidence: {chart_conf:.1%}", (10, 75),
                           font, 0.6, status_color, 1)
                
                # Symbol recognition
                symbol_rec = analysis.get('symbol_recognition', {})
                symbol = symbol_rec.get('symbol', 'UNKNOWN')
                symbol_conf = symbol_rec.get('confidence', 0.0)
                
                # Symbol with background
                symbol_text = f"Symbol: {symbol}"
                if symbol != 'UNKNOWN':
                    cv2.putText(overlay, symbol_text, (10, 110),
                               font, font_scale, (255, 255, 0), thickness)
                    cv2.putText(overlay, f"({symbol_conf:.1%})", (10, 140),
                               font, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(overlay, symbol_text, (10, 110),
                               font, font_scale, (128, 128, 128), thickness)
                
                # Trading analysis - ENHANCED FOR REAL DATA
                trading = analysis.get('trading_analysis', {})
                signal = trading.get('signal', 'HOLD')
                signal_conf = trading.get('confidence', 0.0)
                reason = trading.get('reason', '')
                data_source = trading.get('data_source', 'SYNTHETIC')
                data_points = trading.get('data_points', 0)
                
                # Signal with appropriate color and background
                signal_colors = {
                    'BUY': (0, 255, 0),      # Green
                    'SELL': (0, 0, 255),     # Red  
                    'HOLD': (0, 255, 255)    # Yellow
                }
                signal_color = signal_colors.get(signal, (255, 255, 255))
                
                # Draw signal background rectangle - LARGER for more info
                signal_bg_start = (5, 170)
                signal_bg_end = (350, 250)
                cv2.rectangle(overlay, signal_bg_start, signal_bg_end, (0, 0, 0), -1)
                cv2.rectangle(overlay, signal_bg_start, signal_bg_end, signal_color, 2)
                
                # Signal text
                cv2.putText(overlay, f"SIGNAL: {signal}", (15, 195),
                           font, 0.8, signal_color, thickness)
                cv2.putText(overlay, f"Confidence: {signal_conf:.1%}", (15, 215),
                           font, 0.5, signal_color, 1)
                
                # Data source indicator
                data_source_color = (0, 255, 0) if data_source == 'REAL_CACHE_DATA' else (255, 165, 0)
                cv2.putText(overlay, f"Data: {data_source} ({data_points}pts)", (15, 235),
                           font, 0.4, data_source_color, 1)
                
                # Technical indicators (right side) - ENHANCED
                indicators = trading.get('indicators', {})
                right_x = w - 280
                y_pos = 40
                
                cv2.putText(overlay, "REAL DATA INDICATORS:", (right_x, y_pos),
                           font, 0.6, (255, 255, 255), 1)
                y_pos += 30
                
                # Show price with change
                price_text = indicators.get('price', 'N/A')
                price_change = indicators.get('change', '0.00%')
                change_color = (0, 255, 0) if price_change.startswith('+') else (0, 0, 255) if price_change.startswith('-') else (255, 255, 255)
                
                cv2.putText(overlay, f"PRICE: {price_text}", (right_x, y_pos),
                           font, 0.6, (255, 255, 255), 1)
                y_pos += 20
                cv2.putText(overlay, f"CHANGE: {price_change}", (right_x, y_pos),
                           font, 0.5, change_color, 1)
                y_pos += 30
                
                # Other indicators
                indicator_items = [(k, v) for k, v in indicators.items() if k not in ['price', 'change']]
                for key, value in indicator_items:
                    cv2.putText(overlay, f"{key.upper()}: {value}", (right_x, y_pos),
                               font, 0.5, (200, 200, 200), 1)
                    y_pos += 25
                
                # Trading pair info (if available)
                if symbol != 'UNKNOWN' and symbol in self.available_symbols:
                    pair_text = f"{symbol}/USDT" if symbol not in ['AAPL', 'MSFT'] else f"{symbol}/USD"
                    cv2.putText(overlay, pair_text, (right_x, y_pos),
                               font, 0.5, (180, 180, 180), 1)
                    y_pos += 25
                
                # Cache status
                cache_status = f"Cache: ✅ {len(self.available_symbols)} symbols"
                cv2.putText(overlay, cache_status, (right_x, y_pos),
                           font, 0.4, (100, 255, 100), 1)
                
                # Reason (bottom, if space allows) - ENHANCED
                if len(reason) > 0:
                    reason_y = h - 80
                    cv2.putText(overlay, "REAL DATA ANALYSIS:", (10, reason_y),
                               font, 0.5, (180, 180, 180), 1)
                    
                    # Split long reasons into multiple lines
                    max_chars = 80
                    if len(reason) > max_chars:
                        reason_lines = [reason[i:i+max_chars] for i in range(0, len(reason), max_chars)]
                        for i, line in enumerate(reason_lines[:2]):  # Max 2 lines
                            cv2.putText(overlay, line, (10, reason_y + 25 + i*20),
                                       font, 0.4, (150, 150, 150), 1)
                    else:
                        cv2.putText(overlay, reason, (10, reason_y + 25),
                                   font, 0.4, (150, 150, 150), 1)
                
                # Timestamp with data freshness
                timestamp = analysis.get('timestamp', 'Unknown')
                cv2.putText(overlay, f"Last Update: {timestamp} (Real Data)", (10, h - 20),
                           font, 0.4, (100, 100, 100), 1)
            
            else:
                # No chart detected
                status_color = (0, 0, 255)  # Red
                
                cv2.putText(overlay, f"NO CHART DETECTED", (10, 40),
                           font, 0.9, status_color, thickness)
                cv2.putText(overlay, f"Confidence: {chart_conf:.1%}", (10, 75),
                           font, 0.6, status_color, 1)
                
                # Instructions - ENHANCED
                cv2.putText(overlay, "Instructions:", (10, 120),
                           font, 0.6, (255, 255, 255), 1)
                cv2.putText(overlay, "1. Point camera at trading chart", (10, 150),
                           font, 0.5, (200, 200, 200), 1)
                cv2.putText(overlay, "2. Ensure good lighting", (10, 175),
                           font, 0.5, (200, 200, 200), 1)
                cv2.putText(overlay, "3. Wait for detection", (10, 200),
                           font, 0.5, (200, 200, 200), 1)
                
                # Show available symbols
                available_text = f"Available symbols: {', '.join(self.available_symbols[:8])}"
                if len(self.available_symbols) > 8:
                    available_text += f" + {len(self.available_symbols) - 8} more"
                
                cv2.putText(overlay, "Supported symbols:", (10, 240),
                           font, 0.5, (150, 150, 150), 1)
                cv2.putText(overlay, available_text, (10, 265),
                           font, 0.4, (120, 120, 120), 1)
            
            return overlay
            
        except Exception as e:
            print(f"❌ Overlay drawing error: {str(e)}")
            # Return frame with error message
            error_frame = frame.copy()
            cv2.putText(error_frame, "OVERLAY ERROR", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return error_frame
    
    def run_real_time_analysis(self, camera_index=0):
        """Run real-time chart analysis"""
        print("🚀 SMART CAMERA ANALYZER V2 - REAL TIME")
        print("="*50)
        print("🎯 Features:")
        print("  • Chart Detection (ML-powered)")
        print("  • Symbol Recognition (OCR)")
        print("  • Trading Analysis (BUY/SELL/HOLD)")
        print("  • Real-time Processing")
        print("\n📝 Controls:")
        print("  • 'q' - Quit")
        print("  • 's' - Save screenshot")
        print("  • 'h' - Show analysis history")
        print("  • 'r' - Force refresh analysis")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Camera not available")
            return
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Analyze every 15 frames (for better performance)
                if frame_count % 15 == 0:
                    print(f"🔄 Analyzing frame {frame_count}...")
                    analysis = self.analyze_frame(frame)
                    
                    # Cache the analysis for stable display
                    self.last_analysis = analysis
                    self.analysis_display_count = 0
                    
                    print(f"✅ Analysis complete: {analysis.get('final_status', 'Unknown')}")
                
                # Always draw overlay with last analysis (if available)
                if self.last_analysis is not None:
                    overlay_frame = self.draw_analysis_overlay(frame, self.last_analysis)
                    
                    # Add frame counter and analysis age info
                    age_text = f"Analysis Age: {self.analysis_display_count} frames"
                    cv2.putText(overlay_frame, age_text, (frame.shape[1] - 300, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                    
                    # Add update indicator
                    if self.analysis_display_count < 30:  # Fresh analysis
                        cv2.circle(overlay_frame, (frame.shape[1] - 30, 50), 8, (0, 255, 0), -1)
                    else:  # Stale analysis
                        cv2.circle(overlay_frame, (frame.shape[1] - 30, 50), 8, (0, 165, 255), -1)
                    
                    cv2.imshow('Smart Camera Analyzer V2', overlay_frame)
                    self.analysis_display_count += 1
                else:
                    # No analysis yet - show basic frame with waiting message
                    waiting_frame = frame.copy()
                    cv2.putText(waiting_frame, "INITIALIZING ANALYSIS...", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(waiting_frame, "Point camera at trading chart", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow('Smart Camera Analyzer V2', waiting_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"screenshot_{timestamp}.jpg"
                    if self.last_analysis is not None:
                        # Save with overlay
                        overlay_frame = self.draw_analysis_overlay(frame, self.last_analysis)
                        cv2.imwrite(filename, overlay_frame)
                    else:
                        cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('h'):
                    print("\n📊 ANALYSIS HISTORY:")
                    for i, hist in enumerate(self.analysis_history[-5:]):  # Last 5
                        print(f"  {i+1}. {hist['timestamp']} - {hist['final_status']}")
                        if 'trading_analysis' in hist:
                            signal = hist['trading_analysis'].get('signal', 'N/A')
                            confidence = hist['trading_analysis'].get('confidence', 0.0)
                            print(f"     Signal: {signal} ({confidence:.1%})")
                elif key == ord('r'):
                    # Force refresh analysis
                    print("🔄 Force refreshing analysis...")
                    analysis = self.analyze_frame(frame)
                    self.last_analysis = analysis
                    self.analysis_display_count = 0
                    print(f"✅ Analysis refreshed: {analysis.get('final_status', 'Unknown')}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopping analysis...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Smart Camera Analyzer V2 stopped!")
            
            # Summary
            print(f"\n📊 SESSION SUMMARY:")
            print(f"  • Total analyses: {len(self.analysis_history)}")
            if self.analysis_history:
                signals = [a.get('trading_analysis', {}).get('signal', 'N/A') for a in self.analysis_history]
                buy_count = signals.count('BUY')
                sell_count = signals.count('SELL')
                hold_count = signals.count('HOLD')
                print(f"  • Signals: {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD")

def main():
    """Main function"""
    analyzer = SmartCameraAnalyzerV2()
    analyzer.run_real_time_analysis()

if __name__ == "__main__":
    main() 