#!/usr/bin/env python3
"""
ULTRA FAST OCR SYSTEM
Optimize edilmiş coin name + price extraction
"""

import cv2
import numpy as np
import pytesseract
import re
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class OCRResult:
    """OCR sonuçları"""
    text: str
    confidence: float
    processing_time_ms: float
    method: str

class UltraFastOCR:
    """Ultra-optimized OCR for crypto trading"""
    
    def __init__(self):
        # Pre-compiled patterns
        self.coin_patterns = {
            'symbols': [
                r'(BTC|ETH|SOL|ADA|BNB|XRP|DOT|DOGE|AVAX|MATIC|LINK|UNI|LTC|TRX|ATOM)(?:USDT?)?',
                r'(BITCOIN|ETHEREUM|SOLANA|CARDANO|BINANCE|RIPPLE)',
                r'([A-Z]{2,5})(?:/USDT?|USDT?|\s*USD)',
            ],
            'prices': [
                r'\$?\s*(\d{1,6}[,.]?\d{0,6})',      # $105,801.09 or $2,863.45
                r'(\d{1,3}[,]\d{3}[.]\d{2,6})',     # 105,801.09 or 2,863.45  
                r'(\d{1,6}[.]\d{2,6})',             # 105801.09 or 2863.45
                r'(\d{3,6})',                        # 2863 or 105801
                r'\$(\d{1,3}[,]\d{3})',             # $2,863
                r'\$(\d{3,6})',                      # $2863
            ]
        }
        
        # Fast OCR configs
        self.fast_configs = {
            'symbol': '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ/',
            'price': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,$',
            'combo': '--oem 3 --psm 6'
        }
        
        # Performance cache
        self.region_cache = {}
        self.method_performance = {
            'symbol_roi': {'success': 0, 'attempts': 0, 'avg_time': 0},
            'price_roi': {'success': 0, 'attempts': 0, 'avg_time': 0},
            'enhanced': {'success': 0, 'attempts': 0, 'avg_time': 0}
        }
    
    def extract_coin_symbol_fast(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> OCRResult:
        """Ultra-fast coin symbol extraction"""
        start_time = time.time()
        
        try:
            # ROI extraction - EXPANDED REGIONS FOR SYMBOLS
            if roi:
                x, y, w, h = roi
                region = frame[y:y+h, x:x+w]
            else:
                h, w = frame.shape[:2]
                # Try multiple regions for symbol detection
                regions = [
                    frame[0:int(h*0.25), 0:int(w*0.6)],           # Top-left expanded (main symbol area)
                    frame[int(h*0.4):int(h*0.8), 0:int(w*0.3)],   # Mid-left (vertical menus)
                    frame[int(h*0.05):int(h*0.15), int(w*0.2):int(w*0.8)],  # Top-center band
                ]
                region = regions[0]  # Start with most likely
            
            best_symbol = 'UNKNOWN'
            best_confidence = 0.0
            
            # Try multiple preprocessing methods on current region
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            
            # Enhanced preprocessing variations
            preprocessed_regions = [
                cv2.convertScaleAbs(gray, alpha=2.5, beta=30),    # Enhanced contrast
                cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1],  # Binary threshold
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),  # Adaptive
            ]
            
            for processed_region in preprocessed_regions:
                try:
                    # Try different OCR configs
                    configs = [
                        '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ/',
                        '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        '--oem 3 --psm 6',
                    ]
                    
                    for config in configs:
                        text = pytesseract.image_to_string(processed_region, config=config, timeout=1)
                        symbol = self._extract_symbol_from_text(text)
                        
                        if symbol != 'UNKNOWN':
                            confidence = 0.8 + (0.1 if len(text.strip()) > 2 else 0)
                            if confidence > best_confidence:
                                best_symbol = symbol
                                best_confidence = confidence
                                
                            if best_confidence > 0.8:  # Fast exit on good result
                                break
                except:
                    continue
                    
                if best_confidence > 0.8:
                    break
            
            # If no good result, try other ROI regions
            if best_confidence < 0.5 and not roi:
                for alt_region in regions[1:]:  # Skip first region (already tried)
                    alt_gray = cv2.cvtColor(alt_region, cv2.COLOR_BGR2GRAY) if len(alt_region.shape) == 3 else alt_region
                    alt_enhanced = cv2.convertScaleAbs(alt_gray, alpha=2.5, beta=30)
                    
                    try:
                        text = pytesseract.image_to_string(alt_enhanced, config='--oem 3 --psm 7', timeout=1)
                        symbol = self._extract_symbol_from_text(text)
                        
                        if symbol != 'UNKNOWN':
                            confidence = 0.7
                            if confidence > best_confidence:
                                best_symbol = symbol
                                best_confidence = confidence
                                
                            if best_confidence > 0.7:
                                break
                    except:
                        continue
            
            processing_time = (time.time() - start_time) * 1000
            self._update_performance('symbol_roi', best_confidence > 0.5, processing_time)
            
            return OCRResult(
                text=best_symbol,
                confidence=best_confidence,
                processing_time_ms=processing_time,
                method='fast_symbol_roi'
            )
            
        except Exception as e:
            print(f"❌ Fast symbol OCR error: {str(e)}")
            return OCRResult('UNKNOWN', 0.0, (time.time() - start_time) * 1000, 'error')
    
    def extract_price_fast(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> OCRResult:
        """Ultra-fast price extraction"""
        start_time = time.time()
        
        try:
            # ROI extraction (price usually right side or center)
            if roi:
                x, y, w, h = roi
                region = frame[y:y+h, x:x+w]
            else:
                h, w = frame.shape[:2]
                # Multiple regions for price detection - FIXED COORDINATES
                regions = [
                    frame[int(h*0.05):int(h*0.35), int(w*0.2):int(w*0.8)],   # Top-center (main price area)
                    frame[int(h*0.1):int(h*0.4), int(w*0.6):w],              # Top-right 
                    frame[int(h*0.3):int(h*0.6), int(w*0.1):int(w*0.6)],     # Mid-left (secondary prices)
                    frame[int(h*0.05):int(h*0.25), 0:int(w*0.5)],            # Top-left (symbol area prices)
                ]
                region = regions[0]  # Start with most likely (top-center)
            
            # Multi-preprocessing for price detection
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            
            processed_regions = [
                cv2.convertScaleAbs(gray, alpha=2.2, beta=60),  # Enhanced
                cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1],  # Binary
            ]
            
            best_price = None
            best_confidence = 0.0
            
            # Try all preprocessing methods on current region
            for processed_region in processed_regions:
                try:
                    text = pytesseract.image_to_string(processed_region, config=self.fast_configs['price'], timeout=1)
                    price, conf = self._extract_price_from_text(text)
                    
                    if conf > best_confidence:
                        best_price = price
                        best_confidence = conf
                        
                    if best_confidence > 0.8:  # Fast exit on good result
                        break
                except:
                    continue
            
            # If no good result, try other ROI regions
            if best_confidence < 0.5 and not roi:
                for i, alt_region in enumerate(regions[1:], 1):  # Skip first region (already tried)
                    alt_gray = cv2.cvtColor(alt_region, cv2.COLOR_BGR2GRAY) if len(alt_region.shape) == 3 else alt_region
                    alt_enhanced = cv2.convertScaleAbs(alt_gray, alpha=2.2, beta=60)
                    
                    try:
                        text = pytesseract.image_to_string(alt_enhanced, config=self.fast_configs['price'], timeout=1)
                        price, conf = self._extract_price_from_text(text)
                        
                        if conf > best_confidence:
                            best_price = price
                            best_confidence = conf
                            
                        if best_confidence > 0.8:  # Fast exit on good result
                            break
                    except:
                        continue
            
            processing_time = (time.time() - start_time) * 1000
            self._update_performance('price_roi', best_confidence > 0.5, processing_time)
            
            return OCRResult(
                text=str(best_price) if best_price else '',
                confidence=best_confidence,
                processing_time_ms=processing_time,
                method='fast_price_roi'
            )
            
        except Exception as e:
            print(f"❌ Fast price OCR error: {str(e)}")
            return OCRResult('', 0.0, (time.time() - start_time) * 1000, 'error')
    
    def extract_combo_fast(self, frame: np.ndarray) -> Dict:
        """Combined symbol + price extraction"""
        start_time = time.time()
        
        try:
            # Parallel extraction
            symbol_result = self.extract_coin_symbol_fast(frame)
            price_result = self.extract_price_fast(frame)
            
            # Smart fallback: If symbol detection fails but we have a price, guess the symbol
            symbol = symbol_result.text
            symbol_confidence = symbol_result.confidence
            price = float(price_result.text) if price_result.text and price_result.text.replace('.', '').isdigit() else None
            
            if symbol == 'UNKNOWN' and price is not None:
                # Smart symbol guessing based on price ranges
                if 50000 < price < 200000:  # BTC range
                    symbol = 'BTC'
                    symbol_confidence = 0.6  # Medium confidence from price inference
                elif 1000 < price < 10000:  # ETH range
                    symbol = 'ETH'
                    symbol_confidence = 0.7  # Higher confidence for ETH range
                elif 100 < price < 1000:   # BNB/SOL range
                    symbol = 'SOL'  # Most likely in this range
                    symbol_confidence = 0.5
                elif 10 < price < 100:     # Lower cap altcoins
                    symbol = 'ADA'  # Common in this range
                    symbol_confidence = 0.4
            
            total_time = (time.time() - start_time) * 1000
            
            return {
                'symbol': symbol,
                'symbol_confidence': symbol_confidence,
                'price': price,
                'price_confidence': price_result.confidence,
                'total_time_ms': total_time,
                'method': 'ultra_fast_combo_with_fallback'
            }
            
        except Exception as e:
            print(f"❌ Combo OCR error: {str(e)}")
            return {
                'symbol': 'UNKNOWN',
                'symbol_confidence': 0.0,
                'price': None,
                'price_confidence': 0.0,
                'total_time_ms': (time.time() - start_time) * 1000,
                'method': 'error'
            }
    
    def _extract_symbol_from_text(self, text: str) -> str:
        """Extract crypto symbol from OCR text"""
        if not text:
            return 'UNKNOWN'
        
        text = text.upper().strip()
        
        # Direct matches first (fastest)
        major_coins = ['BTC', 'ETH', 'SOL', 'ADA', 'BNB', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC', 'LINK', 'UNI', 'LTC', 'TRX', 'ATOM']
        for coin in major_coins:
            if coin in text:
                return coin
        
        # Pattern matching
        for pattern in self.coin_patterns['symbols']:
            match = re.search(pattern, text)
            if match:
                symbol = match.group(1)
                if symbol in major_coins:
                    return symbol
        
        return 'UNKNOWN'
    
    def _extract_price_from_text(self, text: str) -> Tuple[Optional[float], float]:
        """Extract price from OCR text"""
        if not text:
            return None, 0.0
        
        prices = []
        
        for pattern in self.coin_patterns['prices']:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    clean_price = match.replace(',', '').replace('$', '')
                    price_value = float(clean_price)
                    
                    # Enhanced sanity check for realistic crypto prices
                    if 10 < price_value < 200000:  # $10 - $200K range for cryptos
                        # Additional filtering for common noise values
                        if price_value not in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
                            prices.append(price_value)
                except:
                    continue
        
        if prices:
            # Return most frequent price, prefer values in typical ranges
            best_price = max(set(prices), key=prices.count)
            
            # Boost confidence for typical ETH/BTC ranges
            confidence = len(prices) * 0.3 + 0.4
            if 1000 < best_price < 100000:  # ETH/BTC typical range
                confidence += 0.2
            if len(prices) > 1:  # Multiple detections
                confidence += 0.1
                
            confidence = min(0.95, confidence)
            return best_price, confidence
        
        return None, 0.0
    
    def _update_performance(self, method: str, success: bool, processing_time: float):
        """Update performance metrics"""
        if method in self.method_performance:
            stats = self.method_performance[method]
            stats['attempts'] += 1
            if success:
                stats['success'] += 1
            
            # Update average time
            current_avg = stats['avg_time']
            attempts = stats['attempts']
            stats['avg_time'] = (current_avg * (attempts - 1) + processing_time) / attempts
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        for method, data in self.method_performance.items():
            if data['attempts'] > 0:
                stats[method] = {
                    'success_rate': data['success'] / data['attempts'],
                    'avg_time_ms': data['avg_time'],
                    'attempts': data['attempts']
                }
        return stats

def test_ultra_fast_ocr():
    """Test ultra fast OCR system"""
    print("⚡ ULTRA FAST OCR TEST")
    print("="*50)
    
    ocr = UltraFastOCR()
    
    cap = cv2.VideoCapture(0)
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
            
            if frame_count % 10 == 0:  # Every 10 frames
                print(f"\n⚡ Frame {frame_count} Ultra Fast OCR:")
                
                # Combined extraction
                result = ocr.extract_combo_fast(frame)
                symbol = result['symbol']
                price = result['price']
                total_time = result['total_time_ms']
                
                print(f"   🎯 Symbol: {symbol} ({result['symbol_confidence']:.1%})")
                print(f"   💰 Price: ${price if price else 'N/A'} ({result['price_confidence']:.1%})")
                print(f"   ⚡ Speed: {total_time:.1f}ms")
            
            # Visualization
            vis_frame = frame.copy()
            cv2.putText(vis_frame, f"Ultra Fast OCR - Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show ROI regions
            h, w = vis_frame.shape[:2]
            cv2.rectangle(vis_frame, (0, 0), (int(w*0.4), int(h*0.15)), (0, 255, 0), 2)  # Symbol ROI
            cv2.rectangle(vis_frame, (int(w*0.6), int(h*0.1)), (w, int(h*0.4)), (255, 0, 0), 2)  # Price ROI
            
            cv2.imshow('Ultra Fast OCR Test', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Test stopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance stats
        print("\n📊 ULTRA FAST OCR PERFORMANCE:")
        stats = ocr.get_performance_stats()
        for method, data in stats.items():
            print(f"  • {method}: {data['success_rate']:.1%} success, {data['avg_time_ms']:.1f}ms avg")

if __name__ == "__main__":
    test_ultra_fast_ocr() 