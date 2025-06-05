#!/usr/bin/env python3
"""
CUSTOM YOLO SYMBOL DETECTION SYSTEM
Advanced symbol detection for crypto trading charts

Features:
- YOLOv8-based symbol detection
- Multi-class crypto symbol recognition
- Bounding box + confidence scoring
- Real-time inference optimization
- Training data pipeline
"""

import cv2
import numpy as np
import torch
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# Try to import YOLO (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SymbolDetection:
    """Symbol detection result"""
    symbol: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    processing_time_ms: float

@dataclass
class TrainingData:
    """Training data structure"""
    image_path: str
    annotations: List[Dict[str, Any]]
    symbol_count: int

class YOLOSymbolDetector:
    """Custom YOLO Symbol Detection System"""
    
    def __init__(self, model_path: Optional[str] = None):
        # Supported crypto symbols (can be expanded)
        self.symbol_classes = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE', 
            'AVAX', 'MATIC', 'LINK', 'UNI', 'LTC', 'TRX', 'ATOM'
        ]
        
        self.num_classes = len(self.symbol_classes)
        self.class_to_id = {symbol: idx for idx, symbol in enumerate(self.symbol_classes)}
        self.id_to_class = {idx: symbol for symbol, idx in self.class_to_id.items()}
        
        # Model configuration
        self.model_path = model_path or 'models/yolo_symbol_detector.pt'
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        self.max_detections = 10
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'avg_processing_time': 0.0,
            'class_detections': {symbol: 0 for symbol in self.symbol_classes}
        }
        
        # Load model if available
        self.load_model()
        
        print(f"🎯 YOLO Symbol Detector initialized")
        print(f"   • Classes: {len(self.symbol_classes)} symbols")
        print(f"   • Device: {self.device}")
        print(f"   • Model: {'✅ Loaded' if self.model else '❌ Not available'}")
    
    def load_model(self):
        """Load trained YOLO model"""
        try:
            if not YOLO_AVAILABLE:
                print("⚠️ YOLO not available - using fallback detection")
                return False
            
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                print(f"✅ Loaded custom YOLO model: {self.model_path}")
                return True
            else:
                # Start with YOLOv8 base model for fine-tuning
                print("🔄 Custom model not found, initializing YOLOv8n...")
                self.model = YOLO('yolov8n.pt')  # Nano version for speed
                print("✅ YOLOv8n base model loaded (ready for training)")
                return True
                
        except Exception as e:
            print(f"❌ Model loading error: {str(e)}")
            self.model = None
            return False
    
    def detect_symbols(self, frame: np.ndarray, detailed_analysis: bool = True) -> List[SymbolDetection]:
        """Detect crypto symbols in trading chart"""
        start_time = time.time()
        
        try:
            if self.model is None:
                return self._fallback_detection(frame)
            
            # Prepare frame for YOLO
            results = self.model(frame, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               max_det=self.max_detections,
                               verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Map class ID to symbol
                        symbol = self.id_to_class.get(class_id, 'UNKNOWN')
                        
                        # Calculate center point
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        detection = SymbolDetection(
                            symbol=symbol,
                            confidence=float(confidence),
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center=(center_x, center_y),
                            processing_time_ms=(time.time() - start_time) * 1000
                        )
                        
                        detections.append(detection)
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_detection_stats(detections, processing_time)
            
            return detections
            
        except Exception as e:
            print(f"❌ YOLO detection error: {str(e)}")
            return self._fallback_detection(frame)
    
    def get_best_symbol(self, frame: np.ndarray) -> Optional[SymbolDetection]:
        """Get the most confident symbol detection"""
        detections = self.detect_symbols(frame)
        
        if detections:
            return detections[0]  # Highest confidence
        return None
    
    def _fallback_detection(self, frame: np.ndarray) -> List[SymbolDetection]:
        """Fallback detection when YOLO is not available"""
        # Simple pattern-based detection (placeholder)
        h, w = frame.shape[:2]
        
        # Mock detection for testing
        mock_detection = SymbolDetection(
            symbol='BTC',  # Default fallback
            confidence=0.3,  # Low confidence for fallback
            bbox=(10, 10, 100, 50),
            center=(55, 30),
            processing_time_ms=50.0
        )
        
        return [mock_detection]
    
    def _update_detection_stats(self, detections: List[SymbolDetection], processing_time: float):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += 1
        
        if detections:
            self.detection_stats['successful_detections'] += 1
            
            # Update class counts
            for detection in detections:
                if detection.symbol in self.detection_stats['class_detections']:
                    self.detection_stats['class_detections'][detection.symbol] += 1
        
        # Update average processing time
        total = self.detection_stats['total_detections']
        current_avg = self.detection_stats['avg_processing_time']
        self.detection_stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total
    
    def visualize_detections(self, frame: np.ndarray, detections: List[SymbolDetection]) -> np.ndarray:
        """Visualize symbol detections on frame"""
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            color = (0, 255, 0) if detection.confidence > 0.7 else (0, 255, 255) if detection.confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.symbol} {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Center point
            cv2.circle(vis_frame, detection.center, 3, color, -1)
        
        return vis_frame
    
    def get_detection_stats(self) -> Dict:
        """Get detection performance statistics"""
        success_rate = 0.0
        if self.detection_stats['total_detections'] > 0:
            success_rate = self.detection_stats['successful_detections'] / self.detection_stats['total_detections']
        
        return {
            'total_detections': self.detection_stats['total_detections'],
            'success_rate': success_rate,
            'avg_processing_time_ms': self.detection_stats['avg_processing_time'],
            'detections_per_minute': 60000 / max(self.detection_stats['avg_processing_time'], 1),
            'class_detections': self.detection_stats['class_detections'].copy(),
            'most_detected': max(self.detection_stats['class_detections'].items(), 
                               key=lambda x: x[1], default=('NONE', 0))[0]
        }

# Training Data Collection System
class YOLOTrainingDataCollector:
    """Collect and prepare training data for YOLO model"""
    
    def __init__(self, data_dir: str = "yolo_training_data"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        self.symbol_classes = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE', 
            'AVAX', 'MATIC', 'LINK', 'UNI', 'LTC', 'TRX', 'ATOM'
        ]
        
        self.class_to_id = {symbol: idx for idx, symbol in enumerate(self.symbol_classes)}
        
        # Create classes.txt file
        with open(os.path.join(data_dir, "classes.txt"), "w") as f:
            for symbol in self.symbol_classes:
                f.write(f"{symbol}\n")
        
        print(f"📊 Training data collector initialized: {data_dir}")
    
    def collect_training_frame(self, frame: np.ndarray, symbol: str, bbox: Tuple[int, int, int, int]):
        """Collect a training frame with annotation"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Save image
            image_filename = f"{symbol}_{timestamp}.jpg"
            image_path = os.path.join(self.images_dir, image_filename)
            cv2.imwrite(image_path, frame)
            
            # Create YOLO annotation
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            center_x = (x1 + x2) / 2.0 / w
            center_y = (y1 + y2) / 2.0 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            class_id = self.class_to_id.get(symbol, 0)
            
            # Save annotation
            label_filename = f"{symbol}_{timestamp}.txt"
            label_path = os.path.join(self.labels_dir, label_filename)
            
            with open(label_path, "w") as f:
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            print(f"✅ Training data collected: {symbol} at {bbox}")
            return True
            
        except Exception as e:
            print(f"❌ Training data collection error: {str(e)}")
            return False
    
    def create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training"""
        dataset_config = {
            'path': os.path.abspath(self.data_dir),
            'train': 'images',
            'val': 'images',  # Same as train for small datasets
            'nc': len(self.symbol_classes),
            'names': self.symbol_classes
        }
        
        yaml_path = os.path.join(self.data_dir, "dataset.yaml")
        import yaml
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"📝 Dataset configuration created: {yaml_path}")
        return yaml_path

def test_yolo_symbol_detector():
    """Test YOLO symbol detection system"""
    print("🎯 YOLO SYMBOL DETECTOR TEST")
    print("="*60)
    
    detector = YOLOSymbolDetector()
    
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
            
            # Detect symbols every 10 frames
            if frame_count % 10 == 0:
                print(f"\n🎯 Frame {frame_count} YOLO Detection:")
                
                detections = detector.detect_symbols(frame)
                
                if detections:
                    for detection in detections:
                        print(f"   📊 Symbol: {detection.symbol} ({detection.confidence:.2f})")
                        print(f"   📍 BBox: {detection.bbox}")
                        print(f"   ⚡ Time: {detection.processing_time_ms:.1f}ms")
                else:
                    print("   ❌ No symbols detected")
            
            # Visualize detections
            detections = detector.detect_symbols(frame)
            vis_frame = detector.visualize_detections(frame, detections)
            
            # Add info overlay
            cv2.putText(vis_frame, f"YOLO Symbol Detector - Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if detections:
                best = detections[0]
                cv2.putText(vis_frame, f"Best: {best.symbol} ({best.confidence:.2f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('YOLO Symbol Detection', vis_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save training data (manual annotation needed)
                print("💾 Save training data - manual annotation required")
    
    except KeyboardInterrupt:
        print("\n⏹️ Test stopped")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        stats = detector.get_detection_stats()
        print(f"\n📊 DETECTION STATISTICS:")
        print(f"   • Total detections: {stats['total_detections']}")
        print(f"   • Success rate: {stats['success_rate']:.1%}")
        print(f"   • Avg time: {stats['avg_processing_time_ms']:.1f}ms")
        print(f"   • Most detected: {stats['most_detected']}")

if __name__ == "__main__":
    test_yolo_symbol_detector() 