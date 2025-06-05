#!/usr/bin/env python3
"""
YOLO TRAINING PIPELINE
Training pipeline for custom crypto symbol detection model

Features:
- Automated data collection from camera
- Data augmentation and preparation  
- YOLOv8 model training
- Model evaluation and deployment
"""

import cv2
import numpy as np
import os
import json
import time
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Try to import YOLO training tools
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

from yolo_symbol_detector import YOLOTrainingDataCollector

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    patience: int = 50
    device: str = 'auto'  # auto, cpu, cuda
    workers: int = 8
    learning_rate: float = 0.01

class YOLOTrainingPipeline:
    """Complete YOLO training pipeline for symbol detection"""
    
    def __init__(self, data_dir: str = "yolo_training_data", model_output_dir: str = "models"):
        self.data_dir = data_dir
        self.model_output_dir = model_output_dir
        
        # Create output directory
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Initialize data collector
        self.data_collector = YOLOTrainingDataCollector(data_dir)
        
        # Training config
        self.config = TrainingConfig()
        
        # Symbol classes for detection
        self.symbol_classes = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE', 
            'AVAX', 'MATIC', 'LINK', 'UNI', 'LTC', 'TRX', 'ATOM'
        ]
        
        print(f"🎯 YOLO Training Pipeline initialized")
        print(f"   • Data directory: {data_dir}")
        print(f"   • Model output: {model_output_dir}")
        print(f"   • Classes: {len(self.symbol_classes)} symbols")
    
    def collect_training_data_interactive(self, target_samples_per_class: int = 50):
        """Interactive data collection from camera"""
        print(f"📊 INTERACTIVE DATA COLLECTION")
        print(f"   • Target: {target_samples_per_class} samples per class")
        print(f"   • Total target: {target_samples_per_class * len(self.symbol_classes)} samples")
        print("\n🎮 CONTROLS:")
        print("   • Click and drag to create bounding box")
        print("   • 'b': BTC  'e': ETH  's': SOL  'a': ADA")
        print("   • 'n': BNB  'x': XRP  'd': DOT  'o': DOGE")
        print("   • 'v': AVAX  'm': MATIC  'l': LINK  'u': UNI")
        print("   • 'c': LTC  't': TRX  'r': ATOM")
        print("   • 'q': Quit collection")
        
        # Symbol key mappings
        key_to_symbol = {
            ord('b'): 'BTC', ord('e'): 'ETH', ord('s'): 'SOL', ord('a'): 'ADA',
            ord('n'): 'BNB', ord('x'): 'XRP', ord('d'): 'DOT', ord('o'): 'DOGE',
            ord('v'): 'AVAX', ord('m'): 'MATIC', ord('l'): 'LINK', ord('u'): 'UNI',
            ord('c'): 'LTC', ord('t'): 'TRX', ord('r'): 'ATOM'
        }
        
        # Collection stats
        collection_stats = {symbol: 0 for symbol in self.symbol_classes}
        
        # Mouse callback for bounding box selection
        drawing = False
        bbox_start = None
        bbox_end = None
        current_frame = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, bbox_start, bbox_end
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                bbox_start = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    bbox_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                bbox_end = (x, y)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not available")
            return False
        
        cv2.namedWindow('YOLO Training Data Collection')
        cv2.setMouseCallback('YOLO Training Data Collection', mouse_callback)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame = frame.copy()
                vis_frame = frame.copy()
                
                # Draw bounding box if being selected
                if bbox_start and bbox_end:
                    cv2.rectangle(vis_frame, bbox_start, bbox_end, (0, 255, 0), 2)
                
                # Draw collection stats
                y_pos = 30
                cv2.putText(vis_frame, "📊 COLLECTION STATS:", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 30
                
                for symbol in self.symbol_classes[:8]:  # First 8 symbols
                    count = collection_stats[symbol]
                    color = (0, 255, 0) if count >= target_samples_per_class else (0, 255, 255)
                    cv2.putText(vis_frame, f"{symbol}: {count}/{target_samples_per_class}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_pos += 25
                
                # Second column for remaining symbols
                y_pos = 60
                for symbol in self.symbol_classes[8:]:
                    count = collection_stats[symbol]
                    color = (0, 255, 0) if count >= target_samples_per_class else (0, 255, 255)
                    cv2.putText(vis_frame, f"{symbol}: {count}/{target_samples_per_class}", 
                               (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_pos += 25
                
                # Instructions
                cv2.putText(vis_frame, "Select bbox and press symbol key", 
                           (10, vis_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(vis_frame, "Press 'q' to finish collection", 
                           (10, vis_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow('YOLO Training Data Collection', vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key in key_to_symbol:
                    # Save training sample
                    if bbox_start and bbox_end and current_frame is not None:
                        symbol = key_to_symbol[key]
                        
                        # Convert bbox to (x1, y1, x2, y2) format
                        x1, y1 = bbox_start
                        x2, y2 = bbox_end
                        
                        # Ensure correct order
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1
                        
                        bbox = (x1, y1, x2, y2)
                        
                        # Save training data
                        success = self.data_collector.collect_training_frame(current_frame, symbol, bbox)
                        
                        if success:
                            collection_stats[symbol] += 1
                            print(f"✅ Collected {symbol} sample: {collection_stats[symbol]}/{target_samples_per_class}")
                            
                            # Reset bbox
                            bbox_start = None
                            bbox_end = None
                        else:
                            print(f"❌ Failed to collect {symbol} sample")
        
        except KeyboardInterrupt:
            print("\n⏹️ Data collection stopped")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Print final stats
        total_collected = sum(collection_stats.values())
        print(f"\n📊 COLLECTION SUMMARY:")
        print(f"   • Total samples: {total_collected}")
        for symbol, count in collection_stats.items():
            print(f"   • {symbol}: {count} samples")
        
        return total_collected > 0
    
    def prepare_dataset(self):
        """Prepare dataset for YOLO training"""
        print(f"📝 PREPARING DATASET...")
        
        # Create dataset.yaml
        dataset_yaml = self.data_collector.create_dataset_yaml()
        
        # Count data samples
        images_dir = os.path.join(self.data_dir, "images")
        labels_dir = os.path.join(self.data_dir, "labels")
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        
        print(f"   • Images: {len(image_files)}")
        print(f"   • Labels: {len(label_files)}")
        print(f"   • Dataset config: {dataset_yaml}")
        
        return len(image_files) > 0 and len(label_files) > 0
    
    def train_model(self, config: Optional[TrainingConfig] = None):
        """Train YOLO model"""
        if not YOLO_AVAILABLE:
            print("❌ YOLO not available for training")
            return None
        
        if config:
            self.config = config
        
        print(f"🚀 STARTING YOLO TRAINING...")
        print(f"   • Epochs: {self.config.epochs}")
        print(f"   • Batch size: {self.config.batch_size}")
        print(f"   • Image size: {self.config.image_size}")
        print(f"   • Device: {self.config.device}")
        
        try:
            # Initialize YOLOv8 model
            model = YOLO('yolov8n.pt')  # Start with nano model for speed
            
            # Dataset path
            dataset_yaml = os.path.join(self.data_dir, "dataset.yaml")
            
            # Train the model
            results = model.train(
                data=dataset_yaml,
                epochs=self.config.epochs,
                batch=self.config.batch_size,
                imgsz=self.config.image_size,
                patience=self.config.patience,
                device=self.config.device,
                workers=self.config.workers,
                lr0=self.config.learning_rate,
                project=self.model_output_dir,
                name='yolo_symbol_detector',
                exist_ok=True
            )
            
            # Save the trained model
            model_path = os.path.join(self.model_output_dir, 'yolo_symbol_detector.pt')
            model.save(model_path)
            
            print(f"✅ Training completed!")
            print(f"   • Model saved: {model_path}")
            print(f"   • Results: {results}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return None
    
    def evaluate_model(self, model_path: str):
        """Evaluate trained model"""
        if not YOLO_AVAILABLE:
            print("❌ YOLO not available for evaluation")
            return None
        
        try:
            print(f"📊 EVALUATING MODEL: {model_path}")
            
            model = YOLO(model_path)
            
            # Validate on test data
            dataset_yaml = os.path.join(self.data_dir, "dataset.yaml")
            results = model.val(data=dataset_yaml)
            
            print(f"✅ Evaluation completed!")
            print(f"   • mAP50: {results.box.map50:.3f}")
            print(f"   • mAP50-95: {results.box.map:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation error: {str(e)}")
            return None
    
    def export_model(self, model_path: str, format: str = 'onnx'):
        """Export model to different formats"""
        if not YOLO_AVAILABLE:
            print("❌ YOLO not available for export")
            return None
        
        try:
            print(f"📦 EXPORTING MODEL to {format.upper()}...")
            
            model = YOLO(model_path)
            exported_model = model.export(format=format)
            
            print(f"✅ Model exported: {exported_model}")
            return exported_model
            
        except Exception as e:
            print(f"❌ Export error: {str(e)}")
            return None

def run_training_pipeline():
    """Run complete YOLO training pipeline"""
    print("🎯 YOLO SYMBOL DETECTION TRAINING PIPELINE")
    print("="*70)
    
    pipeline = YOLOTrainingPipeline()
    
    # Step 1: Collect training data
    print("\n📊 STEP 1: DATA COLLECTION")
    if input("Collect new training data? (y/n): ").lower() == 'y':
        target_samples = int(input("Samples per class (default 20): ") or "20")
        
        success = pipeline.collect_training_data_interactive(target_samples)
        if not success:
            print("❌ Data collection failed")
            return
    
    # Step 2: Prepare dataset
    print("\n📝 STEP 2: DATASET PREPARATION")
    if not pipeline.prepare_dataset():
        print("❌ Dataset preparation failed")
        return
    
    # Step 3: Train model
    print("\n🚀 STEP 3: MODEL TRAINING")
    if input("Start training? (y/n): ").lower() == 'y':
        
        # Configure training
        config = TrainingConfig()
        config.epochs = int(input(f"Epochs (default {config.epochs}): ") or config.epochs)
        config.batch_size = int(input(f"Batch size (default {config.batch_size}): ") or config.batch_size)
        
        model_path = pipeline.train_model(config)
        
        if model_path:
            print(f"✅ Training successful: {model_path}")
            
            # Step 4: Evaluate model
            print("\n📊 STEP 4: MODEL EVALUATION")
            if input("Evaluate model? (y/n): ").lower() == 'y':
                pipeline.evaluate_model(model_path)
            
            # Step 5: Export model
            print("\n📦 STEP 5: MODEL EXPORT")
            if input("Export model? (y/n): ").lower() == 'y':
                export_format = input("Export format (onnx/torchscript/tflite): ") or "onnx"
                pipeline.export_model(model_path, export_format)
        
        else:
            print("❌ Training failed")

if __name__ == "__main__":
    run_training_pipeline() 