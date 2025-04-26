#!/usr/bin/env python3
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

class CustomLicensePlateDetector:
    def __init__(self, weights_path=None, confidence_threshold=0.5, device=None):
        """
        Initialize the custom license plate detector.
        
        Args:
            weights_path (str): Path to the trained YOLOv5 weights.
            confidence_threshold (float): Confidence threshold for detections.
            device (torch.device): Device to run inference on.
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model from weights
        try:
            from ultralytics import YOLO
            
            # Default model path if not provided
            if not weights_path:
                # Check if custom model exists
                default_paths = [
                    'license_plate_model.pt',
                    'license_plate_model.pt',
                    'data/license_plate_data/weights/best.pt',
                    '../data/license_plate_data/weights/best.pt',
                ]
                
                for path in default_paths:
                    if os.path.exists(path):
                        weights_path = path
                        break
                
                # If no model found, use YOLOv5s default model
                if not weights_path:
                    weights_path = 'yolov5s.pt'
                    print("No custom model found, using default YOLOv5s")
            
            print(f"Loading model from {weights_path}")
            self.model = YOLO(weights_path)
            print(f"Model loaded successfully")
            
        except ImportError:
            print("Error: ultralytics package not found. Please install it with: pip install ultralytics")
            sys.exit(1)
    
    def detect(self, image):
        """
        Detect license plates in an image.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            List of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        # Run inference
        results = self.model(image, conf=self.confidence_threshold)
        
        # Extract license plate detections
        license_plates = []
        
        if len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    
                    # Add detection to list
                    license_plates.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
        
        # If no license plates detected and this appears to be a direct license plate image,
        # use the entire image as a fallback
        if not license_plates and self._is_likely_plate_image(image):
            h, w = image.shape[:2]
            license_plates.append([0, 0, w, h, 0.9, 0])  # Use high confidence for direct plate image
        
        return license_plates
    
    def _is_likely_plate_image(self, image):
        """
        Check if an image is likely a direct image of a license plate.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            bool: True if image likely contains only a license plate.
        """
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # License plates typically have aspect ratios between 2:1 and 5:1
        if 1.5 <= aspect_ratio <= 5.0:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection to find text-like features
            edges = cv2.Canny(gray, 100, 200)
            
            # Count edge pixels
            edge_pixels = np.sum(edges > 0)
            total_pixels = h * w
            edge_ratio = edge_pixels / total_pixels
            
            # License plates typically have many edges due to characters
            if edge_ratio > 0.05:
                return True
        
        return False
    
    def train(self, data_yaml_path, epochs=100, batch_size=16, img_size=640, weights='yolov5s.pt'):
        """
        Train a custom license plate detector.
        
        Args:
            data_yaml_path (str): Path to data YAML file.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            img_size (int): Input image size.
            weights (str): Initial weights for training (can be pretrained model).
            
        Returns:
            str: Path to best trained weights.
        """
        try:
            import torch
            from ultralytics import YOLO
            
            # Ensure data YAML file exists
            if not os.path.exists(data_yaml_path):
                print(f"Error: Data YAML file not found at {data_yaml_path}")
                return None
            
            print(f"Starting training with data from {data_yaml_path}")
            print(f"Training for {epochs} epochs with batch size {batch_size}")
            
            # Check CUDA availability
            device = 0 if torch.cuda.is_available() else 'cpu'
            print(f"Training on device: {device} ({'GPU' if device == 0 else 'CPU'})")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU device: {torch.cuda.get_device_name(0)}")
            
            # Load a YOLOv5 model
            if weights.endswith('.pt') and os.path.exists(weights):
                model = YOLO(weights)
                print(f"Using pretrained weights from {weights}")
            else:
                # Use a default model from ultralytics
                model = YOLO(weights)
                print(f"Using model: {weights}")
            
            # Set up training parameters
            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device=device,
                project='license_plate_detection',
                name='train',
                exist_ok=True,
                patience=15,  # Early stopping patience
                workers=0  # Disable multiprocessing to avoid file mapping errors
            )
            
            # Get best weights path
            best_weights = results.best
            
            print(f"Training completed. Best weights saved to {best_weights}")
            return best_weights
            
        except Exception as e:
            print(f"Error during training: {e}")
            return None 