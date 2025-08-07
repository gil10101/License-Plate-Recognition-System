#!/usr/bin/env python3
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

class CustomLicensePlateDetector:
    def __init__(self, weights_path=None, confidence_threshold=0.25, device=None):
        """
        Initialize the custom license plate detector.
        
        Args:
            weights_path (str): Path to the trained YOLOv5 weights.
            confidence_threshold (float): Confidence threshold for detections.
            device (torch.device): Device to run inference on.
        """
        # Lowered the default confidence threshold from 0.5 to 0.25 to capture more potential plates
        self.confidence_threshold = confidence_threshold
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model from weights
        try:
            from ultralytics import YOLO
            
            # Default model path if not provided
            if not weights_path:
                # Check if custom model exists
                default_paths = [
                    'yolov5n.pt',  # Try smaller model first (faster)
                    'yolov5s.pt',
                    'license_plate_model.pt',
                    'data/license_plate_data/weights/best.pt',
                    '../data/license_plate_data/weights/best.pt',
                    'license_plate_detection/train/weights/best.pt'
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
        # Store original image dimensions
        original_height, original_width = image.shape[:2]
        
        # Run multiple detection passes with different configurations for better results
        license_plates = []
        
        # First pass: standard detection with configured confidence threshold
        results = self.model(image, conf=self.confidence_threshold)
        
        # Extract license plate detections
        if len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    
                    # Increase confidence for detections with proper license plate aspect ratio
                    aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                    if 1.8 <= aspect_ratio <= 5.0:  # Common license plate aspect ratios
                        conf = min(conf + 0.1, 1.0)  # Boost confidence but cap at 1.0
                    
                    # Add detection to list
                    license_plates.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
        
        # Second pass: Try with a lower confidence threshold if no plates found
        if not license_plates:
            lower_conf = max(0.15, self.confidence_threshold - 0.1)
            results = self.model(image, conf=lower_conf)
            
            if len(results) > 0:
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        # Increase confidence for detections with proper license plate aspect ratio
                        aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                        if 1.8 <= aspect_ratio <= 5.0:
                            conf = min(conf + 0.1, 1.0)
                        
                        # Add detection to list
                        license_plates.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
        
        # If still no license plates detected and this appears to be a direct license plate image,
        # use the entire image as a fallback
        if not license_plates and self._is_likely_plate_image(image):
            h, w = image.shape[:2]
            license_plates.append([0, 0, w, h, 0.85, 0])  # Slightly lower confidence (0.85) for direct plate image
        
        # Additional check for images that might not be recognized 
        # but have the characteristics of a license plate
        if not license_plates:
            # Try edge-based detection as a last resort
            possible_plates = self._detect_plates_with_edges(image)
            if possible_plates:
                license_plates.extend(possible_plates)
        
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
        
        # License plates typically have aspect ratios between 1.8:1 and 5:1
        if 1.8 <= aspect_ratio <= 5.0:
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
            
            # Also check for horizontal lines which are common in license plates
            # Apply morphological operations to connect horizontal edges
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
            
            # Count horizontal line pixels
            h_line_pixels = np.sum(horizontal_lines > 0)
            h_line_ratio = h_line_pixels / total_pixels
            
            if h_line_ratio > 0.04:  # If there are significant horizontal lines
                return True
                
        return False
    
    def _detect_plates_with_edges(self, image):
        """
        Detect license plates using edge detection when the model fails.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            list: List of potential license plate detections.
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply morphological operations to enhance edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for license plate candidates
        plate_detections = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Check if contour has license plate characteristics
            if (1.8 <= aspect_ratio <= 5.0 and 
                w > 60 and  # Minimum width
                h > 15 and  # Minimum height
                w < image.shape[1] * 0.9 and  # Not too wide
                h < image.shape[0] * 0.2):    # Not too tall
                
                # Calculate confidence based on characteristics
                conf = 0.6  # Base confidence for edge detection
                
                # Boost confidence for better aspect ratios
                if 2.0 <= aspect_ratio <= 4.5:
                    conf += 0.1
                
                # Boost confidence for larger plates
                if w > 100 and h > 30:
                    conf += 0.1
                
                # Add extra padding around the detected area
                padding_x = int(w * 0.05)
                padding_y = int(h * 0.1)
                
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(image.shape[1], x + w + padding_x)
                y2 = min(image.shape[0], y + h + padding_y)
                
                plate_detections.append([x1, y1, x2, y2, conf, 0])
        
        return plate_detections
    
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