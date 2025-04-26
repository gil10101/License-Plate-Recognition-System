import torch
import torchvision
import numpy as np
import cv2
import os
import sys
import glob
from pathlib import Path

class LicensePlateDetector:
    def __init__(self, model_type='yolov5', confidence_threshold=0.5, device=None, custom_weights_path=None):
        """
        Initialize the license plate detector.
        
        Args:
            model_type (str): Type of model to use. Currently supports 'yolov5'.
            confidence_threshold (float): Minimum confidence for detections.
            device (torch.device): Device to run inference on.
            custom_weights_path (str): Path to custom weights file.
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.custom_weights_path = custom_weights_path
        
        # Load YOLOv5 model using ultralytics package instead of torch hub
        if model_type == 'yolov5':
            try:
                # Try using the ultralytics package (newer approach)
                from ultralytics import YOLO
                
                # If custom weights are provided, use them
                if custom_weights_path and os.path.exists(custom_weights_path):
                    self.model = YOLO(custom_weights_path)
                else:
                    self.model = YOLO('yolov5s.pt')
                
                self.use_ultralytics = True
            except (ImportError, ModuleNotFoundError):
                # Fall back to torch hub if ultralytics package is not available
                # Temporarily modify sys.path to avoid namespace collision with our utils module
                original_path = sys.path.copy()
                if '' in sys.path:
                    sys.path.remove('')  # Remove current directory from path to avoid namespace conflict
                
                try:
                    # If custom weights are provided, use them
                    if custom_weights_path and os.path.exists(custom_weights_path):
                        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_weights_path)
                    else:
                        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                    
                    self.model.to(self.device)
                    self.model.eval()
                    # Set confidence threshold
                    self.model.conf = confidence_threshold
                    # Set classes to detect (car = 2, truck = 7)
                    self.classes = [2, 7]  # car and truck in COCO dataset
                    self.model.classes = self.classes
                    self.use_ultralytics = False
                finally:
                    # Restore original path
                    sys.path = original_path
        
        # Load license plate templates from training data
        self.plate_templates = self._load_plate_templates()

    def _load_plate_templates(self):
        """
        Load license plate templates from training data to improve detection.
        
        Returns:
            list: List of license plate template images.
        """
        templates = []
        
        # Try to find training data directories
        train_dirs = [
            'data/train',
            'data/val',
            '../data/train',
            '../data/val',
            './data/train',
            './data/val'
        ]
        
        for train_dir in train_dirs:
            if os.path.exists(train_dir):
                # Load some of the license plate images as templates
                image_files = glob.glob(os.path.join(train_dir, "*.png"))
                
                # Limit the number of templates to avoid memory issues
                max_templates = min(20, len(image_files))
                
                for i in range(max_templates):
                    try:
                        img = cv2.imread(image_files[i])
                        if img is not None:
                            # Convert to grayscale for template matching
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            templates.append(gray)
                    except Exception as e:
                        print(f"Error loading template {image_files[i]}: {e}")
        
        print(f"Loaded {len(templates)} license plate templates")
        return templates

    def detect(self, image):
        """
        Detect license plates in an image.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            List of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        original_height, original_width = image.shape[:2]
        
        # First, try to detect if this is a full traffic scene versus a close-up license plate
        is_traffic_scene = self._is_traffic_scene(image)
        
        if hasattr(self, 'use_ultralytics') and self.use_ultralytics:
            # Using the newer ultralytics YOLO approach
            # For traffic scenes, run vehicle detection followed by plate detection
            if is_traffic_scene:
                # First detect vehicles
                results = self.model(image, conf=self.confidence_threshold, classes=[2, 7])  # car and truck classes
                
                # Process vehicle detections to extract closest/biggest vehicle first
                license_plates = []
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    
                    # Sort boxes by area (descending) to prioritize closest/largest vehicles
                    box_data = []
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = box.cls[0].cpu().numpy()
                        
                        # Calculate area and distance from bottom of image (closer to camera)
                        area = (x2 - x1) * (y2 - y1)
                        # distance_from_bottom is a proxy for how close the vehicle is to the camera
                        distance_from_bottom = original_height - y2
                        
                        box_data.append({
                            'idx': i,
                            'box': box,
                            'area': area,
                            'distance': distance_from_bottom,
                            'coords': (x1, y1, x2, y2),
                            'conf': conf,
                            'class_id': class_id
                        })
                    
                    # Sort boxes by:
                    # 1. First prioritize vehicles closer to bottom of frame (closer to camera)
                    # 2. Then by size (larger vehicles are typically closer)
                    box_data.sort(key=lambda x: (x['distance'], -x['area']))
                    
                    # Process all vehicles in order of priority (closest/largest first)
                    for box_info in box_data:
                        x1, y1, x2, y2 = box_info['coords']
                        conf = box_info['conf']
                        
                        # Get vehicle region
                        vehicle_region = image[int(y1):int(y2), int(x1):int(x2)]
                        if vehicle_region.size == 0:
                            continue
                        
                        # First try general plate detection methods
                        plate_regions = self._locate_license_plate_in_vehicle(vehicle_region)
                        
                        # If we found potential plate regions, add them
                        for plate_box in plate_regions:
                            # Convert coordinates from vehicle region to original image
                            px1, py1, px2, py2 = plate_box
                            # Adjust coordinates to the original image
                            px1, py1 = px1 + int(x1), py1 + int(y1)
                            px2, py2 = px2 + int(x1), py2 + int(y1)
                            
                            # Ensure coordinates are within bounds
                            px1 = max(0, px1)
                            py1 = max(0, py1)
                            px2 = min(original_width, px2)
                            py2 = min(original_height, py2)
                            
                            if px2 > px1 and py2 > py1:
                                license_plates.append([px1, py1, px2, py2, conf, 0])
                
                # If we still don't have any license plates, try template matching
                if not license_plates and self.plate_templates:
                    template_plates = self._detect_plates_with_templates(image)
                    license_plates.extend(template_plates)
                
                # If we still don't have any license plates, try direct plate detection
                if not license_plates:
                    # Try direct edge-based license plate detection
                    edge_plates = self._detect_plates_with_edges(image)
                    license_plates.extend(edge_plates)
                
                # If we still don't have plates, use the bottom part of the first vehicle
                if not license_plates and len(box_data) > 0:
                    box_info = box_data[0]  # The closest/largest vehicle
                    x1, y1, x2, y2 = box_info['coords']
                    conf = box_info['conf']
                    
                    # Assuming license plate is in the bottom third of the vehicle
                    height = y2 - y1
                    width = x2 - x1
                    
                    # For sedan-style vehicles, the plate is typically in the bottom center
                    center_x = (x1 + x2) / 2
                    plate_width = width * 0.4  # License plates are about 40% of vehicle width
                    plate_height = height * 0.1  # License plates are about 10% of vehicle height
                    
                    plate_x1 = center_x - plate_width / 2
                    plate_x2 = center_x + plate_width / 2
                    plate_y1 = y2 - plate_height * 2  # Position slightly above the bottom
                    plate_y2 = y2 - plate_height / 2
                    
                    license_plates.append([plate_x1, plate_y1, plate_x2, plate_y2, conf, 0])
                
                return license_plates
            else:
                # For direct license plate images, try template matching first
                if self.plate_templates:
                    template_plates = self._detect_plates_with_templates(image)
                    if template_plates:
                        return template_plates
                
                # If template matching fails or no templates are available, use the whole image
                h, w = image.shape[:2]
                license_plates = [[0, 0, w, h, 0.9, 0]]  # Use high confidence for direct plate image
                return license_plates
                
        else:
            # Using the original torch hub approach
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(img_rgb)
            
            # Process results
            detections = results.xyxy[0].cpu().numpy()  # xyxy format (x1, y1, x2, y2, confidence, class)
            
            # Sort detections by area (descending) to prioritize closest/largest vehicles
            detection_data = []
            for detection in detections:
                x1, y1, x2, y2, conf, class_id = detection
                
                # Calculate area and distance from bottom of image (closer to camera)
                area = (x2 - x1) * (y2 - y1)
                distance_from_bottom = original_height - y2
                
                detection_data.append({
                    'detection': detection,
                    'area': area,
                    'distance': distance_from_bottom
                })
            
            # Sort detections by distance first, then area
            detection_data.sort(key=lambda x: (x['distance'], -x['area']))
            
            # Process detections to extract license plate regions
            license_plates = []
            for det_info in detection_data:
                x1, y1, x2, y2, conf, class_id = det_info['detection']
                
                # If this is a vehicle (car or truck), extract a region that might contain a license plate
                if int(class_id) in self.classes and conf >= self.confidence_threshold:
                    # Get vehicle region
                    vehicle_region = image[int(y1):int(y2), int(x1):int(x2)]
                    if vehicle_region.size == 0:
                        continue
                    
                    # Try standard license plate detection methods
                    plate_regions = self._locate_license_plate_in_vehicle(vehicle_region)
                    
                    # If we found potential plate regions, add them
                    for plate_box in plate_regions:
                        # Convert coordinates from vehicle region to original image
                        px1, py1, px2, py2 = plate_box
                        # Adjust coordinates to the original image
                        px1, py1 = px1 + int(x1), py1 + int(y1)
                        px2, py2 = px2 + int(x1), py2 + int(y1)
                        
                        # Ensure coordinates are within bounds
                        px1 = max(0, px1)
                        py1 = max(0, py1)
                        px2 = min(original_width, px2)
                        py2 = min(original_height, py2)
                        
                        if px2 > px1 and py2 > py1:
                            license_plates.append([px1, py1, px2, py2, conf, 0])
                    
                    # If we didn't find any plate regions with the advanced method,
                    # fall back to the basic method (bottom portion of vehicle)
                    if not plate_regions:
                        # Assuming license plate is in the bottom third of the vehicle
                        height = y2 - y1
                        plate_y1 = y2 - height / 3
                        plate_y2 = y2
                        
                        # Assuming license plate width is 60% of vehicle width
                        width = x2 - x1
                        center_x = (x1 + x2) / 2
                        plate_x1 = center_x - width * 0.3
                        plate_x2 = center_x + width * 0.3
                        
                        license_plates.append([plate_x1, plate_y1, plate_x2, plate_y2, conf, 0])
            
            # If no license plates detected through vehicle detection, try direct detection
            if not license_plates:
                # Try template matching first
                if self.plate_templates:
                    template_plates = self._detect_plates_with_templates(image)
                    if template_plates:
                        return template_plates
                
                # If template matching fails or no templates available, try edge detection
                edge_plates = self._detect_plates_with_edges(image)
                if edge_plates:
                    return edge_plates
                
                # Use the entire image as a last resort
                if not is_traffic_scene:
                    h, w = image.shape[:2]
                    license_plates.append([0, 0, w, h, 0.9, 0])  # Use high confidence for direct plate image
            
            return license_plates
    
    def _detect_plates_with_templates(self, image):
        """
        Detect license plates using template matching from training data.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            list: List of potential license plate regions as [x1, y1, x2, y2, conf, class_id].
        """
        if not self.plate_templates:
            return []
        
        height, width = image.shape[:2]
        plate_regions = []
        
        # Convert input image to grayscale for template matching
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold to make it easier to match templates
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        best_match_val = 0
        best_match_rect = None
        
        # Try template matching with different scales
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        for template in self.plate_templates:
            template_h, template_w = template.shape
            
            # Skip if template is too large
            if template_h > height or template_w > width:
                continue
                
            for scale in scales:
                # Resize template for scale-invariant matching
                if scale != 1.0:
                    resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
                    template_h, template_w = resized_template.shape
                    
                    # Skip if resized template is too large
                    if template_h > height or template_w > width:
                        continue
                else:
                    resized_template = template
                
                # Apply template matching
                res = cv2.matchTemplate(thresh, resized_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                # If we found a good match, save it
                if max_val > 0.3 and max_val > best_match_val:
                    best_match_val = max_val
                    best_match_rect = (max_loc[0], max_loc[1], 
                                      max_loc[0] + template_w, max_loc[1] + template_h)
        
        # If we found a good match, add it
        if best_match_rect is not None:
            x1, y1, x2, y2 = best_match_rect
            
            # Add some padding
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(width, x2 + padding_x)
            y2 = min(height, y2 + padding_y)
            
            # Confidence based on match value
            conf = best_match_val
            
            plate_regions.append([x1, y1, x2, y2, conf, 0])
        
        return plate_regions
    
    def _is_white_vehicle(self, vehicle_img):
        """
        Check if a vehicle is predominantly white.
        
        Args:
            vehicle_img (numpy.ndarray): Image of a vehicle.
            
        Returns:
            bool: True if the vehicle appears to be white, False otherwise.
        """
        if vehicle_img.size == 0:
            return False
            
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        
        # Define range for white color in HSV
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        
        # Create a mask for white pixels
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Calculate percentage of white pixels
        white_pixel_count = np.sum(white_mask > 0)
        total_pixels = vehicle_img.shape[0] * vehicle_img.shape[1]
        white_percentage = white_pixel_count / total_pixels * 100
        
        # If more than 40% of pixels are white, consider it a white vehicle
        return white_percentage > 40
    
    def _is_traffic_scene(self, image):
        """
        Determine if this is likely a traffic scene (as opposed to a close-up of a license plate).
        Uses basic heuristics based on image characteristics.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            bool: True if this appears to be a traffic scene, False if likely a close-up.
        """
        # Simple heuristic: If the image is very wide relative to height, likely not a plate
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # License plates typically have aspect ratios between 2:1 and 5:1
        if 1.5 <= aspect_ratio <= 5.0:
            # This could be a close-up of a license plate
            # Additional check: Look for text-like features
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # Count strong edges (might indicate text)
            strong_edges = np.sum(sobel > 128)
            edge_ratio = strong_edges / (h * w)
            
            # If there are many edges and it has license plate dimensions, likely a plate
            if edge_ratio > 0.1:
                return False  # Likely a license plate close-up
        
        # Check for common traffic scene characteristics (sky, road, multiple vehicles)
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Look for blue-ish pixels (sky)
        blue_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
        blue_pixels = np.sum(blue_mask > 0)
        blue_ratio = blue_pixels / (h * w)
        
        # Look for gray-ish pixels (road)
        gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 30, 200))
        gray_pixels = np.sum(gray_mask > 0)
        gray_ratio = gray_pixels / (h * w)
        
        # If we have significant sky or road areas, likely a traffic scene
        if blue_ratio > 0.1 or gray_ratio > 0.2:
            return True
        
        # By default, assume it's a traffic scene if it doesn't look like a close-up
        return True
    
    def _locate_license_plate_in_vehicle(self, vehicle_img):
        """
        Locate potential license plate regions within a vehicle image.
        Uses color, edge, and contour analysis to find regions that look like plates.
        
        Args:
            vehicle_img (numpy.ndarray): Image of a vehicle.
            
        Returns:
            list: List of potential license plate regions as [x1, y1, x2, y2].
        """
        if vehicle_img.size == 0 or vehicle_img.shape[0] < 10 or vehicle_img.shape[1] < 10:
            return []
        
        plate_regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get vehicle dimensions for filtering
        height, width = vehicle_img.shape[:2]
        
        # Filter contours to find potential license plates
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # License plate aspect ratio is typically between 1.5:1 and 5:1
            aspect_ratio = w / h if h > 0 else 0
            
            # Typical license plate size relative to vehicle
            relative_width = w / width
            relative_height = h / height
            
            # Check if the contour might be a license plate
            if (1.5 <= aspect_ratio <= 6.0 and 
                0.1 <= relative_width <= 0.9 and
                0.03 <= relative_height <= 0.3):
                
                # Additional check: Is it in the lower half of the vehicle?
                if y > height * 0.3:
                    # This looks like a potential license plate
                    # Add some padding
                    padding_x = int(w * 0.1)
                    padding_y = int(h * 0.1)
                    
                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(width, x + w + padding_x)
                    y2 = min(height, y + h + padding_y)
                    
                    plate_regions.append([x1, y1, x2, y2])
        
        # If we didn't find any regions with contour analysis,
        # try color-based detection (for white or yellow plates)
        if not plate_regions:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
            
            # Look for white-ish regions (common for many plates)
            white_mask = cv2.inRange(hsv, (0, 0, 150), (180, 50, 255))
            
            # Look for yellow-ish regions (common for some plates)
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
            
            # Combine masks
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            # Apply morphological operations
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the color mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by shape and location
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # Check if the color region might be a license plate
                if (1.5 <= aspect_ratio <= 6.0 and 
                    area > (width * height * 0.005) and  # Minimum size
                    area < (width * height * 0.2) and    # Maximum size
                    y > height * 0.3):                   # In lower part of vehicle
                    
                    # Add some padding
                    padding_x = int(w * 0.1)
                    padding_y = int(h * 0.1)
                    
                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(width, x + w + padding_x)
                    y2 = min(height, y + h + padding_y)
                    
                    plate_regions.append([x1, y1, x2, y2])
        
        return plate_regions
    
    def _detect_plates_with_edges(self, image):
        """
        Detect license plates directly using edge detection and contour analysis.
        Useful for images where vehicle detection may not work well.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            list: List of potential license plate regions as [x1, y1, x2, y2, conf, class_id].
        """
        height, width = image.shape[:2]
        plate_regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find potential license plates
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # License plate aspect ratio is typically between 1.5:1 and 6:1
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter by size and aspect ratio
            if (1.5 <= aspect_ratio <= 6.0 and 
                w > 60 and h > 15 and             # Minimum size
                w < width * 0.8 and h < height * 0.3):  # Maximum size
                
                # This looks like a potential license plate
                # Add some padding
                padding_x = int(w * 0.1)
                padding_y = int(h * 0.1)
                
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(width, x + w + padding_x)
                y2 = min(height, y + h + padding_y)
                
                # Assign a confidence value based on the aspect ratio and size
                conf = 0.5  # Base confidence
                
                # Plates with typical aspect ratios get higher confidence
                if 2.0 <= aspect_ratio <= 4.5:
                    conf += 0.1
                
                # Larger plates get higher confidence
                if w > 100 and h > 30:
                    conf += 0.1
                
                plate_regions.append([x1, y1, x2, y2, conf, 0])
        
        return plate_regions
    
    def train_with_custom_data(self, train_dir, val_dir, epochs=10, batch_size=16, img_size=640):
        """
        Train the license plate detector with custom data.
        
        Args:
            train_dir (str): Directory containing training data.
            val_dir (str): Directory containing validation data.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            img_size (int): Input image size for training.
            
        Returns:
            bool: True if training was successful, False otherwise.
        """
        try:
            from ultralytics import YOLO
            
            # Start with a pretrained model
            model = YOLO('yolov5s.pt')
            
            # Define training arguments
            train_args = {
                'data': 'license_plate_data.yaml',  # Create this YAML file
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': img_size,
                'patience': 5,
                'save': True
            }
            
            # Create YAML file for training
            yaml_content = f"""
            path: .
            train: {train_dir}
            val: {val_dir}
            
            nc: 1  # number of classes
            names: ['license_plate']  # class names
            """
            
            with open('license_plate_data.yaml', 'w') as f:
                f.write(yaml_content)
            
            # Train the model
            results = model.train(**train_args)
            
            # Update the current model with the trained weights
            best_weights = str(Path(results.save_dir) / 'weights' / 'best.pt')
            if os.path.exists(best_weights):
                self.model = YOLO(best_weights)
                self.custom_weights_path = best_weights
                return True
            
            return False
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def download_custom_weights(self, weights_url, save_path):
        """
        Download custom weights for the model.
        
        Args:
            weights_url (str): URL to download weights from.
            save_path (str): Path to save weights to.
        """
        import urllib.request
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Download weights
            urllib.request.urlretrieve(weights_url, save_path)
            
            # Update model with downloaded weights
            if os.path.exists(save_path):
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(save_path)
                except:
                    # Fallback to torch hub
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=save_path)
                
                self.custom_weights_path = save_path
                print(f"Successfully loaded weights from {save_path}")
                return True
            
            return False
        except Exception as e:
            print(f"Error downloading weights: {e}")
            return False 