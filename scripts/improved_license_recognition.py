#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import glob
import argparse
import pickle
import json
import pytesseract

# Set Tesseract path for Windows
if os.name == 'nt':  # Check if running on Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ImprovedLicensePlateRecognition:
    """
    Improved license plate recognition using a two-stage approach:
    1. Better license plate detection using a custom detector
    2. Enhanced OCR using preprocessing and the groundtruth data
    """
    
    def __init__(self, detector_weights=None, ocr_model_dir=None):
        """
        Initialize the license plate recognition system.
        
        Args:
            detector_weights (str): Path to the custom detector weights
            ocr_model_dir (str): Path to the OCR model directory
        """
        # Default paths
        if detector_weights is None:
            # Try to find detector weights
            candidate_weights = [
                'us_license_plate_detector.pt',
                'license_plate_model.pt',
                'yolov5su.pt'
            ]
            
            for weight_path in candidate_weights:
                if os.path.exists(weight_path):
                    detector_weights = weight_path
                    break
        
        if ocr_model_dir is None:
            ocr_model_dir = 'data/ocr_model'
            if not os.path.exists(ocr_model_dir):
                ocr_model_dir = None
        
        # Load license plate detector
        if detector_weights and os.path.exists(detector_weights):
            print(f"Loading license plate detector from {detector_weights}")
            try:
                from detector.custom_plate_detector import CustomLicensePlateDetector
                self.detector = CustomLicensePlateDetector(
                    weights_path=detector_weights,
                    confidence_threshold=0.25
                )
                print("License plate detector loaded successfully")
            except Exception as e:
                print(f"Error loading detector: {e}")
                self.detector = None
        else:
            print("Warning: No detector weights found, using direct image processing")
            self.detector = None
        
        # Load OCR model
        self.ocr_configs = None
        self.char_mappings = None
        
        if ocr_model_dir and os.path.exists(ocr_model_dir):
            try:
                # Load OCR configurations
                config_path = os.path.join(ocr_model_dir, 'ocr_configs.pkl')
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        self.ocr_configs = pickle.load(f)
                    print(f"Loaded {len(self.ocr_configs)} OCR configurations")
                
                # Load character mappings
                mapping_path = os.path.join(ocr_model_dir, 'char_mappings.json')
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'r') as f:
                        self.char_mappings = json.load(f)
                    print(f"Loaded {len(self.char_mappings)} character mappings")
            except Exception as e:
                print(f"Error loading OCR model: {e}")
        
        # Set default OCR config if none loaded
        if not self.ocr_configs:
            self.ocr_configs = [
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            ]
        
        # Set default character mappings if none loaded
        if not self.char_mappings:
            self.char_mappings = {
                '0': 'O', 'O': '0', '1': 'I', 'I': '1', '5': 'S', 'S': '5',
                '8': 'B', 'B': '8', 'Z': '2', '2': 'Z'
            }
    
    def recognize(self, image_path, debug=False):
        """
        Recognize license plate from an image.
        
        Args:
            image_path (str): Path to the image
            debug (bool): Whether to save debug images
            
        Returns:
            tuple: (license_plate_text, confidence, plate_image)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None, 0.0, None
        
        # Create debug directory if needed
        if debug:
            debug_dir = 'debug_output'
            os.makedirs(debug_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save original image
            cv2.imwrite(f"{debug_dir}/{base_name}_original.png", image)
        
        # Step 1: Detect license plate
        plate_img = self._detect_plate(image, debug, image_path)
        if plate_img is None:
            return "No plate detected", 0.0, None
        
        # Step 2: Apply OCR to recognize the text
        text, confidence = self._recognize_text(plate_img, debug, image_path)
        
        return text, confidence, plate_img
    
    def _detect_plate(self, image, debug=False, image_path=None):
        """
        Detect license plate in an image.
        
        Args:
            image (numpy.ndarray): Input image
            debug (bool): Whether to save debug images
            image_path (str): Path to the original image (for debug naming)
            
        Returns:
            numpy.ndarray: License plate image or None if not found
        """
        # Try using the detector if available
        if self.detector:
            # Get license plate detections
            detections = self.detector.detect(image)
            
            if detections and len(detections) > 0:
                # If we have detections, use the most confident one
                detections = sorted(detections, key=lambda d: d[4], reverse=True)
                best_detection = detections[0]
                
                # Extract coordinates
                x1, y1, x2, y2 = int(best_detection[0]), int(best_detection[1]), int(best_detection[2]), int(best_detection[3])
                
                # Extract the plate region with some padding
                h, w = image.shape[:2]
                
                # Add padding
                padding_x = int((x2 - x1) * 0.05)
                padding_y = int((y2 - y1) * 0.05)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(w, x2 + padding_x)
                y2 = min(h, y2 + padding_y)
                
                plate_img = image[y1:y2, x1:x2].copy()
                
                if debug and image_path:
                    debug_dir = 'debug_output'
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    
                    # Draw detection on original image
                    debug_img = image.copy()
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imwrite(f"{debug_dir}/{base_name}_detected.png", debug_img)
                    
                    # Save cropped plate
                    cv2.imwrite(f"{debug_dir}/{base_name}_plate.png", plate_img)
                
                return plate_img
        
        # Fallback to traditional image processing if detector failed or not available
        return self._find_plate_by_processing(image, debug, image_path)
    
    def _find_plate_by_processing(self, image, debug=False, image_path=None):
        """
        Find license plate using traditional image processing techniques.
        
        Args:
            image (numpy.ndarray): Input image
            debug (bool): Whether to save debug images
            image_path (str): Path to the original image (for debug naming)
            
        Returns:
            numpy.ndarray: License plate image or None if not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Save debug image
        if debug and image_path:
            debug_dir = 'debug_output'
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(f"{debug_dir}/{base_name}_gray.png", gray)
        
        # Apply bilateral filter to remove noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 200)
        
        if debug and image_path:
            debug_dir = 'debug_output'
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(f"{debug_dir}/{base_name}_edges.png", edges)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Plate candidates
        candidates = []
        
        # Aspect ratio range for license plates
        MIN_ASPECT_RATIO = 1.5
        MAX_ASPECT_RATIO = 6.0
        
        h, w = image.shape[:2]
        
        # Create debug image
        if debug and image_path:
            debug_dir = 'debug_output'
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            contour_img = image.copy()
        
        for contour in contours:
            # Get bounding rectangle
            x, y, rect_w, rect_h = cv2.boundingRect(contour)
            
            # Skip very small contours
            if rect_w < 60 or rect_h < 15:
                continue
            
            # Calculate aspect ratio
            aspect_ratio = float(rect_w) / rect_h if rect_h > 0 else 0
            
            # Draw contour for debugging
            if debug and image_path:
                cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
            
            # Check if the shape resembles a license plate
            if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
                # Calculate relative size
                area_ratio = (rect_w * rect_h) / (w * h)
                
                # License plates are typically not tiny or filling the entire image
                if 0.01 <= area_ratio <= 0.5:
                    # Score this candidate based on how likely it is to be a plate
                    # Higher score means more likely to be a license plate
                    edge_density = self._calculate_edge_density(edges[y:y+rect_h, x:x+rect_w])
                    text_like_score = self._calculate_text_like_score(gray[y:y+rect_h, x:x+rect_w])
                    
                    score = edge_density * 0.7 + text_like_score * 0.3
                    
                    # Add some padding around the plate
                    padding_x = int(rect_w * 0.05)
                    padding_y = int(rect_h * 0.1)
                    
                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(w, x + rect_w + padding_x)
                    y2 = min(h, y + rect_h + padding_y)
                    
                    candidates.append({
                        'score': score,
                        'box': (x1, y1, x2, y2)
                    })
                    
                    # Draw rectangle for debugging
                    if debug and image_path:
                        cv2.rectangle(contour_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Save contour debug image
        if debug and image_path:
            cv2.imwrite(f"{debug_dir}/{base_name}_contours.png", contour_img)
        
        # If we found candidates, return the one with the highest score
        if candidates:
            best_candidate = max(candidates, key=lambda c: c['score'])
            x1, y1, x2, y2 = best_candidate['box']
            
            # Extract the plate region
            plate_img = image[y1:y2, x1:x2].copy()
            
            # Save debug image
            if debug and image_path:
                cv2.imwrite(f"{debug_dir}/{base_name}_best_plate.png", plate_img)
            
            return plate_img
        
        # Fallback: use the central part of the image
        # Assume the plate is in the lower part of the image
        plate_width = int(w * 0.6)
        plate_height = int(h * 0.2)
        
        center_x = w // 2
        y_offset = int(h * 0.6)  # 60% from the top
        
        x1 = center_x - plate_width // 2
        y1 = y_offset
        x2 = x1 + plate_width
        y2 = y1 + plate_height
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract the fallback region
        plate_img = image[y1:y2, x1:x2].copy()
        
        # Save debug image
        if debug and image_path:
            debug_img = image.copy()
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(f"{debug_dir}/{base_name}_fallback.png", debug_img)
            cv2.imwrite(f"{debug_dir}/{base_name}_fallback_plate.png", plate_img)
        
        return plate_img
    
    def _calculate_edge_density(self, img):
        """Calculate the density of edges in an image region"""
        if img.size == 0:
            return 0
        
        # Count non-zero pixels (edges)
        edge_pixels = np.count_nonzero(img)
        total_pixels = img.size
        
        return edge_pixels / total_pixels
    
    def _calculate_text_like_score(self, img):
        """Calculate how likely an image region contains text"""
        if img.size == 0:
            return 0
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours that might be characters
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for character-like contours
        char_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Character aspect ratio and size check
            aspect_ratio = w / h if h > 0 else 0
            if 0.2 <= aspect_ratio <= 1.5 and h > img.shape[0] * 0.3:
                char_count += 1
        
        # More characters usually means more likely to be a plate
        # Most plates have between 5-10 characters
        if 3 <= char_count <= 12:
            return 0.8
        else:
            return 0.2 * (char_count / 8)  # Normalize around 8 characters
    
    def _preprocess_for_ocr(self, plate_img):
        """
        Preprocess the license plate image for better OCR results
        
        Args:
            plate_img (numpy.ndarray): License plate image
            
        Returns:
            list: List of preprocessed image versions
        """
        preprocessed = []
        
        # Check if image is valid
        if plate_img is None or plate_img.size == 0:
            return preprocessed
        
        # Original image
        preprocessed.append(plate_img)
        
        # Convert to grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        preprocessed.append(gray)
        
        # Resize to larger resolution for better OCR
        h, w = plate_img.shape[:2]
        scale_factor = 3.0
        resized = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
        preprocessed.append(resized)
        
        # Apply bilateral filter to remove noise while preserving edges
        filtered = cv2.bilateralFilter(resized, 11, 17, 17)
        preprocessed.append(filtered)
        
        # Increase contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        preprocessed.append(enhanced)
        
        # Apply different thresholding methods
        # Normal binary threshold
        _, thresh_binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        preprocessed.append(thresh_binary)
        
        # Inverse binary threshold (for plates with light text on dark background)
        _, thresh_binary_inv = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY_INV)
        preprocessed.append(thresh_binary_inv)
        
        # Otsu thresholding (automatically finds optimal threshold)
        _, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(thresh_otsu)
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(adaptive)
        
        return preprocessed
    
    def _recognize_text(self, plate_img, debug=False, image_path=None):
        """
        Recognize text from a license plate image
        
        Args:
            plate_img (numpy.ndarray): License plate image
            debug (bool): Whether to save debug images
            image_path (str): Path to the original image (for debug naming)
            
        Returns:
            tuple: (recognized_text, confidence)
        """
        # Preprocess the plate image
        preprocessed_images = self._preprocess_for_ocr(plate_img)
        if not preprocessed_images:
            return "No valid plate image", 0.0
        
        # Save debug images
        if debug and image_path:
            debug_dir = 'debug_output'
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for i, img in enumerate(preprocessed_images):
                cv2.imwrite(f"{debug_dir}/{base_name}_preproc_{i}.png", img)
        
        # Try OCR with different configurations and preprocessing methods
        results = []
        
        for img in preprocessed_images:
            for config in self.ocr_configs:
                try:
                    # Use Tesseract for OCR
                    text = pytesseract.image_to_string(img, config=config).strip()
                    
                    # Apply post-processing
                    if text:
                        # Keep only alphanumeric characters, spaces, and hyphens
                        text = ''.join(c for c in text if c.isalnum() or c.isspace() or c == '-')
                        
                        # Remove excessive whitespace
                        text = ' '.join(text.split())
                        
                        if text:
                            # Calculate confidence based on text length (longer is usually better)
                            # For standard license plates, 5-8 characters is typical
                            text_len = len(text.replace(' ', ''))
                            if 4 <= text_len <= 10:
                                confidence = 0.8
                            elif text_len > 10:
                                confidence = 0.6  # Might be over-detecting
                            else:
                                confidence = 0.4  # Might be missing characters
                            
                            results.append((text, confidence))
                except Exception as e:
                    if debug:
                        print(f"OCR error: {e}")
                    continue
        
        # If we have no results, return empty string
        if not results:
            return "No text detected", 0.0
        
        # Sort results by confidence (high to low)
        results = sorted(results, key=lambda r: r[1], reverse=True)
        
        # Apply character corrections to the highest confidence result
        best_text, best_confidence = results[0]
        corrected_text = self._apply_character_corrections(best_text)
        
        return corrected_text, best_confidence
    
    def _apply_character_corrections(self, text):
        """
        Apply character corrections based on common OCR errors
        
        Args:
            text (str): Recognized text
            
        Returns:
            str: Corrected text
        """
        if not text:
            return text
        
        corrected = text
        
        # Apply character mappings
        for char, replacement in self.char_mappings.items():
            corrected = corrected.replace(char, replacement)
        
        return corrected

def main():
    """Main function to run license plate recognition"""
    parser = argparse.ArgumentParser(description='Improved License Plate Recognition')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--detector', type=str, help='Path to detector weights')
    parser.add_argument('--ocr-model', type=str, help='Path to OCR model directory')
    parser.add_argument('--debug', action='store_true', help='Save debug images')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        sys.exit(1)
    
    # Create license plate recognition system
    lpr = ImprovedLicensePlateRecognition(
        detector_weights=args.detector,
        ocr_model_dir=args.ocr_model
    )
    
    # Recognize license plate
    text, confidence, plate_img = lpr.recognize(args.image, debug=args.debug)
    
    print(f"\nRecognized Text: {text}")
    print(f"Confidence: {confidence:.2f}")
    
    # Save plate image
    if plate_img is not None:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        cv2.imwrite(f"{output_dir}/{base_name}_plate.png", plate_img)
        print(f"Plate image saved to: {output_dir}/{base_name}_plate.png")

if __name__ == "__main__":
    main() 