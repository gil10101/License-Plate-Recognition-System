#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pytesseract
import json
from pathlib import Path
import re
from difflib import SequenceMatcher
from improved_license_recognition import ImprovedLicensePlateRecognition

class EnhancedLicensePlateRecognition:
    """
    Enhanced license plate recognition with improved preprocessing and OCR
    """
    def __init__(self, detector_weights=None, ocr_model_dir=None):
        """
        Initialize the enhanced license plate recognition system.
        
        Args:
            detector_weights (str): Path to detector weights file
            ocr_model_dir (str): Path to OCR model directory
        """
        # Initialize base recognition system
        self.base_recognizer = ImprovedLicensePlateRecognition(
            detector_weights=detector_weights,
            ocr_model_dir=ocr_model_dir
        )
        
        # Load custom OCR parameters if available
        self.ocr_configs = []
        self.char_mappings = {}
        
        if ocr_model_dir:
            # Load optimized Tesseract parameters
            params_file = os.path.join(ocr_model_dir, "optimized_params.txt")
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("Best config:"):
                        config = first_line.split("Best config:")[1].split(",")[0].strip()
                        self.ocr_configs.append(config)
            
            # Load character corrections
            corrections_file = os.path.join(ocr_model_dir, "char_corrections.json")
            if os.path.exists(corrections_file):
                with open(corrections_file, 'r') as f:
                    self.char_mappings = json.load(f)
        
        # Add default configurations if none loaded
        if not self.ocr_configs:
            self.ocr_configs = [
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            ]
        
        # Default character mappings if none loaded
        if not self.char_mappings:
            self.char_mappings = {
                '0': 'O', 'O': '0', '1': 'I', 'I': '1', '5': 'S', 'S': '5',
                '8': 'B', 'B': '8', 'Z': '2', '2': 'Z'
            }
        
        # Load common license plate formats for post-processing
        self.plate_formats = [
            # 3 letters + 4 numbers (common US format)
            r'^([A-Z]{3})[\s\-]*(\d{4})$',
            # 2 letters + 5 numbers (AM 74043 format)
            r'^([A-Z]{2})[\s\-]*(\d{5})$',
            # 1-3 letters + 1-5 numbers
            r'^([A-Z]{1,3})[\s\-]*(\d{1,5})$',
            # 4-7 alphanumeric characters (no separator)
            r'^([A-Z0-9]{4,7})$'
        ]
    
    def recognize(self, image_path, debug=False):
        """
        Recognize license plate from an image with enhanced accuracy.
        
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
            return "Error reading image", 0.0, None
        
        # Create debug directory if needed
        if debug:
            debug_dir = 'debug_output'
            os.makedirs(debug_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(f"{debug_dir}/{base_name}_original.png", image)
        
        # Step 1: Detect license plate with improved detection
        plate_img = self._detect_plate(image, debug, image_path)
        if plate_img is None:
            return "No plate detected", 0.0, None
        
        # Step 2: Apply multiple preprocessing methods
        preprocessed_images = self._preprocess_plate(plate_img, debug, image_path)
        
        # Step 3: Apply OCR with multiple configurations and methods
        text_results = self._apply_ocr(preprocessed_images, debug, image_path)
        
        # Step 4: Select and post-process the best result
        text, confidence = self._select_best_result(text_results, debug)
        
        # Step 5: Apply character corrections
        corrected_text = self._apply_corrections(text)
        
        # Step 6: Apply pattern-based confidence adjustment
        # Better confidence calculation based on similarity to license plate patterns
        adjusted_confidence = confidence
        for pattern in self.plate_formats:
            if re.match(pattern, corrected_text.upper()):
                # This matches a known license plate pattern - increase confidence
                adjusted_confidence = min(1.0, confidence + 0.2)
                break
        
        # Decrease confidence for very short texts which are unlikely to be valid plates
        if len(corrected_text.replace(' ', '')) < 4:
            adjusted_confidence *= 0.7
        
        # Return final result
        return corrected_text, adjusted_confidence, plate_img
    
    def _detect_plate(self, image, debug=False, image_path=None):
        """
        Detect license plate using an improved approach.
        
        Args:
            image (numpy.ndarray): Input image
            debug (bool): Whether to save debug images
            image_path (str): Path to the original image (for debug naming)
            
        Returns:
            numpy.ndarray: License plate image
        """
        # For small test images, assume the entire image is a license plate
        h, w = image.shape[:2]
        if h < 100 or w < 200:
            # This is likely already a cropped license plate image
            if debug and image_path:
                debug_dir = 'debug_output'
                os.makedirs(debug_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                cv2.imwrite(f"{debug_dir}/{base_name}_direct_plate.png", image)
            return image
            
        # Use the base detector first
        plate_img = self.base_recognizer._detect_plate(image, debug, image_path)
        
        # If detection failed, try alternative methods
        if plate_img is None:
            # Apply alternative detection methods
            plate_img = self._detect_plate_alternative(image, debug, image_path)
        
        # If we still don't have a plate, try with the entire image
        if plate_img is None:
            if debug and image_path:
                debug_dir = 'debug_output'
                os.makedirs(debug_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                cv2.imwrite(f"{debug_dir}/{base_name}_fullimage.png", image)
            return image
        
        return plate_img
    
    def _detect_plate_alternative(self, image, debug=False, image_path=None):
        """
        Alternative license plate detection methods for difficult cases.
        
        Args:
            image (numpy.ndarray): Input image
            debug (bool): Whether to save debug images
            image_path (str): Path to the original image
            
        Returns:
            numpy.ndarray: License plate image
        """
        # Create debug directory if needed
        debug_dir = 'debug_output' if debug else None
        base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else None
        
        # Method 1: Edge-based detection with morphological operations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges
        edges = cv2.Canny(filtered, 30, 200)
        
        # Apply morphological operations to close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug image
        if debug and image_path:
            contour_img = image.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(f"{debug_dir}/{base_name}_alt_contours.png", contour_img)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Find rectangles among contours
        candidates = []
        for contour in contours:
            # Get rotated rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get width and height
            width = rect[1][0]
            height = rect[1][1]
            
            # Ensure width is larger than height
            if width < height:
                width, height = height, width
            
            # Check aspect ratio
            if width > 0 and height > 0:
                aspect_ratio = width / height
                
                # License plates typically have aspect ratios between 2:1 and 5:1
                if 1.5 <= aspect_ratio <= 5:
                    # Get upright bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check size
                    if w > 60 and h > 15:
                        area_ratio = w * h / (image.shape[0] * image.shape[1])
                        
                        # License plates typically occupy a reasonable portion of the image
                        if 0.01 <= area_ratio <= 0.3:
                            score = aspect_ratio / 5  # Normalize to [0,1]
                            
                            # Add padding
                            padding_x = int(w * 0.1)
                            padding_y = int(h * 0.1)
                            
                            x1 = max(0, x - padding_x)
                            y1 = max(0, y - padding_y)
                            x2 = min(image.shape[1], x + w + padding_x)
                            y2 = min(image.shape[0], y + h + padding_y)
                            
                            candidates.append({
                                'score': score,
                                'box': (x1, y1, x2, y2)
                            })
        
        # Return best candidate if found
        if candidates:
            best_candidate = max(candidates, key=lambda c: c['score'])
            x1, y1, x2, y2 = best_candidate['box']
            
            # Extract plate region
            plate_img = image[y1:y2, x1:x2].copy()
            
            # Save debug image
            if debug and image_path:
                cv2.imwrite(f"{debug_dir}/{base_name}_alt_plate.png", plate_img)
            
            return plate_img
        
        # If all else fails, use the center region of the image
        h, w = image.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        plate_width = w // 2
        plate_height = h // 4
        
        x1 = center_x - plate_width // 2
        y1 = center_y - plate_height // 2
        x2 = x1 + plate_width
        y2 = y1 + plate_height
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract center region
        plate_img = image[y1:y2, x1:x2].copy()
        
        # Save debug image
        if debug and image_path:
            cv2.imwrite(f"{debug_dir}/{base_name}_alt_center.png", plate_img)
        
        return plate_img
    
    def _preprocess_plate(self, plate_img, debug=False, image_path=None):
        """
        Apply multiple preprocessing methods to enhance the license plate image.
        
        Args:
            plate_img (numpy.ndarray): License plate image
            debug (bool): Whether to save debug images
            image_path (str): Path to the original image
            
        Returns:
            list: List of preprocessed images
        """
        # Create debug directory if needed
        debug_dir = 'debug_output' if debug else None
        base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else None
        
        # Convert to grayscale if needed
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # Resize if too small
        if gray.shape[0] < 30 or gray.shape[1] < 80:
            scale_factor = max(30/gray.shape[0], 80/gray.shape[1])
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # List of preprocessed images
        preprocessed = []
        
        # Method 1: Basic grayscale
        preprocessed.append(gray)
        
        # Method 2: Resize to standardize height for OCR
        standard_height = 50
        aspect_ratio = gray.shape[1] / gray.shape[0]
        standard_width = int(standard_height * aspect_ratio)
        resized = cv2.resize(gray, (standard_width, standard_height), interpolation=cv2.INTER_CUBIC)
        preprocessed.append(resized)
        
        # Method 3: Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        preprocessed.append(blurred)
        
        # Method 4: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed.append(adaptive_thresh)
        
        # Method 5: Otsu's thresholding
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(otsu_thresh)
        
        # Method 6: Local histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        preprocessed.append(equalized)
        
        # Method 7: Thresholding on equalized image
        _, eq_thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(eq_thresh)
        
        # Method 8: Dilated image for connected components
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(otsu_thresh, kernel, iterations=1)
        preprocessed.append(dilated)
        
        # Method 9: Eroded image for text enhancement
        eroded = cv2.erode(otsu_thresh, kernel, iterations=1)
        preprocessed.append(eroded)
        
        # Method 10: Increased contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        preprocessed.append(enhanced)
        
        # Method 11: Sharpened image
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        preprocessed.append(sharpened)
        
        # Method 12: Binary inverted
        _, binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed.append(binary_inv)
        
        # Method 13: Morphological gradient
        kernel = np.ones((2, 2), np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        preprocessed.append(gradient)
        
        # Method 14: Deep copy with specific targeted cleanup
        clean_copy = gray.copy()
        # Apply bilateral filter to reduce noise while preserving edges
        clean_copy = cv2.bilateralFilter(clean_copy, 11, 17, 17)
        # Apply contrast stretching
        clean_copy = cv2.normalize(clean_copy, None, 0, 255, cv2.NORM_MINMAX)
        preprocessed.append(clean_copy)
        
        # Method 15: Character segmentation preparation
        char_seg = gray.copy()
        # Apply light blur to reduce noise
        char_seg = cv2.GaussianBlur(char_seg, (3, 3), 0)
        # Apply adaptive thresholding
        char_seg = cv2.adaptiveThreshold(
            char_seg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
        )
        # Apply morphological operations to connect components
        kernel = np.ones((1, 3), np.uint8)  # Horizontal kernel
        char_seg = cv2.morphologyEx(char_seg, cv2.MORPH_CLOSE, kernel, iterations=1)
        preprocessed.append(char_seg)
        
        # Save debug images
        if debug and image_path:
            for i, img in enumerate(preprocessed):
                cv2.imwrite(f"{debug_dir}/{base_name}_preproc_{i}.png", img)
        
        return preprocessed
    
    def _apply_ocr(self, preprocessed_images, debug=False, image_path=None):
        """
        Apply OCR with multiple configurations to the preprocessed images.
        
        Args:
            preprocessed_images (list): List of preprocessed images
            debug (bool): Whether to save debug images
            image_path (str): Path to the original image
            
        Returns:
            list: List of (text, confidence, method) tuples
        """
        results = []
        
        # Try multiple configurations on each preprocessed image
        for img_idx, img in enumerate(preprocessed_images):
            for config_idx, config in enumerate(self.ocr_configs):
                try:
                    # Direct string recognition
                    text = pytesseract.image_to_string(img, config=config).strip()
                    
                    # Clean up text
                    clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c == '-')
                    clean_text = ' '.join(clean_text.split())
                    
                    if clean_text:
                        # Add data-based confidence if available
                        try:
                            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                            
                            # Calculate average confidence
                            confidences = [float(conf) for conf in data['conf'] if conf != '-1']
                            if confidences:
                                avg_conf = sum(confidences) / len(confidences) / 100.0  # Convert to 0-1 scale
                            else:
                                avg_conf = 0.5  # Default confidence
                            
                            # Use char count for initial confidence adjustment
                            char_count = len(clean_text.replace(' ', ''))
                            if 4 <= char_count <= 8:
                                char_conf = 0.8
                            else:
                                char_conf = 0.5
                            
                            # Weigh the confidence
                            confidence = (avg_conf * 0.7) + (char_conf * 0.3)
                            
                            # Add to results
                            method = f"preproc_{img_idx}_config_{config_idx}"
                            results.append((clean_text, confidence, method))
                        except:
                            # Fallback confidence calculation
                            char_count = len(clean_text.replace(' ', ''))
                            if 4 <= char_count <= 8:
                                confidence = 0.7
                            else:
                                confidence = 0.4
                            
                            method = f"direct_preproc_{img_idx}_config_{config_idx}"
                            results.append((clean_text, confidence, method))
                    
                except Exception as e:
                    if debug:
                        print(f"OCR error: {e}")
                    continue
        
        return results
    
    def _select_best_result(self, text_results, debug=False):
        """
        Select the best OCR result from multiple candidates.
        
        Args:
            text_results (list): List of (text, confidence, method) tuples
            debug (bool): Whether to enable debug output
            
        Returns:
            tuple: (best_text, confidence)
        """
        if not text_results:
            return "No text detected", 0.0
        
        # Remove duplicates and keep highest confidence
        unique_results = {}
        for text, conf, method in text_results:
            # Normalize text for comparison
            norm_text = text.upper().replace(' ', '')
            
            if norm_text not in unique_results or conf > unique_results[norm_text][1]:
                unique_results[norm_text] = (text, conf, method)
        
        # Convert back to list
        results = list(unique_results.values())
        
        # Filter results based on common license plate patterns
        pattern_matches = []
        for text, conf, method in results:
            # Check if the text matches any license plate format
            for pattern in self.plate_formats:
                if re.match(pattern, text.upper()):
                    # This is likely a license plate format
                    pattern_matches.append((text, conf + 0.1, method))  # Boost confidence
                    break
        
        # If we have pattern matches, use those
        if pattern_matches:
            filtered_results = pattern_matches
        else:
            filtered_results = results
        
        # Sort by confidence (high to low)
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        # Get the best result
        if filtered_results:
            best_text, best_conf, best_method = filtered_results[0]
            
            if debug:
                print(f"Best OCR result: '{best_text}' (conf={best_conf:.2f}, method={best_method})")
                
                # Print top 3 results
                print("Top results:")
                for i, (text, conf, method) in enumerate(filtered_results[:3], 1):
                    print(f"{i}. '{text}' (conf={conf:.2f}, method={method})")
            
            return best_text, best_conf
        else:
            return "No text detected", 0.0
    
    def _apply_corrections(self, text):
        """
        Apply character corrections to improve recognition.
        
        Args:
            text (str): Recognized text
            
        Returns:
            str: Corrected text
        """
        # Skip empty text
        if not text or len(text) < 2:
            return text
        
        # Remove common non-alphanumeric characters that might be mistakes
        text = re.sub(r'[^A-Z0-9\s\-]', '', text.upper())
        
        # Apply character-by-character corrections
        corrected_chars = []
        for char in text:
            if char.upper() in self.char_mappings:
                corrected_chars.append(self.char_mappings[char.upper()])
            else:
                corrected_chars.append(char)
        
        corrected_text = ''.join(corrected_chars)
        
        # Common patterns for license plates
        patterns_and_formats = [
            # 3 letters + 4 numbers (common US format)
            (r'^([A-Z]{3})[\s\-]*(\d{4})$', r'\1 \2'),
            # 2 letters + 5 numbers (like AM 74043)
            (r'^([A-Z]{2})[\s\-]*(\d{5})$', r'\1 \2'),
            # 1 letter + 1-6 digits (some state formats)
            (r'^([A-Z])[\s\-]*(\d{1,6})$', r'\1 \2'),
            # 3 letters + 3 numbers (some state formats)
            (r'^([A-Z]{3})[\s\-]*(\d{3})$', r'\1 \2'),
            # 2-3 digits + 3 letters (some state formats)
            (r'^(\d{2,3})[\s\-]*([A-Z]{3})$', r'\1 \2'),
            # 1-3 letters + 1-5 numbers (generic format)
            (r'^([A-Z]{1,3})[\s\-]*(\d{1,5})$', r'\1 \2'),
        ]
        
        # Try to match and format according to license plate patterns
        for pattern, format_str in patterns_and_formats:
            match = re.match(pattern, corrected_text)
            if match:
                return re.sub(pattern, format_str, corrected_text)
        
        # Special corrections for common OCR errors in plates
        # Correct 'B' at the beginning that should be '8'
        if corrected_text.startswith('B') and len(corrected_text) > 1 and corrected_text[1].isdigit():
            corrected_text = '8' + corrected_text[1:]
            
        # Correct 'O' that should be '0' when between digits
        for i in range(1, len(corrected_text) - 1):
            if corrected_text[i] == 'O' and corrected_text[i-1].isdigit() and corrected_text[i+1].isdigit():
                corrected_text = corrected_text[:i] + '0' + corrected_text[i+1:]
        
        # Correct 'I' that should be '1' when between digits
        for i in range(1, len(corrected_text) - 1):
            if corrected_text[i] == 'I' and corrected_text[i-1].isdigit() and corrected_text[i+1].isdigit():
                corrected_text = corrected_text[:i] + '1' + corrected_text[i+1:]
        
        # Fix common spacing issues - add space between letter block and number block
        letter_number_pattern = r'([A-Z]{2,3})(\d{2,5})'
        if re.search(letter_number_pattern, corrected_text):
            corrected_text = re.sub(letter_number_pattern, r'\1 \2', corrected_text)
        
        # Return the corrected text
        return corrected_text

def main():
    """Main function to test the enhanced license plate recognition"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Enhanced License Plate Recognition')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--detector', type=str, help='Path to detector weights')
    parser.add_argument('--ocr-model', type=str, help='Path to OCR model directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Create recognition system
    lpr = EnhancedLicensePlateRecognition(
        detector_weights=args.detector,
        ocr_model_dir=args.ocr_model
    )
    
    # Recognize license plate
    text, confidence, plate_img = lpr.recognize(args.image, debug=args.debug)
    
    # Print results
    print(f"Recognized text: {text}")
    print(f"Confidence: {confidence:.2f}")
    
    # Save detected plate if available
    if plate_img is not None:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(output_dir, f"{base_name}_plate.png")
        cv2.imwrite(output_path, plate_img)
        print(f"Plate image saved to {output_path}")

if __name__ == "__main__":
    main() 