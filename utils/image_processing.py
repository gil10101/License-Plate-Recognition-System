import cv2
import numpy as np
import pytesseract
import os
import re

# Set Tesseract executable path on Windows (change as needed)
if os.name == 'nt':  # Check if running on Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_plate(plate_img):
    """
    Preprocess the license plate image for better OCR results.
    
    Args:
        plate_img (numpy.ndarray): Cropped license plate image.
        
    Returns:
        numpy.ndarray: Preprocessed image ready for OCR.
    """
    # Check if the image exists and is valid
    if plate_img is None or plate_img.size == 0 or plate_img.shape[0] < 5 or plate_img.shape[1] < 5:
        # Return an empty list if the image is invalid
        return []
    
    # Save original image for direct OCR (sometimes works better without preprocessing)
    original_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    
    # Get original image dimensions for later checks
    original_height, original_width = plate_img.shape[:2]
    
    # Check if the plate might have the AM 74043 format (US plate with dark text on light background)
    is_us_style_plate = check_for_us_style_plate(plate_img)
    
    # Check if this is likely a direct license plate image (larger and clearer)
    # or a small region from a traffic scene
    is_direct_plate = original_width > 300 and original_height > 100
    is_small_plate = original_width < 200 or original_height < 50
    
    # List to store all processed versions of the image
    preprocessed_images = []
    
    # Add original image first - sometimes raw image works best
    preprocessed_images.append(original_rgb)
    
    # SUPER RESOLUTION FOR SMALL PLATES
    # If the plate is very small, use a larger upscaling
    if is_small_plate:
        # Super-resolution upscaling for small plates (4x)
        scale_factor = 4
        resized_plate = cv2.resize(plate_img, (original_width*scale_factor, original_height*scale_factor), 
                                   interpolation=cv2.INTER_CUBIC)
    else:
        # Regular upscaling for normal plates (2x)
        scale_factor = 2
        resized_plate = cv2.resize(plate_img, (original_width*scale_factor, original_height*scale_factor), 
                                   interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale - original and resized
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray_large = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)
    
    # Add a new preprocessing step: bilateral filtering to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    bilateral_large = cv2.bilateralFilter(gray_large, 11, 17, 17)
    preprocessed_images.append(bilateral)
    preprocessed_images.append(bilateral_large)
    
    # New preprocessing: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    clahe_large = clahe.apply(gray_large)
    preprocessed_images.append(clahe_img)
    preprocessed_images.append(clahe_large)
    
    # Special processing for US-style plates (like AM 74043)
    if is_us_style_plate:
        # Apply specialized preprocessing for US-style plates
        us_processed_images = preprocess_us_style_plate(plate_img, gray, gray_large)
        preprocessed_images.extend(us_processed_images)
    
    # Try to crop out just the characters for traffic scenes
    if not is_direct_plate:
        # Try to find character regions in the plate
        char_regions = find_character_regions(gray)
        
        if char_regions and len(char_regions) > 0:
            # If we found character regions, extract and add them
            for region in char_regions:
                x, y, w, h = region
                if w > 5 and h > 5:  # Minimum size check
                    char_img = gray[y:y+h, x:x+w]
                    # Resize character for better OCR
                    char_img_large = cv2.resize(char_img, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
                    _, char_thresh = cv2.threshold(char_img_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    preprocessed_images.append(char_thresh)
    
    # SPECIALIZED PROCESSING FOR SMALL PLATES FROM TRAFFIC SCENES
    if is_small_plate:
        # Apply specialized preprocessing for small plates
        # 1. Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray_large, None, 10, 7, 21)
        
        # 2. Strong contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Adaptive thresholding with smaller block sizes
        adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(adaptive_thresh)
        
        # 4. Otsu's thresholding with contrast stretching
        _, contrast_img = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(contrast_img)
        
        # 5. Edge enhancement for small text
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(sharp_thresh)
        
        # 6. Original color image upscaled for color-aware OCR
        color_large = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2RGB)
        preprocessed_images.append(color_large)
        
        # New: Try multiple thresholding values for very small plates
        for thresh_val in [90, 120, 150, 180]:
            _, bin_img = cv2.threshold(denoised, thresh_val, 255, cv2.THRESH_BINARY)
            preprocessed_images.append(bin_img)
    
    # Add original color images (sometimes works better for colored plates)
    plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    preprocessed_images.append(plate_img_rgb)
    
    resized_rgb = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2RGB)
    preprocessed_images.append(resized_rgb)
    
    # Add original and resized grayscale
    preprocessed_images.append(gray)
    preprocessed_images.append(gray_large)
    
    # New: Try different thresholding values
    for thresh_val in [100, 120, 140, 160, 180]:
        _, fixed_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        preprocessed_images.append(fixed_thresh)
        
        _, fixed_thresh_large = cv2.threshold(gray_large, thresh_val, 255, cv2.THRESH_BINARY)
        preprocessed_images.append(fixed_thresh_large)
    
    # Basic thresholding pipeline
    # Simple binary thresholding
    _, thresh_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    preprocessed_images.append(thresh_simple)
    
    _, thresh_simple_large = cv2.threshold(gray_large, 127, 255, cv2.THRESH_BINARY)
    preprocessed_images.append(thresh_simple_large)
    
    # Otsu thresholding
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(thresh_otsu)
    
    _, thresh_otsu_large = cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(thresh_otsu_large)
    
    # Inverted images (sometimes OCR works better on white-on-black)
    thresh_otsu_inv = cv2.bitwise_not(thresh_otsu)
    preprocessed_images.append(thresh_otsu_inv)
    
    thresh_otsu_inv_large = cv2.bitwise_not(thresh_otsu_large)
    preprocessed_images.append(thresh_otsu_inv_large)
    
    # New: Try adaptive thresholding with different block sizes
    blockSizes = [7, 11, 15, 19]
    for blockSize in blockSizes:
        adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY, blockSize, 2)
        adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, blockSize, 2)
        
        preprocessed_images.append(adaptive_mean)
        preprocessed_images.append(adaptive_gaussian)
        
        # Add inverted versions too
        preprocessed_images.append(cv2.bitwise_not(adaptive_mean))
        preprocessed_images.append(cv2.bitwise_not(adaptive_gaussian))
    
    # SPECIALIZED PROCESSING FOR DIRECT LICENSE PLATES
    if is_direct_plate:
        # Increase contrast for direct plates
        alpha = 2.0  # higher contrast for large text
        beta = 0     # brightness control
        contrast_img = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        _, thresh_contrast = cv2.threshold(contrast_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(thresh_contrast)
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 5)
        preprocessed_images.append(adaptive)
        
        # Edge enhancement
        sharpening_kernel = np.array([[-1,-1,-1], 
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, sharpening_kernel)
        _, thresh_sharp = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(thresh_sharp)
        
        # New: Try edge-based approaches for high-contrast plates
        edges = cv2.Canny(gray, 100, 200)
        dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        preprocessed_images.append(dilated_edges)
    
    return preprocessed_images

def check_for_us_style_plate(plate_img):
    """
    Check if the image appears to be a US-style license plate (like AM 74043).
    
    Args:
        plate_img (numpy.ndarray): License plate image.
        
    Returns:
        bool: True if it appears to be a US-style plate.
    """
    # Common features of US license plates:
    # 1. Rectangular with specific aspect ratio
    # 2. Usually light background with dark text
    # 3. Often contrasting colors
    
    h, w = plate_img.shape[:2]
    
    # Check aspect ratio (US plates are typically around 2:1)
    aspect_ratio = w / h
    if not (1.8 <= aspect_ratio <= 2.5):
        return False
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    
    # Look for light background (white or light colors)
    light_mask = cv2.inRange(hsv, (0, 0, 150), (180, 50, 255))
    light_ratio = np.sum(light_mask > 0) / (h * w)
    
    # US plates typically have light backgrounds
    if light_ratio < 0.5:  # If less than 50% is light colored
        return False
        
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate text from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Check for presence of dark elements (text) on light background
    dark_ratio = np.sum(binary > 0) / (h * w)
    
    # Text should be between 15-40% of the plate
    if not (0.15 <= dark_ratio <= 0.4):
        return False
    
    return True

def preprocess_us_style_plate(plate_img, gray, gray_large):
    """
    Special preprocessing for US-style license plates (like AM 74043).
    
    Args:
        plate_img (numpy.ndarray): Original license plate image.
        gray (numpy.ndarray): Grayscale version of plate image.
        gray_large (numpy.ndarray): Larger grayscale version.
        
    Returns:
        list: List of preprocessed images optimized for US plates.
    """
    us_processed = []
    h, w = plate_img.shape[:2]
    
    # 1. Basic binary threshold (good for high contrast plates)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    us_processed.append(binary)
    
    # 2. Adaptive thresholding (handles uneven lighting better)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 17, 11)
    us_processed.append(adaptive)
    
    # 3. Strong contrast enhancement then threshold
    # Stretch histogram for better separation of text and background
    equ = cv2.equalizeHist(gray)
    _, contrast_binary = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    us_processed.append(contrast_binary)
    
    # 4. Apply morphological operations to enhance text
    # Close small gaps in text
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(contrast_binary, cv2.MORPH_CLOSE, kernel)
    us_processed.append(closed)
    
    # 5. For larger images, try different block sizes for adaptive thresholding
    block_sizes = [11, 15, 21]
    for block_size in block_sizes:
        adaptive_large = cv2.adaptiveThreshold(gray_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, block_size, 5)
        us_processed.append(adaptive_large)
    
    # 6. Noise reduction followed by edge enhancement (good for unclear plates)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, enhanced_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    us_processed.append(enhanced_binary)
    
    # 7. Try to separate characters by finding contours
    # Create a mask from thresholded image
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilate to connect nearby components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.copy(), kernel, iterations=1)
    
    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a character-only image
    char_img = np.ones_like(gray) * 255  # White background
    
    # Draw only character-like contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h > 0 else 0
        area = w * h
        
        # Filter for character-like shapes
        if (0.2 < aspect < 1.2 and  # Character aspect ratio
            area > 100 and          # Minimum size
            area < (gray.size / 10)):  # Not too large
            # Extract character from original image
            char_roi = mask[y:y+h, x:x+w]
            # Place character on white background
            char_img[y:y+h, x:x+w] = cv2.bitwise_not(char_roi)
    
    us_processed.append(char_img)
    
    return us_processed

def find_character_regions(gray_img):
    """
    Find potential character regions in a license plate image.
    
    Args:
        gray_img (numpy.ndarray): Grayscale license plate image.
        
    Returns:
        list: List of character bounding boxes as (x, y, w, h).
    """
    # Threshold the image
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to enhance character separation
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for potential character contours
    char_regions = []
    img_height, img_width = gray_img.shape
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Character aspect ratio is typically between 0.2 and 1.0 (height > width)
        aspect_ratio = w / h if h > 0 else 0
        
        # Character size relative to plate
        area = w * h
        relative_height = h / img_height
        
        # Check if the contour might be a character
        if (0.1 < aspect_ratio < 2.0 and
            area > 20 and  # Minimum character size
            area < (img_width * img_height / 4) and  # Not too big
            relative_height > 0.3):  # Height should be significant portion of plate
            
            # Add some padding
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.2)
            
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            w1 = min(img_width - x1, w + 2*padding_x)
            h1 = min(img_height - y1, h + 2*padding_y)
            
            char_regions.append((x1, y1, w1, h1))
    
    return char_regions

def recognize_text(plate_img):
    """
    Recognize text from a preprocessed license plate image.
    
    Args:
        plate_img (numpy.ndarray or list): Preprocessed license plate image(s).
        
    Returns:
        str: Recognized license plate text.
    """
    # Check if plate_img is empty
    if not plate_img:
        return ""
    
    # License plate configurations to try
    configs = [
        # Optimized for license plates - single line with whitelist
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_pageseg_mode=7',
        
        # Optimized for US-style plates (like AM 74043) - single line with character spacing
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_pageseg_mode=7 -c textord_space_size_is_variable=1',
        
        # For small license plates in traffic scenes - single word
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        
        # For cropped characters - single character mode
        r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        
        # Sparse text with OSD - might work better for some plates
        r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        
        # Word list to try to recognize common plate formats
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c textord_min_linesize=2.5',
        
        # Assume a single uniform block of vertically aligned text
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    ]
    
    try:
        # If we have multiple preprocessed images, try each one
        if isinstance(plate_img, list):
            all_results = []
            char_results = ""  # For collecting individual character results
            
            for img in plate_img:
                # Skip empty images
                if img is None or img.size == 0:
                    continue
                    
                # Check if this is a small image (likely a single character)
                is_single_char = False
                if len(img.shape) == 2:  # Grayscale
                    h, w = img.shape
                    is_single_char = (h > 0 and w > 0 and h < 100 and w < 100 and h/w < 3 and w/h < 3)
                elif len(img.shape) == 3:  # Color
                    h, w, _ = img.shape
                    is_single_char = (h > 0 and w > 0 and h < 100 and w < 100 and h/w < 3 and w/h < 3)
                
                for config in configs:
                    # Use PSM 10 (single character) for very small regions
                    if is_single_char and 'psm 10' not in config:
                        continue
                        
                    # Skip PSM 10 for larger images
                    if not is_single_char and 'psm 10' in config:
                        continue
                        
                    # Get direct string first (sometimes more reliable for large text)
                    try:
                        direct_text = pytesseract.image_to_string(img, config=config).strip()
                        if direct_text:
                            # Basic cleanup
                            clean_text = ''.join(c for c in direct_text if c.isalnum() or c.isspace())
                            if clean_text:
                                # For single characters, add to char collection
                                if is_single_char and len(clean_text) == 1:
                                    char_results += clean_text
                                else:
                                    # Boost confidence for results with alphanumerics only
                                    confidence_boost = 0
                                    if all(c.isalnum() or c.isspace() for c in clean_text):
                                        confidence_boost = 10
                                    
                                    all_results.append({
                                        'text': clean_text, 
                                        'confidence': 75.0 + confidence_boost,  # Assign reasonable confidence
                                        'method': 'direct',
                                        'raw': direct_text  # Save raw text for debugging
                                    })
                    except Exception:
                        # Continue if this specific OCR operation fails
                        pass
                    
                    # Get data with confidence info
                    try:
                        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                        
                        # Extract text and calculate average confidence for this image
                        text_parts = []
                        total_conf = 0
                        count = 0
                        
                        for i, conf in enumerate(data['conf']):
                            try:
                                conf_val = float(conf)
                                if conf != '-1' and conf_val > 30:  # Only consider recognized parts with confidence > 30%
                                    text_parts.append(data['text'][i])
                                    total_conf += conf_val
                                    count += 1
                            except ValueError:
                                # Skip entries with invalid confidence values
                                continue
                        
                        if count > 0:
                            avg_conf = total_conf / count
                            text = " ".join(text_parts).strip()
                            
                            if text:
                                # For single character images, add to char collection
                                if is_single_char and len(text) == 1:
                                    char_results += text
                                else:
                                    all_results.append({
                                        'text': text, 
                                        'confidence': avg_conf,
                                        'method': 'data'
                                    })
                    except Exception:
                        # Continue if this specific OCR operation fails
                        pass
            
            # If we've accumulated characters, add as a result
            if char_results:
                all_results.append({
                    'text': char_results,
                    'confidence': 85.0,  # High confidence for character-by-character recognition
                    'method': 'char'
                })
            
            # If we have results, sort by confidence and take the best ones
            if all_results:
                # Filter out common state names when they appear alone
                filtered_results = []
                state_names = ["TEXAS", "CALIFORNIA", "FLORIDA", "NEW YORK", "OHIO", "WASHINGTON", 
                              "OREGON", "MICHIGAN", "ALASKA", "HAWAII", "UTAH", "NEVADA", "CHICAGO", 
                              "ILLINOIS", "MASSACHUSETTS", "MARYLAND", "COLORADO", "DELAWARE", "IDAHO"]
                
                for result in all_results:
                    text = result['text'].strip().upper()
                    
                    # Skip entries that are just state names
                    if text in state_names:
                        continue
                    
                    # Keep entries with meaningful content
                    if len(text.strip()) > 0:
                        # Prioritize results that have a good mix of letters and numbers (typical for plates)
                        has_letters = any(c.isalpha() for c in text)
                        has_numbers = any(c.isdigit() for c in text)
                        
                        if has_letters and has_numbers:
                            result['confidence'] += 10.0  # Boost confidence for mixed alphanumeric
                        
                        # Boost results with common license plate lengths
                        if 5 <= len(text.replace(" ", "")) <= 8:
                            result['confidence'] += 5.0
                        
                        # Penalize too short texts
                        if len(text.replace(" ", "")) < 3:
                            result['confidence'] -= 20.0
                        
                        # Boost confidence for texts with common license plate patterns
                        # 3 letters + 3/4 numbers or 2 letters + 4/5 numbers
                        if re.match(r'^[A-Z]{2,3}[ -]?\d{3,5}$', text):
                            result['confidence'] += 15.0
                            
                        filtered_results.append(result)
                
                # If we filtered everything out, go with the original results
                if not filtered_results and all_results:
                    filtered_results = all_results
                
                # Sort by confidence
                filtered_results.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Take the best candidates and try to apply pattern matching
                if filtered_results:
                    # Get the highest confidence result
                    best_result = filtered_results[0]
                    best_text = best_result['text'].upper()
                    
                    # Apply post-processing to improve the result
                    processed_text = post_process_license_plate(best_text)
                    if processed_text:
                        return processed_text
                    
                    # If no specific pattern was found, return the best result
                    return best_text
            
            return ""
        else:
            # Original single image processing
            text = pytesseract.image_to_string(plate_img, config=configs[0]).strip()
            # Post-process the text
            processed_text = post_process_license_plate(text)
            if processed_text:
                return processed_text
            
            # If post-processing failed, just return alphanumeric characters
            return ''.join(c for c in text if c.isalnum())
            
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def post_process_license_plate(text):
    """
    Apply post-processing to improve license plate text.
    
    Args:
        text (str): Raw recognized text.
        
    Returns:
        str: Post-processed text.
    """
    if not text:
        return ""
    
    # Remove any non-alphanumeric and space characters
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    
    # Common character substitutions for OCR errors
    replacements = {
        '8': 'B',  # 8 is often confused with B
        '0': 'O',  # 0 is often confused with O
        '1': 'I',  # 1 is often confused with I
        '5': 'S',  # 5 is often confused with S
        '2': 'Z',  # 2 is often confused with Z
    }
    
    # Only apply replacements based on context - plates typically have letters first, then numbers
    # So don't replace numbers in numeric positions and don't replace letters in letter positions
    
    # Try to identify plate format first
    # Common formats: 3 letters + 4 numbers, 2 letters + 5 numbers, etc.
    
    # Format: 3 letters + 4 numbers (e.g., ABC1234)
    pattern_3l4n = re.match(r'^([A-Z0-9]{3})[ -]?([A-Z0-9]{4})$', text, re.IGNORECASE)
    if pattern_3l4n:
        # Extract the letter part and number part
        letter_part = pattern_3l4n.group(1)
        number_part = pattern_3l4n.group(2)
        
        # Fix letter part - replace numbers with letters where appropriate
        letter_part_fixed = ""
        for c in letter_part:
            if c.isdigit() and c in replacements:
                letter_part_fixed += replacements[c]
            else:
                letter_part_fixed += c
        
        # Fix number part - replace letters with numbers where appropriate
        # For example, replace 'O' with '0', 'I' with '1'
        number_part_fixed = ""
        for c in number_part:
            if c.isalpha():
                # Map letters back to numbers 
                if c == 'O' or c == 'o': 
                    number_part_fixed += '0'
                elif c == 'I' or c == 'i' or c == 'l' or c == 'L':
                    number_part_fixed += '1'
                elif c == 'Z' or c == 'z':
                    number_part_fixed += '2'
                elif c == 'S' or c == 's':
                    number_part_fixed += '5'
                elif c == 'B' or c == 'b':
                    number_part_fixed += '8'
                else:
                    number_part_fixed += c
            else:
                number_part_fixed += c
                
        return f"{letter_part_fixed} {number_part_fixed}"
    
    # Format: 2 letters + 5 numbers (e.g., AM 74043)
    pattern_2l5n = re.match(r'^([A-Z0-9]{2})[ -]?([A-Z0-9]{5})$', text, re.IGNORECASE)
    if pattern_2l5n:
        letter_part = pattern_2l5n.group(1)
        number_part = pattern_2l5n.group(2)
        
        # Fix letter part
        letter_part_fixed = ""
        for c in letter_part:
            if c.isdigit() and c in replacements:
                letter_part_fixed += replacements[c]
            else:
                letter_part_fixed += c
        
        # Fix number part
        number_part_fixed = ""
        for c in number_part:
            if c.isalpha():
                if c == 'O' or c == 'o': 
                    number_part_fixed += '0'
                elif c == 'I' or c == 'i' or c == 'l' or c == 'L':
                    number_part_fixed += '1'
                elif c == 'Z' or c == 'z':
                    number_part_fixed += '2'
                elif c == 'S' or c == 's':
                    number_part_fixed += '5'
                elif c == 'B' or c == 'b':
                    number_part_fixed += '8'
                else:
                    number_part_fixed += c
            else:
                number_part_fixed += c
                
        return f"{letter_part_fixed} {number_part_fixed}"
    
    # Format: 3 digits + 3 letters + 1 digit (e.g., 123ABC4)
    pattern_3d3l1d = re.match(r'^([A-Z0-9]{3})[ -]?([A-Z0-9]{3})[ -]?([A-Z0-9]{1})$', text, re.IGNORECASE)
    if pattern_3d3l1d:
        # This is a less common pattern, but handle it
        first_part = pattern_3d3l1d.group(1)
        middle_part = pattern_3d3l1d.group(2)
        last_part = pattern_3d3l1d.group(3)
        
        # Process with best guess based on position
        first_part_fixed = ''.join([replacements.get(c, c) if c.isdigit() else c for c in first_part])
        middle_part_fixed = middle_part  # Keep as is, could be either
        last_part_fixed = last_part      # Keep as is, could be either
                
        return f"{first_part_fixed} {middle_part_fixed} {last_part_fixed}"
    
    # For all other formats, return the original text but with basic cleaning
    # Remove multiple spaces
    cleaned_text = ' '.join(text.split())
    
    # Add a space if the pattern appears to be alphanumerics divided by position
    # This handles cases like "ABC1234" turning into "ABC 1234"
    alpha_num_pattern = re.match(r'^([A-Z]{2,3})(\d{3,5})$', cleaned_text, re.IGNORECASE)
    if alpha_num_pattern:
        return f"{alpha_num_pattern.group(1)} {alpha_num_pattern.group(2)}"
    
    # Special case for vanity plates with mixed alphanumerics
    # For these, we'll just return the cleaned text
    return cleaned_text 