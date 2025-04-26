#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import glob
import random
import numpy as np

def main():
    """
    Validate a trained license plate detector model.
    This script allows you to test the performance of a trained model
    on validation images.
    """
    # Force CUDA to be enabled if available
    import torch
    if torch.cuda.is_available():
        print("CUDA is available. Enabling GPU detection.")
        torch.cuda.set_device(0)  # Use the first GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
    else:
        print("WARNING: CUDA is not available. Detection will be slower on CPU.")
    
    parser = argparse.ArgumentParser(description='Validate a trained license plate detector.')
    parser.add_argument('--weights', type=str, default='license_plate_model.pt', 
                        help='Path to the trained model weights')
    parser.add_argument('--val-dir', type=str, 
                        help='Directory containing validation images (if not specified, will use default locations)')
    parser.add_argument('--num-images', type=int, default=5, 
                        help='Number of random validation images to test')
    parser.add_argument('--conf-threshold', type=float, default=0.25, 
                        help='Confidence threshold for detections')
    parser.add_argument('--ocr-model', type=str, default='pytesseract',
                        choices=['pytesseract', 'easyocr'],
                        help='OCR model to use for license plate recognition')
    parser.add_argument('--save-crops', action='store_true',
                        help='Save cropped license plate images for manual inspection')
    args = parser.parse_args()
    
    print("License Plate Detector Validation")
    print("================================")
    
    # Check if model exists
    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found at {args.weights}")
        print("Please specify the correct path to the trained model weights.")
        sys.exit(1)
    
    # Create directory for saving crops if needed
    val_crops_dir = 'val_manual'
    if args.save_crops:
        os.makedirs(val_crops_dir, exist_ok=True)
        print(f"Cropped license plates will be saved to: {val_crops_dir}")
    
    # Try to import OCR libraries
    ocr_available = False
    
    if args.ocr_model == 'pytesseract':
        try:
            import pytesseract
            # Set Tesseract executable path
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            ocr_available = True
            print("Using Tesseract OCR for plate recognition")
        except ImportError:
            print("Warning: pytesseract not installed. Install with: pip install pytesseract")
            print("Note: Tesseract OCR engine must also be installed on your system")
    elif args.ocr_model == 'easyocr':
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            ocr_available = True
            print("Using EasyOCR for plate recognition")
        except ImportError:
            print("Warning: easyocr not installed. Install with: pip install easyocr")
    
    if not ocr_available:
        print("No OCR library available. Will only detect plates, not read text.")
    
    try:
        # Import the detector
        from detector.custom_plate_detector import CustomLicensePlateDetector
        
        # Load the trained detector
        print(f"\nLoading license plate detector from {args.weights}...")
        detector = CustomLicensePlateDetector(weights_path=args.weights)
        
        # Find validation images to test
        val_images = []
        if args.val_dir and os.path.exists(args.val_dir):
            val_images = glob.glob(os.path.join(args.val_dir, '*.png')) + \
                        glob.glob(os.path.join(args.val_dir, '*.jpg'))
        
        # If no validation directory specified or no images found, try default locations
        if not val_images:
            val_images = glob.glob('data/val/*.png') + glob.glob('data/val/*.jpg')
        if not val_images:
            val_images = glob.glob('data/license_plate_data/images/val/*.png') + \
                        glob.glob('data/license_plate_data/images/val/*.jpg')
        
        if not val_images:
            print("No validation images found. Please specify a directory with --val-dir.")
            sys.exit(1)
        
        print(f"Found {len(val_images)} validation images.")
        
        # Test the detector on random images
        num_test = min(args.num_images, len(val_images))
        test_images = random.sample(val_images, num_test)
        
        def preprocess_plate_image(plate_img):
            """Preprocess license plate image for better OCR results"""
            # Resize to higher resolution while maintaining aspect ratio
            height, width = plate_img.shape[:2]
            new_height = 600  # Increased height for even better resolution
            new_width = int(width * (new_height / height))
            resized = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to remove noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Increase contrast with CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)
            
            # Return enhanced grayscale image
            return enhanced
        
        def extract_plate_roi(img, detection):
            """Extract the license plate ROI from detection"""
            
            # Extract the original image dimensions
            img_height, img_width = img.shape[:2]
            
            # Initialize with default values
            x1, y1, x2, y2 = 0, 0, 0, 0
            
            # Handle different detection formats
            if isinstance(detection, (list, tuple, np.ndarray)):
                # Check detector output format
                if len(detection) >= 4:
                    # Check if positions are in normalized format [0-1] or absolute coordinates
                    if isinstance(detection[0], (int, float)) and isinstance(detection[2], (int, float)):
                        # Check if the values are very small (normalized)
                        if 0 <= detection[0] <= 1 and 0 <= detection[2] <= 1:
                            # Normalized coordinates [x1, y1, x2, y2] in [0,1] range
                            x1 = int(detection[0] * img_width)
                            y1 = int(detection[1] * img_height)
                            x2 = int(detection[2] * img_width)
                            y2 = int(detection[3] * img_height)
                        else:
                            # Absolute coordinates [x1, y1, x2, y2]
                            x1 = int(detection[0])
                            y1 = int(detection[1])
                            x2 = int(detection[2])
                            y2 = int(detection[3])
                            
                            # If any value is much larger than the image dimensions, they might be in
                            # [x, y, width, height] format instead of [x1, y1, x2, y2]
                            if x2 > img_width*1.5 or y2 > img_height*1.5:
                                # Convert [x, y, width, height] to [x1, y1, x2, y2]
                                x2 = x1 + x2
                                y2 = y1 + y2
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Add a small buffer margin around the plate for better OCR
            buffer_x = int((x2 - x1) * 0.05)  # 5% buffer on each side
            buffer_y = int((y2 - y1) * 0.05)
            
            x1 = max(0, x1 - buffer_x)
            y1 = max(0, y1 - buffer_y)
            x2 = min(img_width, x2 + buffer_x)
            y2 = min(img_height, y2 + buffer_y)
            
            # Check if the ROI is valid
            if x1 >= x2 or y1 >= y2:
                return None, (0, 0, 0, 0)
                
            # Extract the ROI
            plate_img = img[y1:y2, x1:x2].copy()
            return plate_img, (x1, y1, x2, y2)
        
        def read_license_plate(img, detection, img_name=None, idx=0):
            """Extract and read text from license plate"""
            if not ocr_available:
                return "OCR not available", None
            
            # Create a copy of the original image for visualization
            viz_img = img.copy()
            
            # Extract the license plate region
            plate_img, box_coords = extract_plate_roi(img, detection)
            
            if plate_img is None or plate_img.size == 0:
                return "Empty plate region", None
            
            # Draw green rectangle around the license plate
            x1, y1, x2, y2 = box_coords
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green rectangle with thickness 3
            
            # Save visualization image
            if args.save_crops:
                base_name = os.path.splitext(os.path.basename(img_name))[0]
                viz_filename = f"{val_crops_dir}/{base_name}_detection_{idx}.png"
                cv2.imwrite(viz_filename, viz_img)
                print(f"  Saved visualization to: {viz_filename}")
            
            # Save cropped license plate for debugging
            if args.save_crops:
                base_name = os.path.splitext(os.path.basename(img_name))[0]
                crop_filename = f"{val_crops_dir}/{base_name}_plate_{idx}.png"
                cv2.imwrite(crop_filename, plate_img)
                print(f"  Saved cropped plate to: {crop_filename}")
            
            # Preprocess the plate image - get enhanced grayscale version
            enhanced_plate = preprocess_plate_image(plate_img)
            
            # Save preprocessed image for debugging
            if args.save_crops:
                base_name = os.path.splitext(os.path.basename(img_name))[0]
                enhanced_filename = f"{val_crops_dir}/{base_name}_plate_{idx}_enhanced.png"
                cv2.imwrite(enhanced_filename, enhanced_plate)
            
            # Perform OCR based on selected method
            try:
                if args.ocr_model == 'pytesseract':
                    # Configure tesseract for license plates
                    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                    
                    # Try OCR on enhanced grayscale image
                    enhanced_text = pytesseract.image_to_string(enhanced_plate, config=custom_config).strip()
                    
                    # Try OCR on original cropped image as backup
                    orig_text = pytesseract.image_to_string(plate_img, config=custom_config).strip()
                    
                    # Use the result with the most characters
                    texts = [t for t in [enhanced_text, orig_text] if t]
                    if texts:
                        text = max(texts, key=len)
                        return text, plate_img
                    else:
                        return "No text detected", plate_img
                        
                elif args.ocr_model == 'easyocr':
                    # Try on original image first, then on enhanced if no result
                    result = reader.readtext(plate_img)
                    if not result:
                        result = reader.readtext(enhanced_plate)
                    
                    if result:
                        text = ' '.join([item[1] for item in result])
                        return text, plate_img
                    else:
                        return "No text detected", plate_img
            except Exception as e:
                return f"OCR error: {str(e)}", plate_img
        
        print(f"\nValidating on {num_test} random images...")
        for img_path in test_images:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read image: {img_path}")
                continue
            
            print(f"\nTesting on {os.path.basename(img_path)}...")
            # Detect license plates
            detections = detector.detect(image)
            
            if detections:
                print(f"Found {len(detections)} license plates:")
                for i, detection in enumerate(detections):
                    # Handle different detection formats
                    plate_text, plate_img = read_license_plate(image, detection, img_path, i)
                    
                    # Print detection and text info
                    print(f"  Plate #{i+1}: {detection}")
                    print(f"  Text: {plate_text}")
            else:
                print("No license plates detected")
        
        print("\nValidation completed successfully.")
    except Exception as e:
        print(f"Error validating detector: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 