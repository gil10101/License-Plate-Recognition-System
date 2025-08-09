import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import sys

# Add current directory to path for imports
sys.path.append('.')

from detector.detect import LicensePlateDetector
from utils.image_processing import preprocess_plate, recognize_text
from lpr_system import LicensePlateSystem

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'license_plate_recognition_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize the enterprise license plate system
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initializing Enterprise License Plate System on device: {device}")

# Initialize Enterprise LPR System
try:
    enterprise_system = LicensePlateSystem()
    using_enterprise_system = True
    print("✅ Enterprise License Plate System initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize Enterprise System: {e}")
    print("🔄 Falling back to original detector...")
    
    # Fallback to original system
    try:
        from detector.custom_plate_detector import CustomLicensePlateDetector
        
        # Check if custom model exists
        custom_model_paths = [
            'license_plate_detection/ocr_optimized/weights/best.pt',
            'license_plate_detection/train/weights/best.pt',
            'license_plate_model.pt',
            'yolov5s.pt'
        ]
        
        custom_model_path = None
        for path in custom_model_paths:
            if os.path.exists(path):
                custom_model_path = path
                break
        
        if custom_model_path:
            print(f"Using custom license plate detector with model: {custom_model_path}")
            detector = CustomLicensePlateDetector(weights_path=custom_model_path, confidence_threshold=0.25, device=device)
            using_enterprise_system = False
        else:
            print("Custom model not found, using original detector")
            detector = LicensePlateDetector(confidence_threshold=0.25, device=device)
            using_enterprise_system = False
    except ImportError:
        print("Custom detector not available, using original detector")
        detector = LicensePlateDetector(confidence_threshold=0.25, device=device)
        using_enterprise_system = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image with Enterprise System
            if using_enterprise_system:
                # Use Enterprise License Plate System
                enterprise_result = enterprise_system.process_image(filepath, debug=False)
                
                if enterprise_result['final_result']['success']:
                    # Extract results from enterprise system
                    results = []
                    
                    # Get detection coordinates if available
                    stage1 = enterprise_result.get('stage1_detection', {})
                    if stage1 and not stage1.get('fallback_used', False):
                        detection = stage1.get('best_detection', {})
                        if detection:
                            bbox = detection['bbox']
                            x1, y1, x2, y2 = bbox
                            conf = detection['confidence']
                            
                            # Load image for cropping and display
                            image = cv2.imread(filepath)
                            
                            # Crop detected plate for display
                            plate_img = image[y1:y2, x1:x2]
                            processed_filename = f"enterprise_plate_{filename}"
                            processed_filepath = os.path.join('static/images', processed_filename)
                            cv2.imwrite(processed_filepath, plate_img)
                            
                            results.append({
                                'coordinates': (x1, y1, x2, y2),
                                'confidence': float(conf),
                                'text': enterprise_result['final_result']['license_plate_text'],
                                'processed_image': processed_filename,
                                'enterprise_system': True,
                                'processing_time': enterprise_result['processing_time']
                            })
                    else:
                        # Fallback was used (whole image)
                        image = cv2.imread(filepath)
                        h, w = image.shape[:2]
                        
                        # Save original image as processed plate
                        processed_filename = f"enterprise_fallback_{filename}"
                        processed_filepath = os.path.join('static/images', processed_filename)
                        cv2.imwrite(processed_filepath, image)
                        
                        results.append({
                            'coordinates': (0, 0, w, h),
                            'confidence': 1.0,
                            'text': enterprise_result['final_result']['license_plate_text'],
                            'processed_image': processed_filename,
                            'enterprise_system': True,
                            'processing_time': enterprise_result['processing_time'],
                            'fallback_used': True
                        })
                else:
                    flash(f"Enterprise system failed: {enterprise_result.get('error', 'Unknown error')}")
                    return redirect(request.url)
            else:
                # Use Original System (fallback)
                image = cv2.imread(filepath)
                if image is None:
                    flash('Error reading the image')
                    return redirect(request.url)
                
                # Try multiple scales for better detection
                results = []
                scales = [1.0]  # Start with original size
                
                # For smaller images, also try upscaling
                if min(image.shape[0], image.shape[1]) < 800:
                    scales.append(1.5)  # Add 1.5x upscaling
                    
                # For larger images, also try downscaling
                if max(image.shape[0], image.shape[1]) > 1600:
                    scales.append(0.5)  # Add 0.5x downscaling
                
                all_detections = []
                
                # Try detection at different scales
                for scale in scales:
                    if scale != 1.0:
                        scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                    else:
                        scaled_image = image
                    
                    # Detect license plates
                    detections = detector.detect(scaled_image)
                    
                    # Adjust coordinates back to original scale if needed
                    if scale != 1.0:
                        for i in range(len(detections)):
                            detections[i][0] = int(detections[i][0] / scale)  # x1
                            detections[i][1] = int(detections[i][1] / scale)  # y1
                            detections[i][2] = int(detections[i][2] / scale)  # x2
                            detections[i][3] = int(detections[i][3] / scale)  # y2
                    
                    all_detections.extend(detections)
                
                # Non-maximum suppression to remove duplicate detections
                if len(all_detections) > 1:
                    # Convert to numpy array
                    boxes = np.array([[d[0], d[1], d[2], d[3]] for d in all_detections])
                    confidences = np.array([d[4] for d in all_detections])
                    
                    # Perform non-maximum suppression
                    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), 0.25, 0.45)
                    
                    # Filter detections based on NMS results
                    filtered_detections = [all_detections[i] for i in indices.flatten()]
                    all_detections = filtered_detections
                
                # Process detections
                for i, (x1, y1, x2, y2, conf, _) in enumerate(all_detections):
                    # Ensure coordinates are within image bounds
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(image.shape[1], int(x2))
                    y2 = min(image.shape[0], int(y2))
                    
                    # Crop license plate
                    plate_img = image[y1:y2, x1:x2]
                    
                    # Skip if plate is too small
                    if plate_img.size == 0 or plate_img.shape[0] < 10 or plate_img.shape[1] < 10:
                        continue
                    
                    # Preprocess plate for OCR - returns multiple preprocessed versions
                    processed_plates = preprocess_plate(plate_img)
                    
                    if not processed_plates:
                        continue
                    
                    # Save the first preprocessed plate for display
                    processed_filename = f"plate_{i}_{filename}"
                    processed_filepath = os.path.join('static/images', processed_filename)
                    if processed_plates and len(processed_plates) > 0:
                        first_processed = processed_plates[0]
                        if first_processed is not None and first_processed.size > 0:
                            # Ensure image is in BGR format for saving
                            if len(first_processed.shape) == 3 and first_processed.shape[2] == 3:
                                if first_processed.dtype != np.uint8:
                                    first_processed = (first_processed * 255).astype(np.uint8)
                                # Convert RGB to BGR if needed
                                if first_processed[0, 0, 0] > first_processed[0, 0, 2]:
                                    first_processed = cv2.cvtColor(first_processed, cv2.COLOR_RGB2BGR)
                            # If grayscale, convert to BGR for consistent saving
                            elif len(first_processed.shape) == 2:
                                first_processed = cv2.cvtColor(first_processed, cv2.COLOR_GRAY2BGR)
                            
                            cv2.imwrite(processed_filepath, first_processed)
                    
                    # Recognize text from the multiple preprocessed versions
                    plate_text = recognize_text(processed_plates)
                    
                    results.append({
                        'coordinates': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'text': plate_text,
                        'processed_image': processed_filename,
                        'enterprise_system': False
                    })
            
            # Load image for drawing bounding boxes
            image = cv2.imread(filepath)
            
            # Draw bounding boxes on the image
            output_img = image.copy()
            for result in results:
                x1, y1, x2, y2 = result['coordinates']
                text = result['text']
                conf = result['confidence']
                
                # Color based on confidence and system type
                if result.get('enterprise_system', False):
                    # Enterprise system - use blue tones
                    if conf >= 0.7:
                        color = (255, 0, 0)  # Blue
                    elif conf >= 0.5:
                        color = (255, 255, 0)  # Cyan
                    else:
                        color = (128, 0, 255)  # Purple
                else:
                    # Original system - use traditional colors
                    if conf >= 0.7:
                        color = (0, 255, 0)  # Green
                    elif conf >= 0.5:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                
                # Draw rectangle with the color
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw text with confidence and processing time if available
                text_with_conf = f"{text} ({conf:.2f})"
                if 'processing_time' in result:
                    text_with_conf += f" [{result['processing_time']:.3f}s]"
                
                # Add system indicator
                system_indicator = "🏢" if result.get('enterprise_system', False) else "🔧"
                text_with_conf = f"{system_indicator} {text_with_conf}"
                
                cv2.putText(output_img, text_with_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save the output image
            output_filename = f"output_{filename}"
            output_filepath = os.path.join('static/images', output_filename)
            cv2.imwrite(output_filepath, output_img)
            
            # Save results to session
            session['results'] = {
                'original': filename,
                'output': output_filename,
                'plates': results,
                'using_enterprise_system': using_enterprise_system,
                'system_type': 'Enterprise LPR System' if using_enterprise_system else 'Original System'
            }
            
            return redirect(url_for('results'))
    
    return render_template('index.html', using_enterprise_system=using_enterprise_system)

@app.route('/results')
def results():
    if 'results' not in session:
        return redirect(url_for('index'))
    
    return render_template('results.html', results=session['results'])

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    try:
        epochs = int(request.form.get('epochs', 100))
        batch_size = int(request.form.get('batch_size', 16))
        
        # Run training in a background thread
        import threading
        import subprocess
        
        def train_model():
            subprocess.run([
                'python', 'train_license_detector.py',
                '--epochs', str(epochs),
                '--batch-size', str(batch_size)
            ])
        
        thread = threading.Thread(target=train_model)
        thread.daemon = True
        thread.start()
        
        flash('Training started in the background. Please check the console for progress.')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error starting training: {str(e)}')
        return redirect(url_for('train'))

if __name__ == '__main__':
    print(f"Running on device: {device}")
    app.run(debug=True) 