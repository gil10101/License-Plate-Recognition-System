import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

from detector.detect import LicensePlateDetector
from utils.image_processing import preprocess_plate, recognize_text

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

# Initialize the license plate detector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Try to use the custom license plate detector if available,
# otherwise fall back to the original detector
try:
    from detector.custom_plate_detector import CustomLicensePlateDetector
    
    # Check if custom model exists
    custom_model_paths = [
        'license_plate_model.pt',
        'data/license_plate_data/weights/best.pt',
        'license_plate_detection/train/weights/best.pt',
        'yolov5n.pt',  # Small model for efficiency
        'yolov5s.pt'   # More accurate but slower
    ]
    
    custom_model_path = None
    for path in custom_model_paths:
        if os.path.exists(path):
            custom_model_path = path
            break
    
    if custom_model_path:
        print(f"Using custom license plate detector with model: {custom_model_path}")
        # Use a lower confidence threshold (0.25) to capture more potential plates
        detector = CustomLicensePlateDetector(weights_path=custom_model_path, confidence_threshold=0.25, device=device)
        using_custom_detector = True
    else:
        print("Custom model not found, using original detector")
        # Lower confidence threshold for original detector too
        detector = LicensePlateDetector(confidence_threshold=0.25, device=device)
        using_custom_detector = False
except ImportError:
    print("Custom detector not available, using original detector")
    detector = LicensePlateDetector(confidence_threshold=0.25, device=device)
    using_custom_detector = False

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
            
            # Process the image
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
                    'processed_image': processed_filename
                })
            
            # Draw bounding boxes on the image
            output_img = image.copy()
            for result in results:
                x1, y1, x2, y2 = result['coordinates']
                text = result['text']
                conf = result['confidence']
                
                # Color based on confidence (green for high confidence, yellow for medium, red for low)
                if conf >= 0.7:
                    color = (0, 255, 0)  # Green
                elif conf >= 0.5:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw rectangle with the color
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw text with confidence
                text_with_conf = f"{text} ({conf:.2f})"
                cv2.putText(output_img, text_with_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Save the output image
            output_filename = f"output_{filename}"
            output_filepath = os.path.join('static/images', output_filename)
            cv2.imwrite(output_filepath, output_img)
            
            # Save results to session
            session['results'] = {
                'original': filename,
                'output': output_filename,
                'plates': results,
                'using_custom_detector': using_custom_detector
            }
            
            return redirect(url_for('results'))
    
    return render_template('index.html', using_custom_detector=using_custom_detector)

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