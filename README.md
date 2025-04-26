# License Plate Recognition System

A Flask-based web application that detects and reads license plates from uploaded images using PyTorch, OpenCV, and Tesseract OCR.

## Features

- Image upload interface
- License plate detection using YOLOv5
- Custom license plate detector training
- Image preprocessing with OpenCV
- Character recognition with Tesseract OCR
- GPU acceleration with CUDA/cuDNN (when available)

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd License-Plate-Recognition-System
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`

4. Set the Tesseract executable path in the `app.py` file if it's not in your system PATH.

## Usage

1. Run the Flask application:
```
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000`

3. Upload an image containing a license plate

4. View the detection results

## Custom License Plate Detector

The system offers the ability to train a custom license plate detector using your own dataset:

1. Navigate to the Training page from the main interface

2. Configure the training parameters:
   - Epochs: Number of training iterations (default: 100)
   - Batch size: Number of images processed per batch (default: 16)

3. Start training - this will:
   - Prepare the license plate dataset from your training images
   - Train a YOLOv5 model specifically for license plate detection
   - Automatically use the trained model once complete

Training uses the augmented license plate images in the `data/train` and `data/val` directories. You can add more license plate images to these directories to improve detection accuracy.

You can also run training separately:
```
python train_license_detector.py --epochs 100 --batch-size 16
```

## Project Structure

- `app.py`: Flask web application for the main interface
- `detector/`: License plate detection models
  - `detect.py`: Base YOLOv5 implementation for license plate detection
  - `custom_plate_detector.py`: Enhanced detector with custom training capabilities
- `utils/`: Utility functions
  - `image_processing.py`: OpenCV utilities for image preprocessing and text recognition
- `scripts/`: Helper scripts
  - `enhanced_recognition.py`: Advanced license plate recognition with improved preprocessing
  - `train_improved_ocr.py`: Training script for OCR model optimization
  - `improved_license_recognition.py`: Core recognition implementation
- `data/`: Data storage
  - `prepare_license_data.py`: Script to prepare data for training
  - `extracted_plates/`: Directory for extracted license plate images
  - `train/`, `val/`: Training and validation datasets
- `templates/`: HTML templates for the web interface
- `static/`: Static assets, CSS, JavaScript, and uploaded images
- `train_license_detector.py`: Script to train the custom detector

## Requirements

- Python 3.7+
- Flask
- PyTorch
- OpenCV
- NumPy
- Pytesseract (Tesseract OCR wrapper)
- Ultralytics YOLOv5
- Werkzeug
- Pathlib
- Difflib
- CUDA/cuDNN (optional, for GPU acceleration)