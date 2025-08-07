# License Plate Recognition System

A comprehensive Flask-based license plate recognition system that combines YOLOv5 object detection with advanced OCR capabilities. The system features custom detector training, multi-stage image preprocessing, and comprehensive performance metrics with Power BI/Tableau integration.

## Features

### Core Recognition Capabilities
- **Multi-scale License Plate Detection**: YOLOv5/YOLO11 with custom training support
- **Advanced Image Preprocessing**: 15+ preprocessing methods for optimal OCR results
- **Robust OCR Engine**: Tesseract with multiple configurations and post-processing
- **End-to-End Pipeline**: From traffic scenes to accurate text recognition
- **GPU Acceleration**: CUDA/cuDNN support for faster inference

### Training & Evaluation
- **Custom Detector Training**: Train YOLOv5 models on your specific license plate data
- **Comprehensive Metrics Suite**: Detection, OCR, and performance evaluation
- **Business Intelligence Integration**: Export to Power BI and Tableau
- **Benchmark Testing**: Automated evaluation on large datasets

### Web Interface
- **Drag-and-Drop Upload**: Modern web interface with real-time results
- **Multi-format Support**: PNG, JPG, JPEG image processing
- **Confidence Visualization**: Color-coded detection results
- **Training Interface**: Easy-to-use model training controls

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

## Comprehensive Metrics & Evaluation

The system includes a comprehensive metrics suite for evaluating performance and exporting data to Business Intelligence tools.

### Quick Evaluation

Evaluate your system performance on a dataset:

```bash
python evaluate_system.py --csv results.csv --format powerbi --output-dir metrics_export
```

This generates structured data files optimized for Power BI or Tableau import.

### Metrics Modules

- **Detection Metrics**: Precision, Recall, F1-score, IoU, mAP analysis
- **OCR Metrics**: Character accuracy, string accuracy, pattern matching, confidence correlation
- **Performance Metrics**: Processing time, memory usage, throughput analysis
- **Evaluation Metrics**: End-to-end pipeline evaluation and quality scoring

### Business Intelligence Integration

Export evaluation results for Power BI and Tableau:

```python
from metrics import export_metrics_for_bi, EvaluationMetrics

# Load your results
evaluation = EvaluationMetrics()
evaluation.evaluate_from_results_csv('your_results.csv')

# Export for Power BI
files = export_metrics_for_bi(evaluation, 'powerbi', 'export_folder')
```

The exported files include:
- `license_plate_results.csv` - Main results with all metrics
- `kpi_summary.csv` - Key performance indicators for dashboards  
- `detection_metrics.csv` - Detection performance summary
- `ocr_pattern_performance.csv` - OCR accuracy by license plate pattern
- `character_confusion_matrix.csv` - Character-level error analysis
- `error_analysis.csv` - Failure mode categorization

### Power BI Dashboard Example

Here's an example of training metrics visualized in Power BI using the exported CSV data:

![Training Metrics Dashboard](results.png)

The dashboard shows:
- **Training Progress**: Loss curves for box detection and classification over epochs
- **Model Performance**: Precision, Recall, and mAP metrics progression
- **Learning Rate Analysis**: Correlation between learning rate and validation loss
- **Training Efficiency**: Relationship between training loss and model accuracy (mAP50)

This demonstrates how the metrics system integrates seamlessly with Power BI to provide comprehensive insights into model training and performance.

## Project Structure

```
License-Plate-Recognition-System/
├── app.py                          # Main Flask application
├── evaluate_system.py              # Comprehensive evaluation script
├── requirements.txt                # Python dependencies
├── 
├── detector/                       # Detection engine
│   ├── detect.py                   # Base license plate detector
│   └── custom_plate_detector.py    # Enhanced detector with training
├── 
├── utils/                          # Image processing utilities
│   └── image_processing.py         # Preprocessing and OCR pipeline
├── 
├── metrics/                        # Comprehensive metrics suite
│   ├── detection_metrics.py        # Detection performance metrics
│   ├── ocr_metrics.py              # OCR accuracy metrics
│   ├── performance_metrics.py      # System performance metrics
│   ├── evaluation_metrics.py       # End-to-end evaluation
│   └── data_export.py              # BI tools integration
├── 
├── scripts/                        # Training and testing scripts
│   ├── enhanced_recognition.py     # Advanced recognition pipeline
│   ├── train_license_detector.py   # Custom detector training
│   └── validate.py                 # Model validation
├── 
├── templates/                      # Web interface templates
│   ├── index.html                  # Upload interface
│   ├── results.html                # Results display
│   └── train.html                  # Training interface
├── 
├── static/                         # Static web assets
│   ├── uploads/                    # User uploaded images
│   └── images/                     # Processed results
├── 
└── data/                          # Training and test data
    └── license_plate_detection/    # Trained model weights
```

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