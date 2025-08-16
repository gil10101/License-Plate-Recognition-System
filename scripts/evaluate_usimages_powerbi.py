#!/usr/bin/env python3
"""
Comprehensive evaluation script for testing license plate recognition on US images dataset.
Generates Power BI compatible exports and detailed analytics.
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import cv2
import re
from typing import Dict, List, Tuple, Any
import glob

# Add project root to path
sys.path.append('.')

from lpr_system import LicensePlateSystem, SmartPlateReader

class USImagesEvaluator:
    def __init__(self, images_dir: str = "data/seg_and_ocr/usimages", groundtruth_file: str = "data/seg_and_ocr/groundtruth.csv", use_gpu: bool = True):
        """Initialize evaluator with image directory path."""
        self.images_dir = Path(images_dir)
        self.groundtruth_file = Path(groundtruth_file)
        self.use_gpu = use_gpu
        
        # Load groundtruth data
        self.groundtruth = self.load_groundtruth()
        
        # Check GPU availability
        import torch
        if use_gpu and torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.device = "cuda"
        else:
            print("Using CPU")
            self.device = "cpu"
            
        self.lpr_system = LicensePlateSystem()
        self.smart_reader = SmartPlateReader(use_gpu=use_gpu and torch.cuda.is_available())
        
        # Results storage
        self.results = []
        self.summary_stats = {}
        
        print(f"Initialized evaluator for: {self.images_dir}")
        print(f"Found {len(list(self.images_dir.glob('*.png')))} PNG images")
        print(f"Loaded {len(self.groundtruth)} groundtruth entries")
    
    def load_groundtruth(self) -> Dict[str, Dict[str, str]]:
        """Load groundtruth data from CSV file."""
        groundtruth = {}
        
        if not self.groundtruth_file.exists():
            print(f"Warning: Groundtruth file not found: {self.groundtruth_file}")
            return groundtruth
        
        try:
            with open(self.groundtruth_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 3:
                        filename = parts[0]
                        state_code = parts[1]
                        plate_text = parts[2]
                        
                        groundtruth[filename] = {
                            'state': state_code,
                            'text': plate_text
                        }
                    else:
                        print(f"Warning: Invalid line {line_num} in groundtruth: {line}")
        
        except Exception as e:
            print(f"Error loading groundtruth file: {e}")
        
        return groundtruth
    
    def get_expected_plate_text(self, filename: str) -> str:
        """Get expected plate text from groundtruth data."""
        if filename in self.groundtruth:
            return self.groundtruth[filename]['text']
        
        # Fallback: try to extract from filename pattern
        stem = Path(filename).stem
        match = re.match(r'^([a-z]{2})(\d+)$', stem.lower())
        if match:
            state_code = match.group(1).upper()
            number = match.group(2)
            return f"{state_code}{number}"
        
        return stem.upper()
    
    def get_plate_state(self, filename: str) -> str:
        """Get plate state from groundtruth data."""
        if filename in self.groundtruth:
            return self.groundtruth[filename]['state'].upper()
        
        # Fallback: extract from filename
        stem = Path(filename).stem.lower()
        match = re.match(r'^([a-z]{2})', stem)
        if match:
            return match.group(1).upper()
        
        return "UNKNOWN"
    
    def normalize_plate_text(self, text: str) -> str:
        """Normalize plate text for comparison."""
        if not text:
            return ""
        
        # Remove spaces, convert to uppercase, remove special chars
        normalized = re.sub(r'[^A-Z0-9]', '', text.upper())
        return normalized
    
    def calculate_character_accuracy(self, expected: str, actual: str) -> float:
        """Calculate character-level accuracy between expected and actual text."""
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0
        
        expected_norm = self.normalize_plate_text(expected)
        actual_norm = self.normalize_plate_text(actual)
        
        max_len = max(len(expected_norm), len(actual_norm))
        if max_len == 0:
            return 1.0
        
        matches = sum(1 for i in range(min(len(expected_norm), len(actual_norm))) 
                     if expected_norm[i] == actual_norm[i])
        
        return matches / max_len
    
    def categorize_plate_type(self, filename: str) -> str:
        """Categorize plate by state or type based on groundtruth or filename."""
        return self.get_plate_state(filename)
    
    def evaluate_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Evaluate a single image and return detailed results."""
        print(f"Processing: {image_path.name}")
        
        start_time = time.time()
        
        try:
            # Process image with LPR system (for detection)
            result = self.lpr_system.process_image(str(image_path), debug=False)
            
            # Extract expected text from groundtruth
            expected_text = self.get_expected_plate_text(image_path.name)
            
            # Get detection results
            detection_success = result.get('stage1_detection', {}).get('num_detections', 0) > 0
            detection_confidence = result.get('stage1_detection', {}).get('best_detection', {}).get('confidence', 0.0)
            
            # Use SmartPlateReader for enhanced OCR if detection succeeded
            if detection_success:
                # Load the original image for SmartPlateReader
                img = cv2.imread(str(image_path))
                if img is not None:
                    # Use SmartPlateReader for better OCR
                    smart_ocr_result = self.smart_reader.read_text(img)
                    actual_text = smart_ocr_result.get('text', '')
                    ocr_confidence = smart_ocr_result.get('confidence', 0.0)
                    ocr_status = smart_ocr_result.get('status', 'Unknown')
                    ocr_backend = smart_ocr_result.get('metadata', {}).get('backend', 'smart_plate_reader')
                else:
                    # Fallback to LPR system result
                    actual_text = result.get('final_result', {}).get('license_plate_text', '')
                    ocr_confidence = result.get('final_result', {}).get('confidence', 0.0)
                    ocr_status = result.get('stage2_ocr', {}).get('status', 'Unknown')
                    ocr_backend = result.get('stage2_ocr', {}).get('metadata', {}).get('backend', 'fallback')
            else:
                # No detection, try SmartPlateReader on full image as fallback
                img = cv2.imread(str(image_path))
                if img is not None:
                    smart_ocr_result = self.smart_reader.read_text(img)
                    actual_text = smart_ocr_result.get('text', '')
                    ocr_confidence = smart_ocr_result.get('confidence', 0.0)
                    ocr_status = f"No detection - {smart_ocr_result.get('status', 'Unknown')}"
                    ocr_backend = f"full_image_{smart_ocr_result.get('metadata', {}).get('backend', 'smart_plate_reader')}"
                else:
                    actual_text = ''
                    ocr_confidence = 0.0
                    ocr_status = 'Image load failed'
                    ocr_backend = 'none'
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            exact_match = self.normalize_plate_text(expected_text) == self.normalize_plate_text(actual_text)
            char_accuracy = self.calculate_character_accuracy(expected_text, actual_text)
            
            # Image info
            img = cv2.imread(str(image_path))
            image_height, image_width = img.shape[:2] if img is not None else (0, 0)
            
            evaluation_result = {
                'filename': image_path.name,
                'expected_text': expected_text,
                'actual_text': actual_text,
                'exact_match': exact_match,
                'character_accuracy': char_accuracy,
                'ocr_confidence': ocr_confidence,
                'detection_success': detection_success,
                'detection_confidence': detection_confidence,
                'ocr_status': ocr_status,
                'ocr_backend': ocr_backend,
                'processing_time_ms': processing_time * 1000,
                'image_width': image_width,
                'image_height': image_height,
                'plate_type': self.categorize_plate_type(image_path.name),
                'file_size_kb': image_path.stat().st_size / 1024,
                'timestamp': datetime.now().isoformat(),
                'success': result.get('final_result', {}).get('success', False)
            }
            
            return evaluation_result
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            return {
                'filename': image_path.name,
                'expected_text': self.get_expected_plate_text(image_path.name),
                'actual_text': '',
                'exact_match': False,
                'character_accuracy': 0.0,
                'ocr_confidence': 0.0,
                'detection_success': False,
                'detection_confidence': 0.0,
                'ocr_status': 'Error',
                'ocr_backend': 'Unknown',
                'processing_time_ms': 0.0,
                'image_width': 0,
                'image_height': 0,
                'plate_type': self.categorize_plate_type(image_path.name),
                'file_size_kb': 0.0,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def run_evaluation(self, max_images: int = 100) -> None:
        """Run evaluation on up to max_images from the dataset."""
        print(f"\n=== Starting US Images Evaluation (max {max_images} images) ===")
        
        # Get list of image files
        image_files = list(self.images_dir.glob('*.png'))[:max_images]
        
        if not image_files:
            print(f"No PNG images found in {self.images_dir}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i:3d}/{len(image_files)}] ", end="")
            result = self.evaluate_single_image(image_path)
            self.results.append(result)
        
        # Calculate summary statistics
        self.calculate_summary_stats()
        
        print(f"\n=== Evaluation Complete ===")
        print(f"Processed: {len(self.results)} images")
        print(f"Overall accuracy: {self.summary_stats['overall_exact_match_rate']:.2%}")
        print(f"Average character accuracy: {self.summary_stats['avg_character_accuracy']:.2%}")
        print(f"Detection success rate: {self.summary_stats['detection_success_rate']:.2%}")
    
    def calculate_summary_stats(self) -> None:
        """Calculate comprehensive summary statistics."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        self.summary_stats = {
            'total_images': len(self.results),
            'overall_exact_match_rate': df['exact_match'].mean(),
            'avg_character_accuracy': df['character_accuracy'].mean(),
            'detection_success_rate': df['detection_success'].mean(),
            'avg_ocr_confidence': df['ocr_confidence'].mean(),
            'avg_detection_confidence': df['detection_confidence'].mean(),
            'avg_processing_time_ms': df['processing_time_ms'].mean(),
            'total_processing_time_seconds': df['processing_time_ms'].sum() / 1000,
            
            # By plate type (state)
            'accuracy_by_state': df.groupby('plate_type')['exact_match'].mean().to_dict(),
            'char_accuracy_by_state': df.groupby('plate_type')['character_accuracy'].mean().to_dict(),
            
            # OCR backend performance
            'accuracy_by_backend': df.groupby('ocr_backend')['exact_match'].mean().to_dict(),
            
            # Confidence correlation
            'confidence_correlation': df[['ocr_confidence', 'exact_match']].corr().iloc[0, 1] if len(df) > 1 else 0.0,
            
            # Error analysis
            'ocr_status_distribution': df['ocr_status'].value_counts().to_dict(),
            'success_rate': df['success'].mean(),
            
            # Performance quartiles
            'processing_time_quartiles': {
                'q1': df['processing_time_ms'].quantile(0.25),
                'median': df['processing_time_ms'].quantile(0.5),
                'q3': df['processing_time_ms'].quantile(0.75)
            }
        }
    
    def export_to_powerbi(self, output_dir: str = "output") -> None:
        """Export results in Power BI compatible format."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Detailed results CSV
        detailed_csv = output_path / f"usimages_evaluation_detailed_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(detailed_csv, index=False)
        print(f"Detailed results exported to: {detailed_csv}")
        
        # 2. Summary statistics CSV
        summary_csv = output_path / f"usimages_evaluation_summary_{timestamp}.csv"
        summary_data = []
        
        for key, value in self.summary_stats.items():
            if isinstance(value, dict):
                # Flatten nested dictionaries
                for sub_key, sub_value in value.items():
                    summary_data.append({
                        'metric_category': key,
                        'metric_name': sub_key,
                        'metric_value': sub_value
                    })
            else:
                summary_data.append({
                    'metric_category': 'overall',
                    'metric_name': key,
                    'metric_value': value
                })
        
        pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
        print(f"Summary statistics exported to: {summary_csv}")
        
        # 3. State-wise performance CSV
        state_csv = output_path / f"usimages_evaluation_by_state_{timestamp}.csv"
        df_state = df.groupby('plate_type').agg({
            'exact_match': ['count', 'sum', 'mean'],
            'character_accuracy': 'mean',
            'ocr_confidence': 'mean',
            'detection_confidence': 'mean',
            'processing_time_ms': 'mean',
            'success': 'mean'
        }).round(4)
        
        # Flatten column names
        df_state.columns = ['_'.join(col).strip() for col in df_state.columns]
        df_state = df_state.reset_index()
        df_state.to_csv(state_csv, index=False)
        print(f"State-wise performance exported to: {state_csv}")
        
        # 4. Error analysis CSV
        error_csv = output_path / f"usimages_evaluation_errors_{timestamp}.csv"
        errors_df = df[~df['exact_match']].copy()
        if not errors_df.empty:
            errors_df.to_csv(error_csv, index=False)
            print(f"Error analysis exported to: {error_csv}")
        
        # 5. Performance distribution CSV
        perf_csv = output_path / f"usimages_evaluation_performance_{timestamp}.csv"
        perf_data = []
        
        # Binned processing times
        df['time_bin'] = pd.cut(df['processing_time_ms'], bins=10, labels=False)
        time_bins = df.groupby('time_bin').agg({
            'processing_time_ms': ['min', 'max', 'count', 'mean'],
            'exact_match': 'mean'
        })
        
        for i, row in time_bins.iterrows():
            perf_data.append({
                'bin_type': 'processing_time',
                'bin_number': i,
                'min_value': row[('processing_time_ms', 'min')],
                'max_value': row[('processing_time_ms', 'max')],
                'count': row[('processing_time_ms', 'count')],
                'avg_accuracy': row[('exact_match', 'mean')]
            })
        
        pd.DataFrame(perf_data).to_csv(perf_csv, index=False)
        print(f"Performance distribution exported to: {perf_csv}")
        
        # 6. JSON export for dashboard configuration
        json_export = output_path / f"usimages_evaluation_config_{timestamp}.json"
        config_data = {
            'evaluation_metadata': {
                'timestamp': timestamp,
                'total_images': len(self.results),
                'dataset_path': str(self.images_dir),
                'evaluation_version': '1.0'
            },
            'summary_metrics': self.summary_stats,
            'powerbi_files': {
                'detailed_results': str(detailed_csv),
                'summary_statistics': str(summary_csv),
                'state_performance': str(state_csv),
                'error_analysis': str(error_csv),
                'performance_distribution': str(perf_csv)
            }
        }
        
        with open(json_export, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        print(f"Dashboard configuration exported to: {json_export}")
        
        return {
            'detailed_csv': detailed_csv,
            'summary_csv': summary_csv,
            'state_csv': state_csv,
            'error_csv': error_csv,
            'perf_csv': perf_csv,
            'config_json': json_export
        }

def main():
    """Main execution function."""
    evaluator = USImagesEvaluator()
    
    # Run evaluation on 100 images
    evaluator.run_evaluation(max_images=100)
    
    # Export results for Power BI
    export_files = evaluator.export_to_powerbi()
    
    print("\n=== Power BI Export Summary ===")
    for file_type, file_path in export_files.items():
        print(f"{file_type}: {file_path}")
    
    print(f"\n=== Key Metrics ===")
    stats = evaluator.summary_stats
    print(f"Overall Accuracy: {stats['overall_exact_match_rate']:.2%}")
    print(f"Character Accuracy: {stats['avg_character_accuracy']:.2%}")
    print(f"Detection Success: {stats['detection_success_rate']:.2%}")
    print(f"Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"Total Processing Time: {stats['total_processing_time_seconds']:.1f}s")

if __name__ == "__main__":
    main()
