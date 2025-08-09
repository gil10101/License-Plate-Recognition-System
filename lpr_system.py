#!/usr/bin/env python3
"""
Consolidated License Plate Recognition System
- Stage 1: License plate detection (YOLO)
- Stage 2: Smart OCR (EasyOCR with corrections)

This file aggregates the previous enterprise pipeline and smart OCR logic into a
single, importable module for use by the Flask app and scripts.
"""

import os
import time
from typing import Dict, Any

import cv2
import numpy as np

# Add current directory to path for imports
import sys
sys.path.append('.')

from detector.custom_plate_detector import CustomLicensePlateDetector
import easyocr


class SmartPlateReader:
    """Smart OCR adapter around EasyOCR with plate-aware post-processing."""

    def __init__(self, use_gpu: bool = True) -> None:
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape
        if height < 80:
            scale = 80 / max(1, height)
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 80), interpolation=cv2.INTER_CUBIC)
        return gray

    def looks_like_plate(self, text: str) -> bool:
        if not text:
            return False
        text = ''.join(c for c in text.upper() if c.isalnum())
        if len(text) < 3 or len(text) > 8:
            return False
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        return has_letters or has_numbers

    def apply_corrections(self, text: str) -> str:
        if not text:
            return ""
        t = text.upper()
        mapping = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1', '|': '1',
            'S': '5', 'Z': '2', 'B': '8', 'G': '6'
        }
        corrected = ''.join(mapping.get(ch, ch) for ch in t)
        return corrected

    def estimate_confidence(self, text: str) -> float:
        if not text:
            return 0.0
        score = 0.3
        n = len(text)
        if 5 <= n <= 7:
            score += 0.4
        elif 3 <= n <= 8:
            score += 0.2
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        if has_letters and has_numbers:
            score += 0.3
        elif has_letters or has_numbers:
            score += 0.1
        if len(set(text)) > 1:
            score += 0.1
        return min(score, 1.0)

    def read_text(self, plate_image: np.ndarray) -> Dict[str, Any]:
        processed = self.preprocess_image(plate_image)
        detections = self.reader.readtext(processed)
        if not detections:
            return {
                'text': '',
                'confidence': 0.0,
                'status': 'No text detected',
                'metadata': {'processing_method': 'smart_plate_reader'}
            }

        # Pick best candidate by likelihood and EasyOCR confidence
        best_text = ''
        best_score = -1.0
        for bbox, raw_text, conf in detections:
            clean = ''.join(c for c in raw_text.upper() if c.isalnum())
            corrected = self.apply_corrections(clean)
            score = conf
            if self.looks_like_plate(corrected):
                score += 0.5
            if score > best_score:
                best_score = score
                best_text = corrected

        return {
            'text': best_text,
            'confidence': self.estimate_confidence(best_text),
            'status': 'Success' if best_text else 'No text detected',
            'metadata': {'processing_method': 'smart_plate_reader'}
        }


class LicensePlateSystem:
    """Unified detection + OCR pipeline."""

    def __init__(self, detector_weights: str | None = None) -> None:
        self.detector = self._initialize_detector(detector_weights)
        self.ocr = SmartPlateReader(use_gpu=True)

    def _initialize_detector(self, weights_path: str | None) -> CustomLicensePlateDetector:
        if not weights_path:
            candidates = [
                "license_plate_detection/ocr_optimized/weights/best.pt",
                "license_plate_detection/train/weights/best.pt",
            ]
            for p in candidates:
                if os.path.exists(p):
                    weights_path = p
                    break
            if not weights_path:
                weights_path = "yolov5s.pt"
        return CustomLicensePlateDetector(weights_path=weights_path, confidence_threshold=0.25)

    def process_image(self, image_path: str, debug: bool = False) -> Dict[str, Any]:
        start = time.time()
        image = cv2.imread(image_path)
        if image is None:
            return self._error_result("Could not load image", image_path)

        # Stage 1: detect
        try:
            detections = self.detector.detect(image)
        except Exception as e:
            detections = []
            if debug:
                print(f"Detection error: {e}")

        plate_image = image
        stage1_result: Dict[str, Any]
        if not detections:
            stage1_result = {
                'num_detections': 0,
                'best_detection': None,
                'fallback_used': True,
                'fallback_reason': 'no_detection'
            }
        else:
            x1, y1, x2, y2, conf, class_id = max(detections, key=lambda x: x[4])
            if conf < 0.4:
                stage1_result = {
                    'num_detections': len(detections),
                    'best_detection': {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': int(class_id)
                    },
                    'fallback_used': True,
                    'fallback_reason': 'low_confidence'
                }
            else:
                padding = 30
                y1c = max(0, y1 - padding)
                y2c = min(image.shape[0], y2 + padding)
                x1c = max(0, x1 - padding)
                x2c = min(image.shape[1], x2 + padding)
                crop = image[y1c:y2c, x1c:x2c]
                h, w = crop.shape[:2]
                if min(h, w) < 60 or h < 40 or w < 120:
                    stage1_result = {
                        'num_detections': len(detections),
                        'best_detection': {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_id': int(class_id)
                        },
                        'fallback_used': True,
                        'fallback_reason': 'small_crop'
                    }
                else:
                    plate_image = crop
                    stage1_result = {
                        'num_detections': len(detections),
                        'best_detection': {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_id': int(class_id)
                        },
                        'fallback_used': False
                    }

        # Stage 2: OCR
        ocr_res = self.ocr.read_text(plate_image)

        elapsed = round(time.time() - start, 3)
        final_text = ocr_res.get('text', '')
        final_conf = ocr_res.get('confidence', 0.0)

        return {
            'image_path': image_path,
            'image_size': f"{image.shape[1]}x{image.shape[0]}",
            'processing_time': elapsed,
            'stage1_detection': stage1_result,
            'stage2_ocr': ocr_res,
            'final_result': {
                'license_plate_text': final_text,
                'confidence': final_conf,
                'success': bool(final_text),
                'processing_time_ms': round(elapsed * 1000, 1)
            }
        }

    def _error_result(self, msg: str, image_path: str) -> Dict[str, Any]:
        return {
            'image_path': image_path,
            'error': msg,
            'success': False,
            'final_result': {
                'license_plate_text': '',
                'confidence': 0.0,
                'success': False,
                'error': msg
            }
        }


__all__ = ["LicensePlateSystem", "SmartPlateReader"]


