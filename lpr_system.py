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
# import easyocr  # Commented out - using PaddleOCR instead
import torch
try:
    from models.ocr.char_cnn import load_charcnn
except Exception:
    load_charcnn = None  # optional
try:
    # Optional fallback to legacy Tesseract pipeline
    from utils.image_processing import preprocess_plate as legacy_preprocess_plate, recognize_text as legacy_recognize_text
except Exception:
    legacy_preprocess_plate = None
    legacy_recognize_text = None
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover
    PaddleOCR = None  # Optional dependency; handled gracefully


class SmartPlateReader:
    """Smart OCR adapter around EasyOCR with plate-aware post-processing."""

    def __init__(self, use_gpu: bool = True) -> None:
        # Use PaddleOCR as primary OCR engine
        self.reader = None  # EasyOCR disabled
        self.paddle = None
        try:
            if PaddleOCR is not None:
                # Prefer minimal preprocessing to avoid wrong rotations on small crops
                self.paddle = PaddleOCR(
                    use_angle_cls=False,  # Disable rotation detection
                    lang='en'
                )
                print("PaddleOCR initialized successfully as primary OCR engine")
        except Exception as e:
            print(f"PaddleOCR initialization failed: {e}")
            self.paddle = None
        
        # Use PaddleOCR only
        self.primary_backend = 'paddle' if self.paddle is not None else 'none'
        
        if self.paddle is None:
            raise RuntimeError("No OCR backend available. Please install PaddleOCR.")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape
        # Upscale aggressively for small crops to aid OCR
        target_min_height = 180
        if height < target_min_height:
            scale = target_min_height / max(1, height)
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, target_min_height), interpolation=cv2.INTER_CUBIC)
        # Contrast enhance
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass
        return gray

    def _preprocess_variants(self, image: np.ndarray) -> list[np.ndarray]:
        variants: list[np.ndarray] = []
        base = self.preprocess_image(image)
        variants.append(base)
        # Add text-band focused crop
        try:
            band = self._extract_text_band(image)
            if band is not None and band.size > 0:
                variants.append(self.preprocess_image(band))
        except Exception:
            pass
        # Threshold variants
        try:
            _, th = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(th)
            variants.append(cv2.bitwise_not(th))
        except Exception:
            pass
        # Sharpen
        try:
            k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharp = cv2.filter2D(base, -1, k)
            variants.append(sharp)
        except Exception:
            pass
        # Adaptive threshold
        try:
            ad = cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
            variants.append(ad)
        except Exception:
            pass
        # Larger upscale variant
        try:
            h, w = base.shape
            big = cv2.resize(base, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
            variants.append(big)
        except Exception:
            pass
        # Deduplicate by shape pointer id
        uniq = []
        seen = set()
        for v in variants:
            key = (v.shape, int(v.mean()))
            if key not in seen:
                uniq.append(v)
                seen.add(key)
        return uniq

    def _score_plate_text(self, text: str) -> float:
        if not text:
            return 0.0
        s = 0.0
        t = text.replace(' ', '')
        # pattern bonus
        import re
        if re.match(r'^[A-Z]{3}\d{4}$', t):
            s += 1.0
        if re.match(r'^[A-Z]{2,3}\s?\d{3,5}$', t):
            s += 0.6
        # length bonus
        if 5 <= len(t) <= 7:
            s += 0.3
        # alnum mix bonus
        has_l = any(c.isalpha() for c in t)
        has_n = any(c.isdigit() for c in t)
        if has_l and has_n:
            s += 0.2
        return s

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
        # Conservative mapping: only map in numeric contexts
        import re
        # If text fits letters+digits pattern, only correct the numeric block
        m = re.match(r'^([A-Z]{2,4})\s*([A-Z0-9]{2,6})$', t)
        if m:
            left, right = m.group(1), m.group(2)
            num_map = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', '|': '1'}
            right_corr = ''.join(num_map.get(c, c) for c in right)
            return f"{left} {right_corr}"
        # General case: apply light fixes where a letter is surrounded by digits
        chars = list(t)
        for i, c in enumerate(chars):
            if i > 0 and i < len(chars) - 1:
                if chars[i-1].isdigit() and chars[i+1].isdigit():
                    if c == 'O': chars[i] = '0'
                    elif c in ('I', 'L', '|'): chars[i] = '1'
                    elif c == 'S': chars[i] = '5'
                    elif c == 'Z': chars[i] = '2'
        return ''.join(chars)

    def _extract_text_band(self, plate_bgr: np.ndarray) -> np.ndarray | None:
        """Extract the dominant horizontal text band from a plate crop."""
        if plate_bgr is None or plate_bgr.size == 0:
            return None
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr
        # Normalize and binarize
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        edges = cv2.Canny(gray, 80, 200)
        # Connect horizontal strokes
        h, w = edges.shape
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 20), 3))
        band_map = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        # Horizontal projection
        proj = band_map.sum(axis=1)
        window = max(10, h // 3)
        best_y, best_val = 0, -1
        acc = proj.cumsum()
        for y in range(0, h - window):
            val = acc[y + window] - acc[y]
            if val > best_val:
                best_val = val
                best_y = y
        y1 = max(0, best_y - h // 20)
        y2 = min(h, best_y + window + h // 20)
        band = plate_bgr[y1:y2, :]
        return band

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
        # Prefer PaddleOCR results first when available (try multiple preprocess variants)
        if self.paddle is not None:
            try:
                # Ensure 3-channel input and use predict() API
                padd = []
                for var in self._preprocess_variants(plate_image):
                    rgb3 = cv2.cvtColor(var, cv2.COLOR_GRAY2BGR) if var.ndim == 2 else var
                    out = self.paddle.predict(rgb3)  # list of dicts
                    if out:
                        padd.extend(out)
            except Exception:
                padd = None
            if padd:
                best_text = ''
                best_score = -1.0
                for out in padd:
                    rec_texts = out.get('rec_texts', []) or []
                    rec_scores = out.get('rec_scores', []) or []
                    for t, sc in zip(rec_texts, rec_scores):
                        clean = ''.join(c for c in str(t).upper() if c.isalnum())
                        corrected = self.apply_corrections(clean)
                        score = float(sc) + self._score_plate_text(corrected)
                        if self.looks_like_plate(corrected):
                            score += 0.5
                        if corrected and score > best_score:
                            best_text = corrected
                            best_score = score
                if best_text:
                    return {
                        'text': best_text,
                        'confidence': self.estimate_confidence(best_text),
                        'status': 'Success',
                        'metadata': {'processing_method': 'smart_plate_reader', 'backend': 'paddleocr'}
                    }

        # No EasyOCR fallback - PaddleOCR only
        # If PaddleOCR failed, return empty result
        return {
            'text': '',
            'confidence': 0.0,
            'status': 'No text detected',
            'metadata': {'processing_method': 'smart_plate_reader', 'backend': 'paddleocr'}
        }

    def _rectify_plate(self, crop: np.ndarray) -> np.ndarray:
        """Attempt to de-skew plate region using perspective transform."""
        try:
            if crop is None or crop.size == 0:
                return crop
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            edges = cv2.Canny(gray, 80, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return crop
            c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)
            # order points tl,tr,br,bl
            s = box.sum(axis=1)
            diff = np.diff(box, axis=1).ravel()
            tl, br = box[np.argmin(s)], box[np.argmax(s)]
            tr, bl = box[np.argmin(diff)], box[np.argmax(diff)]
            w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
            h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
            if w < 1 or h < 1:
                return crop
            dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst)
            warped = cv2.warpPerspective(crop, M, (w, h))
            return warped
        except Exception:
            return crop


class LicensePlateSystem:
    """Unified detection + OCR pipeline."""

    def __init__(self, detector_weights: str | None = None) -> None:
        self.detector = self._initialize_detector(detector_weights)
        self.ocr = SmartPlateReader(use_gpu=True)
        # Optional char-level CNN
        self.char_model = None
        self.char_classes = None
        try:
            if load_charcnn is not None and os.path.exists('models/ocr_charcnn/charcnn_best.pt'):
                self.char_model, self.char_classes = load_charcnn(
                    'models/ocr_charcnn/charcnn_best.pt', 'models/ocr_charcnn/classes.json'
                )
        except Exception:
            self.char_model = None
            self.char_classes = None

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
        # Increase detector inference resolution for more detailed crops
        return CustomLicensePlateDetector(weights_path=weights_path, confidence_threshold=0.25, inference_imgsz=1280)

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
                padding = 60
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

        # Rectify plate region before OCR
        try:
            plate_image = self.ocr._rectify_plate(plate_image)
        except Exception:
            pass

        # Stage 2: OCR
        ocr_res = self.ocr.read_text(plate_image)
        # Optional char-level fallback when full-line OCR weak
        if (not ocr_res.get('text')) and self.char_model is not None:
            try:
                band = self.ocr._extract_text_band(plate_image) or plate_image
                gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
                # Simple segmentation
                _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((1, 3), np.uint8)
                binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(binv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
                chars = []
                device = next(self.char_model.parameters()).device
                with torch.no_grad():
                    for c in contours:
                        x, y, w, h = cv2.boundingRect(c)
                        if h >= 0.4 * gray.shape[0] and w >= 6:
                            roi = gray[y:y+h, x:x+w]
                            roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_CUBIC)
                            t = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).float().div(255.0).sub(0.5).div(0.5).to(device)
                            logits = self.char_model(t)
                            idx = int(logits.argmax(dim=1).item())
                            chars.append(self.char_classes[idx])
                pred = ''.join(chars)
                if pred:
                    ocr_res = {
                        'text': pred,
                        'confidence': self.estimate_confidence(pred),
                        'status': 'Success',
                        'metadata': {'processing_method': 'smart_plate_reader', 'backend': 'charcnn'}
                    }
            except Exception:
                pass
        if not ocr_res.get('text') and legacy_preprocess_plate and legacy_recognize_text:
            try:
                variants = legacy_preprocess_plate(plate_image)
                legacy_text = legacy_recognize_text(variants)
                if legacy_text:
                    ocr_res = {
                        'text': legacy_text,
                        'confidence': self.estimate_confidence(legacy_text),
                        'status': 'Success',
                        'metadata': {'processing_method': 'smart_plate_reader', 'backend': 'tesseract_utils'}
                    }
            except Exception:
                pass

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


