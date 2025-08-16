## Enterprise Upgrade Task List

This document tracks the implementation of proposed upgrades to improve precision, accuracy, and robustness. Tasks are executed in order; each is crossed off when implemented and tested.

### 1) OCR improvements: plate rectification + PaddleOCR fallback
- [x] Add perspective rectification of plate crops before OCR in `lpr_system.py`.
- [x] Integrate PaddleOCR as an additional OCR backend in `SmartPlateReader`, re-rank with EasyOCR.
- [x] Graceful fallback when PaddleOCR is unavailable.
- [x] Lightweight test: `scripts/test_task1_smart_plate_reader.py` stubs OCR backends and validates rectification and aggregation logic.

### 2) Detector improvements: multi-scale TTA + Weighted Boxes Fusion (WBF)
- [x] Add multi-scale inference (0.75x, 1.0x, 1.25x) in `detector/custom_plate_detector.py`.
- [x] Fuse boxes with `ensemble-boxes` WBF; return fused detections.
- [x] Fallback to existing detection flow if no boxes produced.
- [x] Lightweight test: `scripts/test_task2_tta_wbf.py` monkey-patches the model to produce boxes and verifies robust behavior without heavy model load.

### 3) Confidence calibration for OCR
- [ ] Add isotonic regression calibration using `data/seg_and_ocr/results.csv`; export to `models/calibration/ocr_isotonic.joblib`.
- [ ] Load and apply calibrated confidence in `SmartPlateReader` when available.
- [ ] Notebook/script: `scripts/calibrate_ocr_confidence.py` and plotting utilities.

### 4) Confusion-aware post-processing and pattern re-ranking
- [ ] Build character confusion priors from results; save to `models/ocr_model/char_corrections.json`.
- [ ] Use weighted edit distance + pattern constraints to re-rank OCR candidates in `SmartPlateReader`.
- [ ] Tests on synthetic and benchmarked cases.

### 5) Detector retraining for small objects and strong augmentations
- [ ] Add Albumentations pipeline for small-plate heavy augmentation.
- [ ] Training config: higher `imgsz`, mosaic/perspective, CIoU/EIoU, EMA; evaluate per-size mAP.
- [ ] Report mAP improvements; integrate best weights.

### Notes
- Dependencies updated in `requirements.txt` for tasks 1 and 2 (EasyOCR, PaddleOCR, ensemble-boxes). Further deps (scikit-learn, joblib, albumentations) will be added with tasks 3 and 5.


