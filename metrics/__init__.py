"""
Comprehensive metrics module for License Plate Recognition System.

This module provides evaluation metrics for:
- Object detection performance
- OCR accuracy and quality
- System performance benchmarking
- Overall evaluation and reporting
"""

from .detection_metrics import DetectionMetrics
from .ocr_metrics import OCRMetrics
from .performance_metrics import PerformanceMetrics
from .evaluation_metrics import EvaluationMetrics
from .data_export import MetricsDataExporter, export_metrics_for_bi

__all__ = [
    'DetectionMetrics',
    'OCRMetrics', 
    'PerformanceMetrics',
    'EvaluationMetrics',
    'MetricsDataExporter',
    'export_metrics_for_bi'
]

__version__ = '1.0.0'
