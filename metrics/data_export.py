"""
Data export utilities for Power BI and Tableau integration.

This module provides comprehensive data export capabilities to feed
external BI tools like Power BI and Tableau with structured metrics data.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import csv
from datetime import datetime
import sqlite3

from .detection_metrics import DetectionMetrics
from .ocr_metrics import OCRMetrics  
from .performance_metrics import PerformanceMetrics
from .evaluation_metrics import EvaluationMetrics


class MetricsDataExporter:
    """
    Comprehensive data exporter for Business Intelligence tools.
    """
    
    def __init__(self, evaluation_metrics: EvaluationMetrics):
        """
        Initialize the data exporter.
        
        Args:
            evaluation_metrics: EvaluationMetrics instance with collected data
        """
        self.evaluation_metrics = evaluation_metrics
        self.export_timestamp = datetime.now()
    
    def export_for_powerbi(self, output_dir: str) -> Dict[str, str]:
        """
        Export all metrics data in Power BI friendly format.
        
        Args:
            output_dir: Directory to save the export files
            
        Returns:
            Dictionary mapping data type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # 1. Main results table - Core data for Power BI
        main_results_df = self._create_main_results_dataframe()
        main_path = output_path / "license_plate_results.csv"
        main_results_df.to_csv(main_path, index=False)
        exported_files['main_results'] = str(main_path)
        
        # 2. Detection metrics summary
        detection_summary_df = self._create_detection_summary_dataframe()
        detection_path = output_path / "detection_metrics.csv"
        detection_summary_df.to_csv(detection_path, index=False)
        exported_files['detection_metrics'] = str(detection_path)
        
        # 3. OCR performance by pattern
        ocr_pattern_df = self._create_ocr_pattern_dataframe()
        ocr_path = output_path / "ocr_pattern_performance.csv"
        ocr_pattern_df.to_csv(ocr_path, index=False)
        exported_files['ocr_patterns'] = str(ocr_path)
        
        # 4. Character confusion matrix
        char_confusion_df = self._create_character_confusion_dataframe()
        char_path = output_path / "character_confusion_matrix.csv"
        char_confusion_df.to_csv(char_path, index=False)
        exported_files['character_confusion'] = str(char_path)
        
        # 5. Performance metrics over time
        performance_df = self._create_performance_timeseries_dataframe()
        if not performance_df.empty:
            perf_path = output_path / "performance_timeseries.csv"
            performance_df.to_csv(perf_path, index=False)
            exported_files['performance_timeseries'] = str(perf_path)
        
        # 6. Summary KPIs for dashboard
        kpi_df = self._create_kpi_summary_dataframe()
        kpi_path = output_path / "kpi_summary.csv"
        kpi_df.to_csv(kpi_path, index=False)
        exported_files['kpi_summary'] = str(kpi_path)
        
        # 7. Error analysis
        error_df = self._create_error_analysis_dataframe()
        error_path = output_path / "error_analysis.csv"
        error_df.to_csv(error_path, index=False)
        exported_files['error_analysis'] = str(error_path)
        
        # 8. Metadata for Power BI context
        metadata_path = output_path / "export_metadata.json"
        self._save_export_metadata(metadata_path, exported_files)
        exported_files['metadata'] = str(metadata_path)
        
        return exported_files
    
    def export_for_tableau(self, output_dir: str) -> Dict[str, str]:
        """
        Export all metrics data in Tableau friendly format.
        
        Args:
            output_dir: Directory to save the export files
            
        Returns:
            Dictionary mapping data type to file path
        """
        # Tableau prefers similar format to Power BI, but with some specific optimizations
        return self.export_for_powerbi(output_dir)
    
    def export_to_sqlite(self, db_path: str) -> str:
        """
        Export all data to SQLite database for flexible querying.
        
        Args:
            db_path: Path to SQLite database file
            
        Returns:
            Path to created database
        """
        conn = sqlite3.connect(db_path)
        
        try:
            # Main results table
            main_df = self._create_main_results_dataframe()
            main_df.to_sql('license_plate_results', conn, if_exists='replace', index=False)
            
            # Detection metrics
            detection_df = self._create_detection_summary_dataframe()
            detection_df.to_sql('detection_metrics', conn, if_exists='replace', index=False)
            
            # OCR patterns
            ocr_pattern_df = self._create_ocr_pattern_dataframe()
            ocr_pattern_df.to_sql('ocr_pattern_performance', conn, if_exists='replace', index=False)
            
            # Character confusion
            char_confusion_df = self._create_character_confusion_dataframe()
            char_confusion_df.to_sql('character_confusion_matrix', conn, if_exists='replace', index=False)
            
            # Performance data
            performance_df = self._create_performance_timeseries_dataframe()
            if not performance_df.empty:
                performance_df.to_sql('performance_timeseries', conn, if_exists='replace', index=False)
            
            # KPI summary
            kpi_df = self._create_kpi_summary_dataframe()
            kpi_df.to_sql('kpi_summary', conn, if_exists='replace', index=False)
            
            # Error analysis
            error_df = self._create_error_analysis_dataframe()
            error_df.to_sql('error_analysis', conn, if_exists='replace', index=False)
            
            # Create indexes for better query performance
            self._create_database_indexes(conn)
            
        finally:
            conn.close()
        
        return db_path
    
    def _create_main_results_dataframe(self) -> pd.DataFrame:
        """Create the main results dataframe with all key metrics."""
        data = []
        
        for result in self.evaluation_metrics.end_to_end_results:
            # Extract image metadata
            image_size_category = 'unknown'
            if result.image_path:
                try:
                    import cv2
                    img = cv2.imread(result.image_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        pixels = h * w
                        if pixels < 500_000:
                            image_size_category = 'small'
                        elif pixels < 2_000_000:
                            image_size_category = 'medium'
                        else:
                            image_size_category = 'large'
                except:
                    pass
            
            # Determine license plate pattern
            gt_text = result.ground_truth_text.upper().strip().replace(' ', '')
            plate_pattern = 'other'
            if len(gt_text) >= 5:
                import re
                if re.match(r'^[A-Z]{3}[0-9]{4}$', gt_text):
                    plate_pattern = '3L4N'
                elif re.match(r'^[A-Z]{2}[0-9]{5}$', gt_text):
                    plate_pattern = '2L5N'
                elif re.match(r'^[A-Z]{4}[0-9]{3}$', gt_text):
                    plate_pattern = '4L3N'
                elif re.match(r'^[A-Z]+$', gt_text):
                    plate_pattern = 'vanity'
                elif re.match(r'^[A-Z0-9]+$', gt_text):
                    plate_pattern = 'mixed'
            
            data.append({
                'image_id': result.image_id,
                'image_path': result.image_path,
                'export_timestamp': self.export_timestamp,
                'predicted_text': result.predicted_text,
                'ground_truth_text': result.ground_truth_text,
                'plate_pattern': plate_pattern,
                'image_size_category': image_size_category,
                'detection_success': result.detection_success,
                'detection_confidence': result.detection_confidence,
                'detection_iou': result.detection_iou,
                'ocr_success': result.ocr_success,
                'ocr_confidence': result.ocr_confidence,
                'text_similarity': result.text_similarity,
                'end_to_end_success': result.end_to_end_success,
                'character_accuracy': self._calculate_character_accuracy(result.predicted_text, result.ground_truth_text),
                'edit_distance': self._calculate_edit_distance(result.predicted_text, result.ground_truth_text),
                'total_processing_time': result.total_processing_time,
                'detection_time': result.detection_time,
                'ocr_time': result.ocr_time,
                'memory_usage': result.memory_usage,
                # Failure categorization
                'failure_type': self._categorize_failure(result),
                'quality_score': self._calculate_sample_quality_score(result)
            })
        
        return pd.DataFrame(data)
    
    def _create_detection_summary_dataframe(self) -> pd.DataFrame:
        """Create detection metrics summary dataframe."""
        detection_report = self.evaluation_metrics.detection_metrics.get_detailed_report()
        
        data = [{
            'metric_type': 'detection_summary',
            'export_timestamp': self.export_timestamp,
            'total_images': detection_report.get('summary', {}).get('total_images', 0),
            'precision': detection_report.get('detection_metrics', {}).get('precision', 0),
            'recall': detection_report.get('detection_metrics', {}).get('recall', 0),
            'f1_score': detection_report.get('detection_metrics', {}).get('f1_score', 0),
            'average_precision': detection_report.get('detection_metrics', {}).get('average_precision', 0),
            'mean_average_precision': detection_report.get('detection_metrics', {}).get('mean_average_precision', 0),
            'true_positives': detection_report.get('confusion_matrix', {}).get('true_positives', 0),
            'false_positives': detection_report.get('confusion_matrix', {}).get('false_positives', 0),
            'false_negatives': detection_report.get('confusion_matrix', {}).get('false_negatives', 0),
            'iou_threshold': detection_report.get('summary', {}).get('iou_threshold', 0.5),
            'confidence_threshold': detection_report.get('summary', {}).get('confidence_threshold', 0.5)
        }]
        
        return pd.DataFrame(data)
    
    def _create_ocr_pattern_dataframe(self) -> pd.DataFrame:
        """Create OCR performance by pattern dataframe."""
        ocr_report = self.evaluation_metrics.ocr_metrics.get_detailed_report()
        pattern_analysis = ocr_report.get('pattern_analysis', {})
        
        data = []
        for pattern, metrics in pattern_analysis.items():
            data.append({
                'pattern': pattern,
                'export_timestamp': self.export_timestamp,
                'sample_count': metrics.get('count', 0),
                'exact_match_accuracy': metrics.get('exact_match_accuracy', 0),
                'character_accuracy': metrics.get('character_accuracy', 0),
                'average_similarity': metrics.get('average_similarity', 0),
                'accuracy_std': metrics.get('accuracy_std', 0)
            })
        
        return pd.DataFrame(data)
    
    def _create_character_confusion_dataframe(self) -> pd.DataFrame:
        """Create character confusion matrix dataframe."""
        char_analysis = self.evaluation_metrics.ocr_metrics.analyze_character_errors()
        confusion_pairs = char_analysis.get('most_confused_pairs', [])
        
        data = []
        for pair in confusion_pairs:
            data.append({
                'actual_character': pair['actual'],
                'predicted_character': pair['predicted'],
                'confusion_count': pair['count'],
                'error_rate': pair['error_rate'],
                'export_timestamp': self.export_timestamp
            })
        
        return pd.DataFrame(data)
    
    def _create_performance_timeseries_dataframe(self) -> pd.DataFrame:
        """Create performance metrics timeseries dataframe."""
        performance_metrics = self.evaluation_metrics.performance_metrics.metrics
        
        if not performance_metrics:
            return pd.DataFrame()
        
        data = []
        for metric in performance_metrics:
            data.append({
                'timestamp': pd.to_datetime(metric.start_time, unit='s'),
                'operation': metric.operation,
                'duration': metric.duration,
                'memory_before': metric.memory_before,
                'memory_after': metric.memory_after,
                'memory_delta': metric.memory_delta,
                'cpu_percent': metric.cpu_percent,
                'image_width': metric.image_size[0],
                'image_height': metric.image_size[1],
                'processing_rate': metric.processing_rate,
                'success': metric.success,
                'export_timestamp': self.export_timestamp
            })
        
        return pd.DataFrame(data)
    
    def _create_kpi_summary_dataframe(self) -> pd.DataFrame:
        """Create KPI summary dataframe for dashboard tiles."""
        comprehensive_report = self.evaluation_metrics.get_comprehensive_report()
        end_to_end_metrics = comprehensive_report.get('end_to_end_metrics', {}).get('overall_metrics', {})
        quality_assessment = comprehensive_report.get('quality_assessment', {})
        
        data = [{
            'kpi_name': 'End-to-End Success Rate',
            'kpi_value': end_to_end_metrics.get('end_to_end_success_rate', 0),
            'kpi_category': 'Overall Performance',
            'target_value': 0.85,  # Target 85% success rate
            'export_timestamp': self.export_timestamp
        }, {
            'kpi_name': 'Detection Success Rate',
            'kpi_value': end_to_end_metrics.get('detection_success_rate', 0),
            'kpi_category': 'Detection Performance',
            'target_value': 0.90,  # Target 90% detection rate
            'export_timestamp': self.export_timestamp
        }, {
            'kpi_name': 'OCR Success Rate',
            'kpi_value': end_to_end_metrics.get('ocr_success_rate', 0),
            'kpi_category': 'OCR Performance',
            'target_value': 0.80,  # Target 80% OCR success rate
            'export_timestamp': self.export_timestamp
        }, {
            'kpi_name': 'Overall Quality Score',
            'kpi_value': quality_assessment.get('overall_quality_score', 0),
            'kpi_category': 'Quality Assessment',
            'target_value': 0.75,  # Target 75% overall quality
            'export_timestamp': self.export_timestamp
        }, {
            'kpi_name': 'Total Samples Processed',
            'kpi_value': len(self.evaluation_metrics.end_to_end_results),
            'kpi_category': 'Volume',
            'target_value': None,
            'export_timestamp': self.export_timestamp
        }]
        
        # Add component scores
        component_scores = quality_assessment.get('component_scores', {})
        for component, score in component_scores.items():
            data.append({
                'kpi_name': f'{component.replace("_", " ").title()}',
                'kpi_value': score,
                'kpi_category': 'Component Performance',
                'target_value': 0.70,
                'export_timestamp': self.export_timestamp
            })
        
        return pd.DataFrame(data)
    
    def _create_error_analysis_dataframe(self) -> pd.DataFrame:
        """Create error analysis dataframe."""
        failure_analysis = self.evaluation_metrics.analyze_failure_modes()
        
        data = []
        
        # Detection failures
        det_failures = failure_analysis.get('detection_failures', {})
        data.append({
            'error_category': 'Detection Failures',
            'error_count': det_failures.get('count', 0),
            'error_percentage': det_failures.get('percentage', 0),
            'error_subcategory': 'Total',
            'export_timestamp': self.export_timestamp
        })
        data.append({
            'error_category': 'Detection Failures',
            'error_count': det_failures.get('low_confidence_detections', 0),
            'error_percentage': det_failures.get('low_confidence_detections', 0) / max(det_failures.get('count', 1), 1) * 100,
            'error_subcategory': 'Low Confidence',
            'export_timestamp': self.export_timestamp
        })
        data.append({
            'error_category': 'Detection Failures',
            'error_count': det_failures.get('no_detections', 0),
            'error_percentage': det_failures.get('no_detections', 0) / max(det_failures.get('count', 1), 1) * 100,
            'error_subcategory': 'No Detection',
            'export_timestamp': self.export_timestamp
        })
        
        # OCR failures
        ocr_failures = failure_analysis.get('ocr_failures', {})
        data.append({
            'error_category': 'OCR Failures',
            'error_count': ocr_failures.get('count', 0),
            'error_percentage': ocr_failures.get('percentage', 0),
            'error_subcategory': 'Total',
            'export_timestamp': self.export_timestamp
        })
        data.append({
            'error_category': 'OCR Failures',
            'error_count': ocr_failures.get('empty_predictions', 0),
            'error_percentage': ocr_failures.get('empty_predictions', 0) / max(ocr_failures.get('count', 1), 1) * 100,
            'error_subcategory': 'Empty Predictions',
            'export_timestamp': self.export_timestamp
        })
        
        # Text accuracy failures
        text_failures = failure_analysis.get('text_accuracy_failures', {})
        data.append({
            'error_category': 'Text Accuracy Failures',
            'error_count': text_failures.get('count', 0),
            'error_percentage': text_failures.get('percentage', 0),
            'error_subcategory': 'Total',
            'export_timestamp': self.export_timestamp
        })
        data.append({
            'error_category': 'Text Accuracy Failures',
            'error_count': text_failures.get('partial_matches', 0),
            'error_percentage': text_failures.get('partial_matches', 0) / max(text_failures.get('count', 1), 1) * 100,
            'error_subcategory': 'Partial Matches',
            'export_timestamp': self.export_timestamp
        })
        
        return pd.DataFrame(data)
    
    def _save_export_metadata(self, filepath: str, exported_files: Dict[str, str]) -> None:
        """Save export metadata."""
        metadata = {
            'export_timestamp': self.export_timestamp.isoformat(),
            'total_samples': len(self.evaluation_metrics.end_to_end_results),
            'exported_files': exported_files,
            'data_schema': {
                'main_results': {
                    'description': 'Main results table with all sample-level metrics',
                    'key_columns': ['image_id', 'end_to_end_success', 'detection_success', 'ocr_success']
                },
                'detection_metrics': {
                    'description': 'Detection performance summary metrics',
                    'key_columns': ['precision', 'recall', 'f1_score']
                },
                'ocr_pattern_performance': {
                    'description': 'OCR performance broken down by license plate pattern',
                    'key_columns': ['pattern', 'exact_match_accuracy', 'character_accuracy']
                },
                'kpi_summary': {
                    'description': 'Key performance indicators for dashboard',
                    'key_columns': ['kpi_name', 'kpi_value', 'target_value']
                }
            },
            'powerbi_connection_info': {
                'recommended_refresh_rate': 'daily',
                'data_source_type': 'csv',
                'main_table': 'license_plate_results'
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _create_database_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for better query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_main_image_id ON license_plate_results(image_id)",
            "CREATE INDEX IF NOT EXISTS idx_main_success ON license_plate_results(end_to_end_success)",
            "CREATE INDEX IF NOT EXISTS idx_main_pattern ON license_plate_results(plate_pattern)",
            "CREATE INDEX IF NOT EXISTS idx_perf_operation ON performance_timeseries(operation)",
            "CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_timeseries(timestamp)",
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.Error:
                pass  # Index might already exist
        
        conn.commit()
    
    def _calculate_character_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Calculate character-level accuracy."""
        if not ground_truth:
            return 1.0 if not predicted else 0.0
        
        pred = predicted.upper().strip()
        gt = ground_truth.upper().strip()
        
        if not gt:
            return 1.0 if not pred else 0.0
        
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            return 1.0
        
        correct_chars = sum(1 for i, (p, g) in enumerate(zip(pred, gt)) if p == g)
        
        if len(pred) != len(gt):
            correct_chars -= abs(len(pred) - len(gt))
        
        return max(0, correct_chars) / max_len
    
    def _calculate_edit_distance(self, predicted: str, ground_truth: str) -> int:
        """Calculate Levenshtein edit distance."""
        try:
            import Levenshtein
            return Levenshtein.distance(predicted.upper().strip(), ground_truth.upper().strip())
        except ImportError:
            # Fallback implementation
            pred = predicted.upper().strip()
            gt = ground_truth.upper().strip()
            
            if len(pred) < len(gt):
                return self._calculate_edit_distance(gt, pred)
            
            if len(gt) == 0:
                return len(pred)
            
            previous_row = range(len(gt) + 1)
            for i, c1 in enumerate(pred):
                current_row = [i + 1]
                for j, c2 in enumerate(gt):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
    
    def _categorize_failure(self, result) -> str:
        """Categorize the type of failure for a result."""
        if result.end_to_end_success:
            return 'success'
        elif not result.detection_success:
            return 'detection_failure'
        elif not result.ocr_success:
            return 'ocr_failure'
        else:
            return 'text_accuracy_failure'
    
    def _calculate_sample_quality_score(self, result) -> float:
        """Calculate a quality score for a single sample."""
        weights = {
            'detection': 0.3,
            'ocr': 0.4,
            'text_similarity': 0.3
        }
        
        detection_score = 1.0 if result.detection_success else 0.0
        ocr_score = 1.0 if result.ocr_success else 0.0
        similarity_score = result.text_similarity
        
        return (weights['detection'] * detection_score + 
                weights['ocr'] * ocr_score + 
                weights['text_similarity'] * similarity_score)


def export_metrics_for_bi(evaluation_metrics: EvaluationMetrics, 
                         output_format: str = 'powerbi',
                         output_dir: str = 'bi_export') -> Dict[str, str]:
    """
    Convenience function to export metrics for BI tools.
    
    Args:
        evaluation_metrics: EvaluationMetrics instance
        output_format: 'powerbi', 'tableau', or 'sqlite'
        output_dir: Output directory for files
        
    Returns:
        Dictionary of exported file paths
    """
    exporter = MetricsDataExporter(evaluation_metrics)
    
    if output_format.lower() == 'powerbi':
        return exporter.export_for_powerbi(output_dir)
    elif output_format.lower() == 'tableau':
        return exporter.export_for_tableau(output_dir)
    elif output_format.lower() == 'sqlite':
        db_path = Path(output_dir) / 'metrics_database.db'
        exporter.export_to_sqlite(str(db_path))
        return {'database': str(db_path)}
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

