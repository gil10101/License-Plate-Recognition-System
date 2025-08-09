"""
Comprehensive evaluation metrics for the license plate recognition system.

This module provides end-to-end evaluation capabilities that combine
detection, OCR, and performance metrics to give a complete picture
of system performance including:
- Overall system accuracy
- End-to-end pipeline evaluation
- Comparative analysis
- Benchmark reporting
- Quality scoring
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import cv2
from collections import defaultdict
import warnings

from .detection_metrics import DetectionMetrics, BoundingBox
from .ocr_metrics import OCRMetrics, OCRResult
from .performance_metrics import PerformanceMetrics


@dataclass
class EndToEndResult:
    """Represents a complete end-to-end evaluation result."""
    image_id: str
    image_path: str
    
    # Detection results
    detected_boxes: List[BoundingBox]
    ground_truth_boxes: List[BoundingBox]
    detection_success: bool
    detection_confidence: float
    
    # OCR results
    predicted_text: str
    ground_truth_text: str
    ocr_confidence: float
    ocr_success: bool
    
    # Performance metrics
    total_processing_time: float
    detection_time: float
    ocr_time: float
    memory_usage: float
    
    # Overall success
    end_to_end_success: bool
    
    @property
    def detection_iou(self) -> float:
        """Calculate best IoU between detected and ground truth boxes."""
        if not self.detected_boxes or not self.ground_truth_boxes:
            return 0.0
        
        best_iou = 0.0
        for det_box in self.detected_boxes:
            for gt_box in self.ground_truth_boxes:
                # Calculate IoU
                x1 = max(det_box.x1, gt_box.x1)
                y1 = max(det_box.y1, gt_box.y1)
                x2 = min(det_box.x2, gt_box.x2)
                y2 = min(det_box.y2, gt_box.y2)
                
                if x2 <= x1 or y2 <= y1:
                    iou = 0.0
                else:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = det_box.area + gt_box.area - intersection
                    iou = intersection / union if union > 0 else 0.0
                
                best_iou = max(best_iou, iou)
        
        return best_iou
    
    @property
    def text_similarity(self) -> float:
        """Calculate text similarity between predicted and ground truth."""
        from difflib import SequenceMatcher
        pred = self.predicted_text.upper().strip()
        gt = self.ground_truth_text.upper().strip()
        return SequenceMatcher(None, pred, gt).ratio()


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for the license plate recognition system.
    """
    
    def __init__(self, iou_threshold: float = 0.5, confidence_threshold: float = 0.5):
        """
        Initialize evaluation metrics.
        
        Args:
            iou_threshold: IoU threshold for detection evaluation
            confidence_threshold: Confidence threshold for filtering detections
        """
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        
        # Initialize component metrics
        self.detection_metrics = DetectionMetrics(iou_threshold, confidence_threshold)
        self.ocr_metrics = OCRMetrics()
        self.performance_metrics = PerformanceMetrics()
        
        # Store end-to-end results
        self.end_to_end_results: List[EndToEndResult] = []
        
    def add_end_to_end_result(self, result: EndToEndResult) -> None:
        """
        Add an end-to-end evaluation result.
        
        Args:
            result: EndToEndResult instance
        """
        self.end_to_end_results.append(result)
        
        # Also add to component metrics
        self.detection_metrics.add_result(
            result.image_id,
            result.detected_boxes,
            result.ground_truth_boxes
        )
        
        self.ocr_metrics.add_result(
            result.image_id,
            result.predicted_text,
            result.ground_truth_text,
            result.ocr_confidence
        )
    
    def evaluate_from_results_csv(self, csv_path: str, images_dir: str = "", 
                                 detection_results: Optional[Dict[str, List[BoundingBox]]] = None,
                                 ground_truth_boxes: Optional[Dict[str, List[BoundingBox]]] = None) -> None:
        """
        Evaluate system performance from a results CSV file.
        
        Args:
            csv_path: Path to CSV file with results
            images_dir: Directory containing images
            detection_results: Optional detection results mapping
            ground_truth_boxes: Optional ground truth boxes mapping
        """
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            image_id = str(row.get('image', ''))
            predicted_text = str(row.get('predicted', '')) if pd.notna(row.get('predicted')) else ""
            ground_truth_text = str(row.get('expected', '')) if pd.notna(row.get('expected')) else ""
            confidence = float(row.get('confidence', 0.0)) if pd.notna(row.get('confidence')) else 0.0
            
            # Get detection results if available
            detected_boxes = detection_results.get(image_id, []) if detection_results else []
            gt_boxes = ground_truth_boxes.get(image_id, []) if ground_truth_boxes else []
            
            # Create end-to-end result
            result = EndToEndResult(
                image_id=image_id,
                image_path=str(Path(images_dir) / image_id) if images_dir else image_id,
                detected_boxes=detected_boxes,
                ground_truth_boxes=gt_boxes,
                detection_success=len(detected_boxes) > 0,
                detection_confidence=max([box.confidence for box in detected_boxes], default=0.0),
                predicted_text=predicted_text,
                ground_truth_text=ground_truth_text,
                ocr_confidence=confidence,
                ocr_success=predicted_text.strip() != "",
                total_processing_time=0.0,  # Not available from CSV
                detection_time=0.0,
                ocr_time=0.0,
                memory_usage=0.0,
                end_to_end_success=predicted_text.upper().strip() == ground_truth_text.upper().strip()
            )
            
            self.add_end_to_end_result(result)
    
    def calculate_end_to_end_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive end-to-end metrics.
        
        Returns:
            Dictionary with end-to-end evaluation metrics
        """
        if not self.end_to_end_results:
            return {}
        
        # Overall success rates
        total_samples = len(self.end_to_end_results)
        detection_successes = sum(1 for r in self.end_to_end_results if r.detection_success)
        ocr_successes = sum(1 for r in self.end_to_end_results if r.ocr_success)
        end_to_end_successes = sum(1 for r in self.end_to_end_results if r.end_to_end_success)
        
        # Calculate conditional accuracies
        detection_successful_results = [r for r in self.end_to_end_results if r.detection_success]
        ocr_given_detection = sum(1 for r in detection_successful_results if r.end_to_end_success)
        
        # IoU statistics
        ious = [r.detection_iou for r in self.end_to_end_results if r.detection_iou > 0]
        
        # Text similarity statistics
        similarities = [r.text_similarity for r in self.end_to_end_results]
        
        # Processing time statistics
        total_times = [r.total_processing_time for r in self.end_to_end_results if r.total_processing_time > 0]
        detection_times = [r.detection_time for r in self.end_to_end_results if r.detection_time > 0]
        ocr_times = [r.ocr_time for r in self.end_to_end_results if r.ocr_time > 0]
        
        return {
            'overall_metrics': {
                'total_samples': total_samples,
                'detection_success_rate': detection_successes / total_samples,
                'ocr_success_rate': ocr_successes / total_samples,
                'end_to_end_success_rate': end_to_end_successes / total_samples,
                'ocr_accuracy_given_detection': ocr_given_detection / len(detection_successful_results) if detection_successful_results else 0
            },
            'detection_quality': {
                'mean_iou': np.mean(ious) if ious else 0,
                'median_iou': np.median(ious) if ious else 0,
                'iou_std': np.std(ious) if ious else 0,
                'samples_with_detection': len(ious)
            },
            'ocr_quality': {
                'mean_text_similarity': np.mean(similarities),
                'median_text_similarity': np.median(similarities),
                'similarity_std': np.std(similarities)
            },
            'performance_summary': {
                'mean_total_time': np.mean(total_times) if total_times else 0,
                'mean_detection_time': np.mean(detection_times) if detection_times else 0,
                'mean_ocr_time': np.mean(ocr_times) if ocr_times else 0,
                'samples_with_timing': len(total_times)
            }
        }
    
    def calculate_quality_score(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate an overall quality score for the system.
        
        Args:
            weights: Optional weights for different metrics
            
        Returns:
            Dictionary with quality scores
        """
        if weights is None:
            weights = {
                'detection_accuracy': 0.3,
                'ocr_accuracy': 0.4,
                'end_to_end_accuracy': 0.2,
                'processing_speed': 0.1
            }
        
        # Get component metrics
        detection_metrics = self.detection_metrics.get_detailed_report()
        ocr_metrics = self.ocr_metrics.get_detailed_report()
        end_to_end_metrics = self.calculate_end_to_end_metrics()
        
        # Extract key metrics (normalized to 0-1)
        detection_score = detection_metrics.get('detection_metrics', {}).get('f1_score', 0)
        ocr_score = ocr_metrics.get('accuracy_metrics', {}).get('exact_match_accuracy', 0)
        end_to_end_score = end_to_end_metrics.get('overall_metrics', {}).get('end_to_end_success_rate', 0)
        
        # Speed score (inverse of processing time, normalized)
        performance_summary = end_to_end_metrics.get('performance_summary', {})
        mean_time = performance_summary.get('mean_total_time', 1.0)
        speed_score = min(1.0, 1.0 / max(mean_time, 0.1))  # Cap at 1.0, avoid division by zero
        
        # Calculate weighted quality score
        quality_score = (
            weights['detection_accuracy'] * detection_score +
            weights['ocr_accuracy'] * ocr_score +
            weights['end_to_end_accuracy'] * end_to_end_score +
            weights['processing_speed'] * speed_score
        )
        
        return {
            'overall_quality_score': quality_score,
            'component_scores': {
                'detection_score': detection_score,
                'ocr_score': ocr_score,
                'end_to_end_score': end_to_end_score,
                'speed_score': speed_score
            },
            'weights_used': weights
        }
    
    def analyze_failure_modes(self) -> Dict[str, Any]:
        """
        Analyze different failure modes in the system.
        
        Returns:
            Dictionary with failure mode analysis
        """
        if not self.end_to_end_results:
            return {}
        
        # Categorize failures
        detection_failures = [r for r in self.end_to_end_results if not r.detection_success]
        ocr_failures = [r for r in self.end_to_end_results if r.detection_success and not r.ocr_success]
        text_accuracy_failures = [r for r in self.end_to_end_results 
                                 if r.detection_success and r.ocr_success and not r.end_to_end_success]
        complete_successes = [r for r in self.end_to_end_results if r.end_to_end_success]
        
        # Analyze detection failures
        detection_failure_analysis = {
            'count': len(detection_failures),
            'percentage': len(detection_failures) / len(self.end_to_end_results) * 100,
            'low_confidence_detections': len([r for r in detection_failures if r.detection_confidence < 0.3]),
            'no_detections': len([r for r in detection_failures if len(r.detected_boxes) == 0])
        }
        
        # Analyze OCR failures
        ocr_failure_analysis = {
            'count': len(ocr_failures),
            'percentage': len(ocr_failures) / len(self.end_to_end_results) * 100,
            'empty_predictions': len([r for r in ocr_failures if r.predicted_text.strip() == ""]),
            'low_confidence_ocr': len([r for r in ocr_failures if r.ocr_confidence < 0.5])
        }
        
        # Analyze text accuracy failures
        text_failure_analysis = {
            'count': len(text_accuracy_failures),
            'percentage': len(text_accuracy_failures) / len(self.end_to_end_results) * 100,
            'partial_matches': len([r for r in text_accuracy_failures if r.text_similarity > 0.5]),
            'complete_mismatches': len([r for r in text_accuracy_failures if r.text_similarity <= 0.2])
        }
        
        # Overall pipeline efficiency
        pipeline_efficiency = {
            'detection_only_success': len([r for r in self.end_to_end_results if r.detection_success]),
            'detection_and_ocr_success': len([r for r in self.end_to_end_results 
                                            if r.detection_success and r.ocr_success]),
            'complete_pipeline_success': len(complete_successes)
        }
        
        return {
            'detection_failures': detection_failure_analysis,
            'ocr_failures': ocr_failure_analysis,
            'text_accuracy_failures': text_failure_analysis,
            'pipeline_efficiency': pipeline_efficiency,
            'failure_mode_recommendations': self._generate_failure_recommendations(
                detection_failure_analysis, ocr_failure_analysis, text_failure_analysis
            )
        }
    
    def _generate_failure_recommendations(self, detection_failures: Dict, 
                                        ocr_failures: Dict, text_failures: Dict) -> List[str]:
        """
        Generate recommendations based on failure analysis.
        
        Args:
            detection_failures: Detection failure analysis
            ocr_failures: OCR failure analysis
            text_failures: Text accuracy failure analysis
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Detection-related recommendations
        if detection_failures['percentage'] > 20:
            recommendations.append("High detection failure rate - consider retraining detection model")
        
        if detection_failures.get('low_confidence_detections', 0) > detection_failures.get('no_detections', 0):
            recommendations.append("Many low-confidence detections - consider adjusting confidence threshold")
        
        # OCR-related recommendations
        if ocr_failures['percentage'] > 15:
            recommendations.append("High OCR failure rate - consider improving preprocessing pipeline")
        
        if ocr_failures.get('empty_predictions', 0) > ocr_failures['count'] * 0.5:
            recommendations.append("Many empty OCR predictions - check image quality and preprocessing")
        
        # Text accuracy recommendations
        if text_failures['percentage'] > 25:
            recommendations.append("High text accuracy failure rate - review OCR post-processing rules")
        
        if text_failures.get('partial_matches', 0) > text_failures.get('complete_mismatches', 0):
            recommendations.append("Many partial matches - improve character correction algorithms")
        
        return recommendations
    
    def compare_configurations(self, other_evaluation: 'EvaluationMetrics', 
                             config_name_1: str = "Config 1", 
                             config_name_2: str = "Config 2") -> Dict[str, Any]:
        """
        Compare two evaluation configurations.
        
        Args:
            other_evaluation: Another EvaluationMetrics instance to compare against
            config_name_1: Name for this configuration
            config_name_2: Name for the other configuration
            
        Returns:
            Dictionary with comparison results
        """
        # Get metrics for both configurations
        metrics_1 = self.calculate_end_to_end_metrics()
        metrics_2 = other_evaluation.calculate_end_to_end_metrics()
        
        quality_1 = self.calculate_quality_score()
        quality_2 = other_evaluation.calculate_quality_score()
        
        # Compare key metrics
        comparison = {
            'configurations': {
                config_name_1: {
                    'end_to_end_success_rate': metrics_1.get('overall_metrics', {}).get('end_to_end_success_rate', 0),
                    'detection_success_rate': metrics_1.get('overall_metrics', {}).get('detection_success_rate', 0),
                    'ocr_success_rate': metrics_1.get('overall_metrics', {}).get('ocr_success_rate', 0),
                    'quality_score': quality_1.get('overall_quality_score', 0),
                    'sample_count': len(self.end_to_end_results)
                },
                config_name_2: {
                    'end_to_end_success_rate': metrics_2.get('overall_metrics', {}).get('end_to_end_success_rate', 0),
                    'detection_success_rate': metrics_2.get('overall_metrics', {}).get('detection_success_rate', 0),
                    'ocr_success_rate': metrics_2.get('overall_metrics', {}).get('ocr_success_rate', 0),
                    'quality_score': quality_2.get('overall_quality_score', 0),
                    'sample_count': len(other_evaluation.end_to_end_results)
                }
            },
            'improvements': {},
            'winner': config_name_1 if quality_1.get('overall_quality_score', 0) > quality_2.get('overall_quality_score', 0) else config_name_2
        }
        
        # Calculate improvements
        config_1_metrics = comparison['configurations'][config_name_1]
        config_2_metrics = comparison['configurations'][config_name_2]
        
        for metric in ['end_to_end_success_rate', 'detection_success_rate', 'ocr_success_rate', 'quality_score']:
            diff = config_1_metrics[metric] - config_2_metrics[metric]
            comparison['improvements'][metric] = {
                'absolute_difference': diff,
                'relative_improvement': (diff / config_2_metrics[metric] * 100) if config_2_metrics[metric] > 0 else 0
            }
        
        return comparison
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Dictionary with complete evaluation results
        """
        # Get component reports
        detection_report = self.detection_metrics.get_detailed_report()
        ocr_report = self.ocr_metrics.get_detailed_report()
        performance_report = self.performance_metrics.get_detailed_report()
        
        # Get evaluation-specific metrics
        end_to_end_metrics = self.calculate_end_to_end_metrics()
        quality_score = self.calculate_quality_score()
        failure_analysis = self.analyze_failure_modes()
        
        return {
            'evaluation_summary': {
                'total_samples': len(self.end_to_end_results),
                'evaluation_timestamp': pd.Timestamp.now().isoformat(),
                'configuration': {
                    'iou_threshold': self.iou_threshold,
                    'confidence_threshold': self.confidence_threshold
                }
            },
            'end_to_end_metrics': end_to_end_metrics,
            'quality_assessment': quality_score,
            'failure_analysis': failure_analysis,
            'component_reports': {
                'detection_metrics': detection_report,
                'ocr_metrics': ocr_report,
                'performance_metrics': performance_report
            },
            'recommendations': self._generate_overall_recommendations(
                end_to_end_metrics, quality_score, failure_analysis
            )
        }
    
    def _generate_overall_recommendations(self, end_to_end_metrics: Dict, 
                                        quality_score: Dict, failure_analysis: Dict) -> List[str]:
        """
        Generate overall system recommendations.
        
        Args:
            end_to_end_metrics: End-to-end metrics
            quality_score: Quality score results
            failure_analysis: Failure mode analysis
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        overall_metrics = end_to_end_metrics.get('overall_metrics', {})
        component_scores = quality_score.get('component_scores', {})
        
        # Overall performance recommendations
        if overall_metrics.get('end_to_end_success_rate', 0) < 0.8:
            recommendations.append("Overall success rate is below 80% - comprehensive system review needed")
        
        # Component-specific recommendations
        if component_scores.get('detection_score', 0) < 0.7:
            recommendations.append("Detection performance is suboptimal - consider model retraining or parameter tuning")
        
        if component_scores.get('ocr_score', 0) < 0.6:
            recommendations.append("OCR accuracy needs improvement - review preprocessing and post-processing steps")
        
        if component_scores.get('speed_score', 0) < 0.5:
            recommendations.append("Processing speed is slow - consider optimization or hardware upgrades")
        
        # Add failure-specific recommendations
        failure_recs = failure_analysis.get('failure_mode_recommendations', [])
        recommendations.extend(failure_recs)
        
        # Quality-based recommendations
        if quality_score.get('overall_quality_score', 0) < 0.7:
            recommendations.append("Overall system quality is below acceptable threshold - prioritize improvement efforts")
        
        return recommendations
    
    def save_comprehensive_report(self, filepath: str) -> None:
        """
        Save the comprehensive evaluation report to a JSON file.
        
        Args:
            filepath: Path to save the report
        """
        report = self.get_comprehensive_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def save_results_csv(self, filepath: str) -> None:
        """
        Save detailed end-to-end results to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        data = []
        for result in self.end_to_end_results:
            data.append({
                'image_id': result.image_id,
                'image_path': result.image_path,
                'predicted_text': result.predicted_text,
                'ground_truth_text': result.ground_truth_text,
                'detection_success': result.detection_success,
                'detection_confidence': result.detection_confidence,
                'detection_iou': result.detection_iou,
                'ocr_success': result.ocr_success,
                'ocr_confidence': result.ocr_confidence,
                'text_similarity': result.text_similarity,
                'end_to_end_success': result.end_to_end_success,
                'total_processing_time': result.total_processing_time,
                'detection_time': result.detection_time,
                'ocr_time': result.ocr_time,
                'memory_usage': result.memory_usage
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def plot_pipeline_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Plot pipeline success analysis.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.end_to_end_results:
            print("No results to plot")
            return
        
        # Calculate pipeline success rates
        total = len(self.end_to_end_results)
        detection_success = sum(1 for r in self.end_to_end_results if r.detection_success)
        ocr_success = sum(1 for r in self.end_to_end_results if r.detection_success and r.ocr_success)
        end_to_end_success = sum(1 for r in self.end_to_end_results if r.end_to_end_success)
        
        # Create funnel chart
        stages = ['Total Images', 'Detection Success', 'OCR Success', 'End-to-End Success']
        counts = [total, detection_success, ocr_success, end_to_end_success]
        percentages = [100, (detection_success/total)*100, (ocr_success/total)*100, (end_to_end_success/total)*100]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Funnel chart (absolute counts)
        ax1.barh(stages, counts, color=['blue', 'green', 'orange', 'red'])
        ax1.set_xlabel('Number of Images')
        ax1.set_title('Processing Pipeline Success Funnel')
        for i, count in enumerate(counts):
            ax1.text(count + total*0.01, i, str(count), va='center')
        
        # Percentage chart
        ax2.barh(stages, percentages, color=['blue', 'green', 'orange', 'red'])
        ax2.set_xlabel('Percentage (%)')
        ax2.set_title('Processing Pipeline Success Rates')
        for i, pct in enumerate(percentages):
            ax2.text(pct + 1, i, f'{pct:.1f}%', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_quality_comparison(self, other_evaluations: List[Tuple['EvaluationMetrics', str]], 
                               save_path: Optional[str] = None) -> None:
        """
        Plot quality comparison across multiple configurations.
        
        Args:
            other_evaluations: List of (EvaluationMetrics, name) tuples
            save_path: Optional path to save the plot
        """
        # Include this evaluation
        all_evaluations = [(self, "Current")] + other_evaluations
        
        names = [name for _, name in all_evaluations]
        quality_scores = []
        component_scores = []
        
        for evaluation, _ in all_evaluations:
            quality = evaluation.calculate_quality_score()
            quality_scores.append(quality.get('overall_quality_score', 0))
            component_scores.append(quality.get('component_scores', {}))
        
        # Extract component scores
        detection_scores = [scores.get('detection_score', 0) for scores in component_scores]
        ocr_scores = [scores.get('ocr_score', 0) for scores in component_scores]
        end_to_end_scores = [scores.get('end_to_end_score', 0) for scores in component_scores]
        speed_scores = [scores.get('speed_score', 0) for scores in component_scores]
        
        # Create comparison plot
        x = np.arange(len(names))
        width = 0.15
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Overall quality scores
        ax1.bar(names, quality_scores, color='purple', alpha=0.7)
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Overall Quality Score Comparison')
        ax1.set_ylim([0, 1])
        for i, score in enumerate(quality_scores):
            ax1.text(i, score + 0.02, f'{score:.3f}', ha='center')
        
        # Component scores
        ax2.bar(x - 1.5*width, detection_scores, width, label='Detection', alpha=0.8)
        ax2.bar(x - 0.5*width, ocr_scores, width, label='OCR', alpha=0.8)
        ax2.bar(x + 0.5*width, end_to_end_scores, width, label='End-to-End', alpha=0.8)
        ax2.bar(x + 1.5*width, speed_scores, width, label='Speed', alpha=0.8)
        
        ax2.set_ylabel('Component Score')
        ax2.set_title('Component Score Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.legend()
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_evaluation_from_csv(csv_path: str, images_dir: str = "") -> EvaluationMetrics:
    """
    Create an EvaluationMetrics instance from a CSV file.
    
    Args:
        csv_path: Path to the CSV file with results
        images_dir: Optional directory containing images
        
    Returns:
        EvaluationMetrics instance
    """
    evaluation = EvaluationMetrics()
    evaluation.evaluate_from_results_csv(csv_path, images_dir)
    return evaluation

