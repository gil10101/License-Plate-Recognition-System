"""
Detection metrics for license plate object detection evaluation.

This module provides comprehensive metrics for evaluating the performance
of license plate detection models including:
- Precision, Recall, F1-score
- Intersection over Union (IoU)
- Average Precision (AP) and mean Average Precision (mAP)
- Confusion matrices
- Detection confidence analysis
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and optional confidence."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = 0
    
    @property
    def area(self) -> float:
        """Calculate the area of the bounding box."""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    """Represents a detection result with metrics."""
    image_id: str
    predicted_boxes: List[BoundingBox]
    ground_truth_boxes: List[BoundingBox]
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.5


class DetectionMetrics:
    """
    Comprehensive detection metrics calculator for license plate detection.
    """
    
    def __init__(self, iou_threshold: float = 0.5, confidence_threshold: float = 0.5):
        """
        Initialize detection metrics calculator.
        
        Args:
            iou_threshold: IoU threshold for considering a detection as correct
            confidence_threshold: Confidence threshold for filtering detections
        """
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.results: List[DetectionResult] = []
        
    def add_result(self, image_id: str, predicted_boxes: List[BoundingBox], 
                   ground_truth_boxes: List[BoundingBox]) -> None:
        """
        Add a detection result for evaluation.
        
        Args:
            image_id: Unique identifier for the image
            predicted_boxes: List of predicted bounding boxes
            ground_truth_boxes: List of ground truth bounding boxes
        """
        # Filter predictions by confidence threshold
        filtered_predictions = [
            box for box in predicted_boxes 
            if box.confidence >= self.confidence_threshold
        ]
        
        result = DetectionResult(
            image_id=image_id,
            predicted_boxes=filtered_predictions,
            ground_truth_boxes=ground_truth_boxes,
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold
        )
        self.results.append(result)
    
    def calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box
            box2: Second bounding box
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        union_area = box1.area + box2.area - intersection_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def match_detections(self, predicted_boxes: List[BoundingBox], 
                        ground_truth_boxes: List[BoundingBox]) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """
        Match predicted boxes with ground truth boxes based on IoU.
        
        Args:
            predicted_boxes: List of predicted bounding boxes
            ground_truth_boxes: List of ground truth bounding boxes
            
        Returns:
            Tuple of (matches, unmatched_predictions, unmatched_ground_truths)
        """
        matches = []
        matched_gt = set()
        matched_pred = set()
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predicted_boxes), len(ground_truth_boxes)))
        for i, pred_box in enumerate(predicted_boxes):
            for j, gt_box in enumerate(ground_truth_boxes):
                iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)
        
        # Find best matches
        while True:
            # Find the maximum IoU
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            
            # Stop if no valid matches left
            if max_iou < self.iou_threshold:
                break
                
            pred_idx, gt_idx = max_iou_idx
            matches.append((pred_idx, gt_idx, max_iou))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            
            # Remove matched boxes from consideration
            iou_matrix[pred_idx, :] = 0
            iou_matrix[:, gt_idx] = 0
        
        # Find unmatched predictions and ground truths
        unmatched_predictions = [i for i in range(len(predicted_boxes)) if i not in matched_pred]
        unmatched_ground_truths = [i for i in range(len(ground_truth_boxes)) if i not in matched_gt]
        
        return matches, unmatched_predictions, unmatched_ground_truths
    
    def calculate_precision_recall(self) -> Tuple[float, float, float]:
        """
        Calculate overall precision, recall, and F1-score.
        
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        total_tp = 0  # True positives
        total_fp = 0  # False positives
        total_fn = 0  # False negatives
        
        for result in self.results:
            matches, unmatched_pred, unmatched_gt = self.match_detections(
                result.predicted_boxes, result.ground_truth_boxes
            )
            
            total_tp += len(matches)
            total_fp += len(unmatched_pred)
            total_fn += len(unmatched_gt)
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1_score
    
    def calculate_average_precision(self, confidence_thresholds: Optional[List[float]] = None) -> float:
        """
        Calculate Average Precision (AP) using precision-recall curve.
        
        Args:
            confidence_thresholds: List of confidence thresholds to evaluate
            
        Returns:
            Average Precision value
        """
        if confidence_thresholds is None:
            confidence_thresholds = np.arange(0.0, 1.01, 0.01)
        
        precisions = []
        recalls = []
        
        original_threshold = self.confidence_threshold
        
        for threshold in confidence_thresholds:
            self.confidence_threshold = threshold
            precision, recall, _ = self.calculate_precision_recall()
            precisions.append(precision)
            recalls.append(recall)
        
        # Restore original threshold
        self.confidence_threshold = original_threshold
        
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        sorted_recalls = np.array(recalls)[sorted_indices]
        sorted_precisions = np.array(precisions)[sorted_indices]
        
        # Calculate AP using trapezoidal rule
        ap = 0.0
        for i in range(1, len(sorted_recalls)):
            ap += (sorted_recalls[i] - sorted_recalls[i-1]) * sorted_precisions[i]
        
        return ap
    
    def calculate_map(self, iou_thresholds: Optional[List[float]] = None) -> float:
        """
        Calculate mean Average Precision (mAP) across different IoU thresholds.
        
        Args:
            iou_thresholds: List of IoU thresholds to evaluate
            
        Returns:
            mean Average Precision value
        """
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        aps = []
        original_iou_threshold = self.iou_threshold
        
        for iou_threshold in iou_thresholds:
            self.iou_threshold = iou_threshold
            ap = self.calculate_average_precision()
            aps.append(ap)
        
        # Restore original IoU threshold
        self.iou_threshold = original_iou_threshold
        
        return np.mean(aps)
    
    def analyze_confidence_distribution(self) -> Dict[str, any]:
        """
        Analyze the distribution of detection confidences.
        
        Returns:
            Dictionary with confidence statistics
        """
        all_confidences = []
        correct_confidences = []
        incorrect_confidences = []
        
        for result in self.results:
            matches, unmatched_pred, _ = self.match_detections(
                result.predicted_boxes, result.ground_truth_boxes
            )
            
            # Collect confidences for matched (correct) detections
            for pred_idx, _, _ in matches:
                confidence = result.predicted_boxes[pred_idx].confidence
                all_confidences.append(confidence)
                correct_confidences.append(confidence)
            
            # Collect confidences for unmatched (incorrect) detections
            for pred_idx in unmatched_pred:
                confidence = result.predicted_boxes[pred_idx].confidence
                all_confidences.append(confidence)
                incorrect_confidences.append(confidence)
        
        return {
            'all_confidences': all_confidences,
            'correct_confidences': correct_confidences,
            'incorrect_confidences': incorrect_confidences,
            'mean_confidence': np.mean(all_confidences) if all_confidences else 0,
            'mean_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0,
            'mean_incorrect_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0,
            'confidence_threshold_optimal': self._find_optimal_confidence_threshold()
        }
    
    def _find_optimal_confidence_threshold(self) -> float:
        """
        Find the optimal confidence threshold that maximizes F1-score.
        
        Returns:
            Optimal confidence threshold
        """
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        original_threshold = self.confidence_threshold
        
        for threshold in thresholds:
            self.confidence_threshold = threshold
            _, _, f1 = self.calculate_precision_recall()
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Restore original threshold
        self.confidence_threshold = original_threshold
        return best_threshold
    
    def generate_confusion_matrix(self) -> Dict[str, int]:
        """
        Generate confusion matrix for detection results.
        
        Returns:
            Dictionary with TP, FP, FN counts
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for result in self.results:
            matches, unmatched_pred, unmatched_gt = self.match_detections(
                result.predicted_boxes, result.ground_truth_boxes
            )
            
            total_tp += len(matches)
            total_fp += len(unmatched_pred)
            total_fn += len(unmatched_gt)
        
        return {
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'true_negatives': 0  # Not applicable for object detection
        }
    
    def get_detailed_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive detection metrics report.
        
        Returns:
            Dictionary with all calculated metrics
        """
        precision, recall, f1_score = self.calculate_precision_recall()
        ap = self.calculate_average_precision()
        map_score = self.calculate_map()
        confidence_analysis = self.analyze_confidence_distribution()
        confusion_matrix = self.generate_confusion_matrix()
        
        return {
            'summary': {
                'total_images': len(self.results),
                'iou_threshold': self.iou_threshold,
                'confidence_threshold': self.confidence_threshold,
            },
            'detection_metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'average_precision': ap,
                'mean_average_precision': map_score,
            },
            'confidence_analysis': confidence_analysis,
            'confusion_matrix': confusion_matrix,
            'per_image_results': self._get_per_image_metrics()
        }
    
    def _get_per_image_metrics(self) -> List[Dict[str, any]]:
        """
        Calculate metrics for each image individually.
        
        Returns:
            List of per-image metric dictionaries
        """
        per_image_results = []
        
        for result in self.results:
            matches, unmatched_pred, unmatched_gt = self.match_detections(
                result.predicted_boxes, result.ground_truth_boxes
            )
            
            tp = len(matches)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate average IoU for matched detections
            avg_iou = np.mean([iou for _, _, iou in matches]) if matches else 0
            
            per_image_results.append({
                'image_id': result.image_id,
                'num_predictions': len(result.predicted_boxes),
                'num_ground_truths': len(result.ground_truth_boxes),
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'average_iou': avg_iou
            })
        
        return per_image_results
    
    def save_report(self, filepath: str) -> None:
        """
        Save the detailed metrics report to a JSON file.
        
        Args:
            filepath: Path to save the report
        """
        report = self.get_detailed_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            save_path: Optional path to save the plot
        """
        confidence_thresholds = np.arange(0.0, 1.01, 0.02)
        precisions = []
        recalls = []
        
        original_threshold = self.confidence_threshold
        
        for threshold in confidence_thresholds:
            self.confidence_threshold = threshold
            precision, recall, _ = self.calculate_precision_recall()
            precisions.append(precision)
            recalls.append(recall)
        
        # Restore original threshold
        self.confidence_threshold = original_threshold
        
        plt.figure(figsize=(10, 8))
        plt.plot(recalls, precisions, 'b-', linewidth=2, label=f'PR Curve (AP={self.calculate_average_precision():.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for License Plate Detection')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_histogram(self, save_path: Optional[str] = None) -> None:
        """
        Plot histogram of detection confidences.
        
        Args:
            save_path: Optional path to save the plot
        """
        confidence_analysis = self.analyze_confidence_distribution()
        
        plt.figure(figsize=(12, 8))
        
        # Plot separate histograms for correct and incorrect detections
        if confidence_analysis['correct_confidences']:
            plt.hist(confidence_analysis['correct_confidences'], bins=30, alpha=0.7, 
                    label=f'Correct Detections (n={len(confidence_analysis["correct_confidences"])})', 
                    color='green', density=True)
        
        if confidence_analysis['incorrect_confidences']:
            plt.hist(confidence_analysis['incorrect_confidences'], bins=30, alpha=0.7, 
                    label=f'Incorrect Detections (n={len(confidence_analysis["incorrect_confidences"])})', 
                    color='red', density=True)
        
        plt.xlabel('Detection Confidence')
        plt.ylabel('Density')
        plt.title('Distribution of Detection Confidences')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add vertical line for optimal threshold
        optimal_threshold = confidence_analysis['confidence_threshold_optimal']
        plt.axvline(x=optimal_threshold, color='black', linestyle='--', 
                   label=f'Optimal Threshold: {optimal_threshold:.3f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_coco_annotations(annotation_file: str) -> Dict[str, List[BoundingBox]]:
    """
    Load bounding box annotations from COCO format JSON file.
    
    Args:
        annotation_file: Path to COCO format annotation file
        
    Returns:
        Dictionary mapping image IDs to lists of bounding boxes
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Create mapping from image ID to image info
    images = {img['id']: img for img in data['images']}
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id in images:
            image_filename = images[image_id]['file_name']
            bbox = ann['bbox']  # [x, y, width, height] in COCO format
            
            # Convert to [x1, y1, x2, y2] format
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            bounding_box = BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                class_id=ann.get('category_id', 0)
            )
            annotations_by_image[image_filename].append(bounding_box)
    
    return dict(annotations_by_image)


def load_yolo_annotations(labels_dir: str, images_dir: str) -> Dict[str, List[BoundingBox]]:
    """
    Load bounding box annotations from YOLO format text files.
    
    Args:
        labels_dir: Directory containing YOLO format label files
        images_dir: Directory containing corresponding image files
        
    Returns:
        Dictionary mapping image filenames to lists of bounding boxes
    """
    import glob
    import os
    
    annotations_by_image = {}
    
    # Get all label files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    for label_file in label_files:
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        
        # Find corresponding image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_path = None
        for ext in image_extensions:
            potential_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            continue
        
        # Read image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        image_filename = os.path.basename(image_path)
        
        # Read YOLO annotations
        bboxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert from normalized YOLO format to absolute coordinates
                    x1 = (x_center - width / 2) * img_width
                    y1 = (y_center - height / 2) * img_height
                    x2 = (x_center + width / 2) * img_width
                    y2 = (y_center + height / 2) * img_height
                    
                    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id)
                    bboxes.append(bbox)
        
        if bboxes:
            annotations_by_image[image_filename] = bboxes
    
    return annotations_by_image
