"""
OCR metrics for license plate text recognition evaluation.

This module provides comprehensive metrics for evaluating the performance
of OCR systems on license plate text including:
- Character-level accuracy
- String-level accuracy
- Edit distance metrics (Levenshtein)
- Confidence correlation analysis
- Pattern matching accuracy
- Text similarity measures
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from difflib import SequenceMatcher
import Levenshtein


@dataclass
class OCRResult:
    """Represents an OCR result with confidence and metadata."""
    image_id: str
    predicted_text: str
    ground_truth_text: str
    confidence: float = 0.0
    processing_method: str = ""
    preprocessing_applied: str = ""
    
    @property
    def is_exact_match(self) -> bool:
        """Check if prediction exactly matches ground truth."""
        return self.predicted_text.upper().strip() == self.ground_truth_text.upper().strip()
    
    @property
    def character_accuracy(self) -> float:
        """Calculate character-level accuracy."""
        if not self.ground_truth_text:
            return 1.0 if not self.predicted_text else 0.0
        
        pred = self.predicted_text.upper().strip()
        gt = self.ground_truth_text.upper().strip()
        
        if not gt:
            return 1.0 if not pred else 0.0
        
        # Calculate character-level accuracy
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            return 1.0
        
        correct_chars = sum(1 for i, (p, g) in enumerate(zip(pred, gt)) if p == g)
        
        # Handle length differences
        if len(pred) != len(gt):
            correct_chars -= abs(len(pred) - len(gt))
        
        return max(0, correct_chars) / max_len
    
    @property
    def similarity_score(self) -> float:
        """Calculate text similarity using SequenceMatcher."""
        pred = self.predicted_text.upper().strip()
        gt = self.ground_truth_text.upper().strip()
        return SequenceMatcher(None, pred, gt).ratio()
    
    @property
    def edit_distance(self) -> int:
        """Calculate Levenshtein edit distance."""
        pred = self.predicted_text.upper().strip()
        gt = self.ground_truth_text.upper().strip()
        return Levenshtein.distance(pred, gt)
    
    @property
    def normalized_edit_distance(self) -> float:
        """Calculate normalized edit distance."""
        pred = self.predicted_text.upper().strip()
        gt = self.ground_truth_text.upper().strip()
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            return 0.0
        return self.edit_distance / max_len


class OCRMetrics:
    """
    Comprehensive OCR metrics calculator for license plate text recognition.
    """
    
    def __init__(self):
        """Initialize OCR metrics calculator."""
        self.results: List[OCRResult] = []
        self.character_confusion_matrix = defaultdict(Counter)
        
    def add_result(self, image_id: str, predicted_text: str, ground_truth_text: str,
                   confidence: float = 0.0, processing_method: str = "",
                   preprocessing_applied: str = "") -> None:
        """
        Add an OCR result for evaluation.
        
        Args:
            image_id: Unique identifier for the image
            predicted_text: OCR predicted text
            ground_truth_text: Ground truth text
            confidence: OCR confidence score
            processing_method: Method used for OCR processing
            preprocessing_applied: Preprocessing steps applied
        """
        result = OCRResult(
            image_id=image_id,
            predicted_text=predicted_text,
            ground_truth_text=ground_truth_text,
            confidence=confidence,
            processing_method=processing_method,
            preprocessing_applied=preprocessing_applied
        )
        self.results.append(result)
        self._update_character_confusion_matrix(result)
    
    def _update_character_confusion_matrix(self, result: OCRResult) -> None:
        """Update character-level confusion matrix."""
        pred = result.predicted_text.upper().strip()
        gt = result.ground_truth_text.upper().strip()
        
        # Align strings for character comparison
        for i in range(min(len(pred), len(gt))):
            predicted_char = pred[i]
            actual_char = gt[i]
            self.character_confusion_matrix[actual_char][predicted_char] += 1
        
        # Handle length differences
        if len(pred) > len(gt):
            # Extra characters in prediction
            for i in range(len(gt), len(pred)):
                self.character_confusion_matrix['<MISSING>'][pred[i]] += 1
        elif len(gt) > len(pred):
            # Missing characters in prediction
            for i in range(len(pred), len(gt)):
                self.character_confusion_matrix[gt[i]]['<MISSING>'] += 1
    
    def calculate_accuracy_metrics(self) -> Dict[str, float]:
        """
        Calculate various accuracy metrics.
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.results:
            return {
                'exact_match_accuracy': 0.0,
                'character_accuracy': 0.0,
                'average_similarity': 0.0,
                'average_edit_distance': 0.0,
                'normalized_edit_distance': 0.0
            }
        
        exact_matches = sum(1 for result in self.results if result.is_exact_match)
        character_accuracies = [result.character_accuracy for result in self.results]
        similarities = [result.similarity_score for result in self.results]
        edit_distances = [result.edit_distance for result in self.results]
        normalized_edit_distances = [result.normalized_edit_distance for result in self.results]
        
        return {
            'exact_match_accuracy': exact_matches / len(self.results),
            'character_accuracy': np.mean(character_accuracies),
            'average_similarity': np.mean(similarities),
            'average_edit_distance': np.mean(edit_distances),
            'normalized_edit_distance': np.mean(normalized_edit_distances),
            'total_samples': len(self.results)
        }
    
    def analyze_confidence_correlation(self) -> Dict[str, float]:
        """
        Analyze correlation between OCR confidence and actual accuracy.
        
        Returns:
            Dictionary with confidence correlation metrics
        """
        if not self.results:
            return {'confidence_accuracy_correlation': 0.0}
        
        # Filter results with confidence scores
        results_with_confidence = [r for r in self.results if r.confidence > 0]
        
        if len(results_with_confidence) < 2:
            return {'confidence_accuracy_correlation': 0.0}
        
        confidences = [r.confidence for r in results_with_confidence]
        accuracies = [r.character_accuracy for r in results_with_confidence]
        exact_matches = [1.0 if r.is_exact_match else 0.0 for r in results_with_confidence]
        similarities = [r.similarity_score for r in results_with_confidence]
        
        # Calculate correlations
        confidence_char_corr = np.corrcoef(confidences, accuracies)[0, 1] if len(confidences) > 1 else 0.0
        confidence_exact_corr = np.corrcoef(confidences, exact_matches)[0, 1] if len(confidences) > 1 else 0.0
        confidence_sim_corr = np.corrcoef(confidences, similarities)[0, 1] if len(confidences) > 1 else 0.0
        
        return {
            'confidence_accuracy_correlation': confidence_char_corr,
            'confidence_exact_match_correlation': confidence_exact_corr,
            'confidence_similarity_correlation': confidence_sim_corr,
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'samples_with_confidence': len(results_with_confidence)
        }
    
    def analyze_pattern_accuracy(self) -> Dict[str, any]:
        """
        Analyze accuracy for different license plate patterns.
        
        Returns:
            Dictionary with pattern-specific accuracy metrics
        """
        pattern_stats = defaultdict(list)
        
        # Define common license plate patterns
        patterns = {
            '3L4N': r'^[A-Z]{3}[0-9]{4}$',  # 3 letters + 4 numbers
            '2L5N': r'^[A-Z]{2}[0-9]{5}$',  # 2 letters + 5 numbers
            '4L3N': r'^[A-Z]{4}[0-9]{3}$',  # 4 letters + 3 numbers
            '3L3N': r'^[A-Z]{3}[0-9]{3}$',  # 3 letters + 3 numbers
            'MIXED': r'^[A-Z0-9]+$',        # Mixed alphanumeric
            'VANITY': r'^[A-Z]+$'           # All letters (vanity)
        }
        
        for result in self.results:
            gt_text = result.ground_truth_text.upper().strip().replace(' ', '')
            
            # Categorize by pattern
            pattern_matched = 'OTHER'
            for pattern_name, pattern_regex in patterns.items():
                if re.match(pattern_regex, gt_text):
                    pattern_matched = pattern_name
                    break
            
            pattern_stats[pattern_matched].append(result)
        
        # Calculate metrics for each pattern
        pattern_metrics = {}
        for pattern, results_list in pattern_stats.items():
            if results_list:
                exact_matches = sum(1 for r in results_list if r.is_exact_match)
                char_accuracies = [r.character_accuracy for r in results_list]
                similarities = [r.similarity_score for r in results_list]
                
                pattern_metrics[pattern] = {
                    'count': len(results_list),
                    'exact_match_accuracy': exact_matches / len(results_list),
                    'character_accuracy': np.mean(char_accuracies),
                    'average_similarity': np.mean(similarities),
                    'accuracy_std': np.std(char_accuracies)
                }
        
        return pattern_metrics
    
    def analyze_character_errors(self) -> Dict[str, any]:
        """
        Analyze character-level OCR errors.
        
        Returns:
            Dictionary with character error analysis
        """
        # Most commonly confused characters
        confusion_pairs = []
        for actual_char, predicted_counts in self.character_confusion_matrix.items():
            for predicted_char, count in predicted_counts.items():
                if actual_char != predicted_char:
                    confusion_pairs.append({
                        'actual': actual_char,
                        'predicted': predicted_char,
                        'count': count,
                        'error_rate': count / sum(predicted_counts.values())
                    })
        
        # Sort by frequency
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        # Character accuracy per character
        char_accuracies = {}
        for actual_char, predicted_counts in self.character_confusion_matrix.items():
            total_count = sum(predicted_counts.values())
            correct_count = predicted_counts.get(actual_char, 0)
            char_accuracies[actual_char] = correct_count / total_count if total_count > 0 else 0
        
        return {
            'most_confused_pairs': confusion_pairs[:20],  # Top 20 confusions
            'character_accuracies': char_accuracies,
            'worst_performing_characters': sorted(char_accuracies.items(), key=lambda x: x[1])[:10],
            'best_performing_characters': sorted(char_accuracies.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def analyze_length_impact(self) -> Dict[str, any]:
        """
        Analyze how text length affects OCR accuracy.
        
        Returns:
            Dictionary with length impact analysis
        """
        length_stats = defaultdict(list)
        
        for result in self.results:
            gt_length = len(result.ground_truth_text.strip())
            length_stats[gt_length].append(result)
        
        length_metrics = {}
        for length, results_list in length_stats.items():
            if results_list:
                exact_matches = sum(1 for r in results_list if r.is_exact_match)
                char_accuracies = [r.character_accuracy for r in results_list]
                
                length_metrics[length] = {
                    'count': len(results_list),
                    'exact_match_accuracy': exact_matches / len(results_list),
                    'character_accuracy': np.mean(char_accuracies),
                    'accuracy_std': np.std(char_accuracies)
                }
        
        return length_metrics
    
    def analyze_preprocessing_impact(self) -> Dict[str, any]:
        """
        Analyze impact of different preprocessing methods on accuracy.
        
        Returns:
            Dictionary with preprocessing impact analysis
        """
        preprocessing_stats = defaultdict(list)
        
        for result in self.results:
            method = result.preprocessing_applied if result.preprocessing_applied else 'none'
            preprocessing_stats[method].append(result)
        
        preprocessing_metrics = {}
        for method, results_list in preprocessing_stats.items():
            if results_list:
                exact_matches = sum(1 for r in results_list if r.is_exact_match)
                char_accuracies = [r.character_accuracy for r in results_list]
                
                preprocessing_metrics[method] = {
                    'count': len(results_list),
                    'exact_match_accuracy': exact_matches / len(results_list),
                    'character_accuracy': np.mean(char_accuracies),
                    'accuracy_std': np.std(char_accuracies)
                }
        
        return preprocessing_metrics
    
    def get_detailed_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive OCR metrics report.
        
        Returns:
            Dictionary with all calculated metrics
        """
        accuracy_metrics = self.calculate_accuracy_metrics()
        confidence_correlation = self.analyze_confidence_correlation()
        pattern_accuracy = self.analyze_pattern_accuracy()
        character_errors = self.analyze_character_errors()
        length_impact = self.analyze_length_impact()
        preprocessing_impact = self.analyze_preprocessing_impact()
        
        return {
            'summary': {
                'total_samples': len(self.results),
                'unique_images': len(set(r.image_id for r in self.results)),
            },
            'accuracy_metrics': accuracy_metrics,
            'confidence_analysis': confidence_correlation,
            'pattern_analysis': pattern_accuracy,
            'character_analysis': character_errors,
            'length_analysis': length_impact,
            'preprocessing_analysis': preprocessing_impact,
            'per_sample_results': self._get_per_sample_metrics()
        }
    
    def _get_per_sample_metrics(self) -> List[Dict[str, any]]:
        """
        Get detailed metrics for each sample.
        
        Returns:
            List of per-sample metric dictionaries
        """
        per_sample_results = []
        
        for result in self.results:
            per_sample_results.append({
                'image_id': result.image_id,
                'predicted_text': result.predicted_text,
                'ground_truth_text': result.ground_truth_text,
                'confidence': result.confidence,
                'processing_method': result.processing_method,
                'preprocessing_applied': result.preprocessing_applied,
                'is_exact_match': result.is_exact_match,
                'character_accuracy': result.character_accuracy,
                'similarity_score': result.similarity_score,
                'edit_distance': result.edit_distance,
                'normalized_edit_distance': result.normalized_edit_distance
            })
        
        return per_sample_results
    
    def save_report(self, filepath: str) -> None:
        """
        Save the detailed metrics report to a JSON file.
        
        Args:
            filepath: Path to save the report
        """
        report = self.get_detailed_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def save_results_csv(self, filepath: str) -> None:
        """
        Save detailed results to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        data = []
        for result in self.results:
            data.append({
                'image_id': result.image_id,
                'predicted_text': result.predicted_text,
                'ground_truth_text': result.ground_truth_text,
                'confidence': result.confidence,
                'processing_method': result.processing_method,
                'preprocessing_applied': result.preprocessing_applied,
                'is_exact_match': result.is_exact_match,
                'character_accuracy': result.character_accuracy,
                'similarity_score': result.similarity_score,
                'edit_distance': result.edit_distance,
                'normalized_edit_distance': result.normalized_edit_distance
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def plot_accuracy_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot distribution of accuracy scores.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        char_accuracies = [r.character_accuracy for r in self.results]
        similarities = [r.similarity_score for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Character accuracy distribution
        ax1.hist(char_accuracies, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Character Accuracy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Character Accuracy')
        ax1.axvline(np.mean(char_accuracies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(char_accuracies):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Similarity score distribution
        ax2.hist(similarities, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Similarity Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Similarity Scores')
        ax2.axvline(np.mean(similarities), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(similarities):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_accuracy_correlation(self, save_path: Optional[str] = None) -> None:
        """
        Plot correlation between confidence and accuracy.
        
        Args:
            save_path: Optional path to save the plot
        """
        results_with_confidence = [r for r in self.results if r.confidence > 0]
        
        if len(results_with_confidence) < 2:
            print("Insufficient data with confidence scores for plotting")
            return
        
        confidences = [r.confidence for r in results_with_confidence]
        accuracies = [r.character_accuracy for r in results_with_confidence]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(confidences, accuracies, alpha=0.6, s=50)
        
        # Add trend line
        z = np.polyfit(confidences, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(confidences, p(confidences), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        correlation = np.corrcoef(confidences, accuracies)[0, 1]
        
        plt.xlabel('OCR Confidence')
        plt.ylabel('Character Accuracy')
        plt.title(f'Confidence vs Accuracy Correlation (r={correlation:.3f})')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_character_confusion_matrix(self, save_path: Optional[str] = None, top_n: int = 20) -> None:
        """
        Plot character confusion matrix for most common characters.
        
        Args:
            save_path: Optional path to save the plot
            top_n: Number of most common characters to include
        """
        if not self.character_confusion_matrix:
            print("No character confusion data available")
            return
        
        # Get most common characters
        char_counts = defaultdict(int)
        for actual_char, predicted_counts in self.character_confusion_matrix.items():
            char_counts[actual_char] += sum(predicted_counts.values())
        
        most_common_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        common_chars = [char for char, _ in most_common_chars if char != '<MISSING>']
        
        # Build confusion matrix
        matrix = np.zeros((len(common_chars), len(common_chars)))
        for i, actual_char in enumerate(common_chars):
            for j, predicted_char in enumerate(common_chars):
                matrix[i, j] = self.character_confusion_matrix[actual_char][predicted_char]
        
        # Normalize by row (actual character)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums!=0)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(normalized_matrix, 
                   xticklabels=common_chars, 
                   yticklabels=common_chars,
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   cbar_kws={'label': 'Normalized Frequency'})
        plt.xlabel('Predicted Character')
        plt.ylabel('Actual Character')
        plt.title('Character Confusion Matrix (Normalized)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pattern_accuracy(self, save_path: Optional[str] = None) -> None:
        """
        Plot accuracy by license plate pattern.
        
        Args:
            save_path: Optional path to save the plot
        """
        pattern_metrics = self.analyze_pattern_accuracy()
        
        if not pattern_metrics:
            print("No pattern data available")
            return
        
        patterns = list(pattern_metrics.keys())
        exact_accuracies = [pattern_metrics[p]['exact_match_accuracy'] for p in patterns]
        char_accuracies = [pattern_metrics[p]['character_accuracy'] for p in patterns]
        counts = [pattern_metrics[p]['count'] for p in patterns]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy by pattern
        x_pos = np.arange(len(patterns))
        width = 0.35
        
        ax1.bar(x_pos - width/2, exact_accuracies, width, label='Exact Match', alpha=0.8)
        ax1.bar(x_pos + width/2, char_accuracies, width, label='Character Accuracy', alpha=0.8)
        
        ax1.set_xlabel('License Plate Pattern')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by License Plate Pattern')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(patterns, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sample count by pattern
        ax2.bar(patterns, counts, alpha=0.8, color='orange')
        ax2.set_xlabel('License Plate Pattern')
        ax2.set_ylabel('Sample Count')
        ax2.set_title('Sample Distribution by Pattern')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_ocr_results_from_csv(csv_path: str, 
                             image_col: str = 'image', 
                             predicted_col: str = 'predicted',
                             ground_truth_col: str = 'expected',
                             confidence_col: Optional[str] = 'confidence') -> OCRMetrics:
    """
    Load OCR results from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        image_col: Column name for image identifiers
        predicted_col: Column name for predicted text
        ground_truth_col: Column name for ground truth text
        confidence_col: Column name for confidence scores (optional)
        
    Returns:
        OCRMetrics object with loaded results
    """
    df = pd.read_csv(csv_path)
    metrics = OCRMetrics()
    
    for _, row in df.iterrows():
        confidence = row.get(confidence_col, 0.0) if confidence_col else 0.0
        
        metrics.add_result(
            image_id=str(row[image_col]),
            predicted_text=str(row[predicted_col]) if pd.notna(row[predicted_col]) else "",
            ground_truth_text=str(row[ground_truth_col]) if pd.notna(row[ground_truth_col]) else "",
            confidence=float(confidence) if pd.notna(confidence) else 0.0
        )
    
    return metrics

