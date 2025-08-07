"""
Performance metrics for license plate recognition system benchmarking.

This module provides comprehensive performance metrics including:
- Processing time analysis
- Memory usage monitoring
- Throughput measurement
- Resource utilization tracking
- Bottleneck identification
- System efficiency metrics
"""

import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
import pandas as pd
from contextlib import contextmanager
import tracemalloc
import gc


@dataclass
class PerformanceMetric:
    """Represents a single performance measurement."""
    operation: str
    start_time: float
    end_time: float
    memory_before: float = 0.0
    memory_after: float = 0.0
    cpu_percent: float = 0.0
    image_size: tuple = (0, 0)
    image_path: str = ""
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get the duration of the operation in seconds."""
        return self.end_time - self.start_time
    
    @property
    def memory_delta(self) -> float:
        """Get the memory usage change in MB."""
        return self.memory_after - self.memory_before
    
    @property
    def processing_rate(self) -> float:
        """Get processing rate in pixels per second."""
        if self.duration > 0 and self.image_size[0] > 0 and self.image_size[1] > 0:
            total_pixels = self.image_size[0] * self.image_size[1]
            return total_pixels / self.duration
        return 0.0


class PerformanceProfiler:
    """Context manager for profiling performance of operations."""
    
    def __init__(self, operation: str, metrics_collector: 'PerformanceMetrics',
                 image_path: str = "", image_size: tuple = (0, 0),
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize performance profiler.
        
        Args:
            operation: Name of the operation being profiled
            metrics_collector: PerformanceMetrics instance to collect data
            image_path: Path to the image being processed
            image_size: Size of the image (width, height)
            metadata: Additional metadata to store
        """
        self.operation = operation
        self.metrics_collector = metrics_collector
        self.image_path = image_path
        self.image_size = image_size
        self.metadata = metadata or {}
        self.start_time = 0.0
        self.memory_before = 0.0
        self.cpu_before = 0.0
        
    def __enter__(self):
        """Start profiling."""
        self.start_time = time.time()
        self.memory_before = self._get_memory_usage()
        self.cpu_before = psutil.cpu_percent(interval=None)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and record metrics."""
        end_time = time.time()
        memory_after = self._get_memory_usage()
        cpu_after = psutil.cpu_percent(interval=None)
        
        success = exc_type is None
        error_message = str(exc_val) if exc_val else ""
        
        metric = PerformanceMetric(
            operation=self.operation,
            start_time=self.start_time,
            end_time=end_time,
            memory_before=self.memory_before,
            memory_after=memory_after,
            cpu_percent=(self.cpu_before + cpu_after) / 2,
            image_size=self.image_size,
            image_path=self.image_path,
            success=success,
            error_message=error_message,
            metadata=self.metadata
        )
        
        self.metrics_collector.add_metric(metric)
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class PerformanceMetrics:
    """
    Comprehensive performance metrics collector and analyzer.
    """
    
    def __init__(self, enable_detailed_tracking: bool = True):
        """
        Initialize performance metrics collector.
        
        Args:
            enable_detailed_tracking: Whether to enable detailed resource tracking
        """
        self.metrics: List[PerformanceMetric] = []
        self.enable_detailed_tracking = enable_detailed_tracking
        self.system_info = self._get_system_info()
        self.monitoring_thread = None
        self.monitoring_active = False
        self.resource_history = {
            'timestamps': deque(maxlen=1000),
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'memory_mb': deque(maxlen=1000)
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': psutil.os.name,
            'python_version': psutil.__version__
        }
    
    def add_metric(self, metric: PerformanceMetric) -> None:
        """
        Add a performance metric.
        
        Args:
            metric: PerformanceMetric instance to add
        """
        self.metrics.append(metric)
    
    @contextmanager
    def profile(self, operation: str, image_path: str = "", 
                image_size: tuple = (0, 0), metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling operations.
        
        Args:
            operation: Name of the operation
            image_path: Path to the image being processed
            image_size: Size of the image (width, height)
            metadata: Additional metadata
        """
        profiler = PerformanceProfiler(
            operation=operation,
            metrics_collector=self,
            image_path=image_path,
            image_size=image_size,
            metadata=metadata
        )
        with profiler:
            yield profiler
    
    def start_system_monitoring(self, interval: float = 1.0) -> None:
        """
        Start continuous system monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor():
            while self.monitoring_active:
                try:
                    timestamp = time.time()
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory = psutil.virtual_memory()
                    
                    self.resource_history['timestamps'].append(timestamp)
                    self.resource_history['cpu_percent'].append(cpu_percent)
                    self.resource_history['memory_percent'].append(memory.percent)
                    self.resource_history['memory_mb'].append(memory.used / 1024 / 1024)
                    
                    time.sleep(interval)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    break
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def stop_system_monitoring(self) -> None:
        """Stop continuous system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def calculate_operation_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate statistics for operations.
        
        Args:
            operation: Specific operation to analyze (None for all)
            
        Returns:
            Dictionary with operation statistics
        """
        if operation:
            filtered_metrics = [m for m in self.metrics if m.operation == operation]
        else:
            filtered_metrics = self.metrics
        
        if not filtered_metrics:
            return {}
        
        durations = [m.duration for m in filtered_metrics]
        memory_deltas = [m.memory_delta for m in filtered_metrics]
        cpu_usage = [m.cpu_percent for m in filtered_metrics]
        success_rate = sum(1 for m in filtered_metrics if m.success) / len(filtered_metrics)
        
        # Calculate processing rates for image operations
        processing_rates = [m.processing_rate for m in filtered_metrics if m.processing_rate > 0]
        
        stats = {
            'operation': operation or 'all_operations',
            'total_calls': len(filtered_metrics),
            'success_rate': success_rate,
            'duration_stats': {
                'mean': np.mean(durations),
                'median': np.median(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'p95': np.percentile(durations, 95),
                'p99': np.percentile(durations, 99)
            },
            'memory_stats': {
                'mean_delta': np.mean(memory_deltas),
                'max_delta': np.max(memory_deltas),
                'min_delta': np.min(memory_deltas),
                'std_delta': np.std(memory_deltas)
            },
            'cpu_stats': {
                'mean_usage': np.mean(cpu_usage),
                'max_usage': np.max(cpu_usage),
                'min_usage': np.min(cpu_usage)
            }
        }
        
        if processing_rates:
            stats['processing_rate_stats'] = {
                'mean_pixels_per_sec': np.mean(processing_rates),
                'median_pixels_per_sec': np.median(processing_rates),
                'max_pixels_per_sec': np.max(processing_rates)
            }
        
        return stats
    
    def calculate_throughput_metrics(self) -> Dict[str, float]:
        """
        Calculate throughput metrics.
        
        Returns:
            Dictionary with throughput metrics
        """
        if not self.metrics:
            return {}
        
        # Group by operation
        operation_metrics = defaultdict(list)
        for metric in self.metrics:
            operation_metrics[metric.operation].append(metric)
        
        throughput_metrics = {}
        
        for operation, metrics_list in operation_metrics.items():
            if not metrics_list:
                continue
                
            # Calculate operations per second
            total_duration = sum(m.duration for m in metrics_list)
            ops_per_second = len(metrics_list) / total_duration if total_duration > 0 else 0
            
            # Calculate average processing time
            avg_processing_time = total_duration / len(metrics_list) if metrics_list else 0
            
            # Calculate images per minute (for image processing operations)
            images_per_minute = (len(metrics_list) * 60) / total_duration if total_duration > 0 else 0
            
            throughput_metrics[operation] = {
                'operations_per_second': ops_per_second,
                'average_processing_time': avg_processing_time,
                'images_per_minute': images_per_minute,
                'total_operations': len(metrics_list),
                'total_duration': total_duration
            }
        
        return throughput_metrics
    
    def identify_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify performance bottlenecks.
        
        Returns:
            Dictionary with bottleneck analysis
        """
        if not self.metrics:
            return {}
        
        # Group by operation
        operation_stats = {}
        operations = set(m.operation for m in self.metrics)
        
        for operation in operations:
            operation_stats[operation] = self.calculate_operation_stats(operation)
        
        # Find slowest operations
        slowest_operations = sorted(
            operation_stats.items(),
            key=lambda x: x[1].get('duration_stats', {}).get('mean', 0),
            reverse=True
        )
        
        # Find operations with highest memory usage
        memory_intensive_operations = sorted(
            operation_stats.items(),
            key=lambda x: x[1].get('memory_stats', {}).get('mean_delta', 0),
            reverse=True
        )
        
        # Find operations with lowest success rate
        unreliable_operations = sorted(
            operation_stats.items(),
            key=lambda x: x[1].get('success_rate', 1.0)
        )
        
        # Analyze variance (operations with inconsistent performance)
        high_variance_operations = sorted(
            operation_stats.items(),
            key=lambda x: x[1].get('duration_stats', {}).get('std', 0),
            reverse=True
        )
        
        return {
            'slowest_operations': slowest_operations[:5],
            'memory_intensive_operations': memory_intensive_operations[:5],
            'unreliable_operations': unreliable_operations[:5],
            'high_variance_operations': high_variance_operations[:5],
            'recommendations': self._generate_recommendations(operation_stats)
        }
    
    def _generate_recommendations(self, operation_stats: Dict[str, Any]) -> List[str]:
        """
        Generate performance optimization recommendations.
        
        Args:
            operation_stats: Dictionary of operation statistics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        for operation, stats in operation_stats.items():
            duration_stats = stats.get('duration_stats', {})
            memory_stats = stats.get('memory_stats', {})
            
            # High processing time
            if duration_stats.get('mean', 0) > 1.0:
                recommendations.append(f"Consider optimizing '{operation}' - average processing time is {duration_stats['mean']:.2f}s")
            
            # High memory usage
            if memory_stats.get('mean_delta', 0) > 100:  # 100MB
                recommendations.append(f"'{operation}' uses significant memory ({memory_stats['mean_delta']:.1f}MB average)")
            
            # High variance
            if duration_stats.get('std', 0) > duration_stats.get('mean', 0) * 0.5:
                recommendations.append(f"'{operation}' has inconsistent performance (high variance)")
            
            # Low success rate
            if stats.get('success_rate', 1.0) < 0.95:
                recommendations.append(f"'{operation}' has low success rate ({stats['success_rate']:.1%})")
        
        return recommendations
    
    def analyze_image_size_impact(self) -> Dict[str, Any]:
        """
        Analyze how image size affects processing performance.
        
        Returns:
            Dictionary with image size impact analysis
        """
        image_metrics = [m for m in self.metrics if m.image_size[0] > 0 and m.image_size[1] > 0]
        
        if not image_metrics:
            return {}
        
        # Calculate image areas (in megapixels)
        areas = []
        durations = []
        processing_rates = []
        
        for metric in image_metrics:
            area_mp = (metric.image_size[0] * metric.image_size[1]) / 1_000_000
            areas.append(area_mp)
            durations.append(metric.duration)
            if metric.processing_rate > 0:
                processing_rates.append(metric.processing_rate)
        
        # Calculate correlations
        area_duration_corr = np.corrcoef(areas, durations)[0, 1] if len(areas) > 1 else 0
        
        # Group by size categories
        size_categories = {
            'small': [m for m in image_metrics if (m.image_size[0] * m.image_size[1]) < 500_000],
            'medium': [m for m in image_metrics if 500_000 <= (m.image_size[0] * m.image_size[1]) < 2_000_000],
            'large': [m for m in image_metrics if (m.image_size[0] * m.image_size[1]) >= 2_000_000]
        }
        
        category_stats = {}
        for category, metrics_list in size_categories.items():
            if metrics_list:
                durations_cat = [m.duration for m in metrics_list]
                category_stats[category] = {
                    'count': len(metrics_list),
                    'mean_duration': np.mean(durations_cat),
                    'median_duration': np.median(durations_cat),
                    'std_duration': np.std(durations_cat)
                }
        
        return {
            'area_duration_correlation': area_duration_corr,
            'size_category_stats': category_stats,
            'total_images_analyzed': len(image_metrics),
            'average_processing_rate': np.mean(processing_rates) if processing_rates else 0
        }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary with all performance metrics
        """
        overall_stats = self.calculate_operation_stats()
        throughput_metrics = self.calculate_throughput_metrics()
        bottlenecks = self.identify_bottlenecks()
        image_size_impact = self.analyze_image_size_impact()
        
        # Operation-specific stats
        operations = set(m.operation for m in self.metrics)
        operation_details = {}
        for operation in operations:
            operation_details[operation] = self.calculate_operation_stats(operation)
        
        return {
            'system_info': self.system_info,
            'summary': {
                'total_operations': len(self.metrics),
                'unique_operations': len(operations),
                'overall_success_rate': overall_stats.get('success_rate', 0),
                'monitoring_duration': self._get_monitoring_duration()
            },
            'overall_stats': overall_stats,
            'operation_details': operation_details,
            'throughput_metrics': throughput_metrics,
            'bottleneck_analysis': bottlenecks,
            'image_size_impact': image_size_impact,
            'resource_history': self._get_resource_history_summary()
        }
    
    def _get_monitoring_duration(self) -> float:
        """Get total monitoring duration in seconds."""
        if not self.metrics:
            return 0.0
        
        start_time = min(m.start_time for m in self.metrics)
        end_time = max(m.end_time for m in self.metrics)
        return end_time - start_time
    
    def _get_resource_history_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage history."""
        if not self.resource_history['timestamps']:
            return {}
        
        cpu_data = list(self.resource_history['cpu_percent'])
        memory_data = list(self.resource_history['memory_percent'])
        
        return {
            'cpu_usage': {
                'mean': np.mean(cpu_data),
                'max': np.max(cpu_data),
                'min': np.min(cpu_data),
                'std': np.std(cpu_data)
            },
            'memory_usage': {
                'mean': np.mean(memory_data),
                'max': np.max(memory_data),
                'min': np.min(memory_data),
                'std': np.std(memory_data)
            },
            'monitoring_points': len(cpu_data)
        }
    
    def save_report(self, filepath: str) -> None:
        """
        Save the detailed performance report to a JSON file.
        
        Args:
            filepath: Path to save the report
        """
        report = self.get_detailed_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def save_metrics_csv(self, filepath: str) -> None:
        """
        Save detailed metrics to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        data = []
        for metric in self.metrics:
            data.append({
                'operation': metric.operation,
                'duration': metric.duration,
                'memory_before': metric.memory_before,
                'memory_after': metric.memory_after,
                'memory_delta': metric.memory_delta,
                'cpu_percent': metric.cpu_percent,
                'image_width': metric.image_size[0],
                'image_height': metric.image_size[1],
                'image_path': metric.image_path,
                'success': metric.success,
                'error_message': metric.error_message,
                'processing_rate': metric.processing_rate,
                'start_time': metric.start_time,
                'end_time': metric.end_time
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def plot_operation_timings(self, save_path: Optional[str] = None) -> None:
        """
        Plot operation timing distributions.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.metrics:
            print("No metrics to plot")
            return
        
        # Group by operation
        operation_durations = defaultdict(list)
        for metric in self.metrics:
            operation_durations[metric.operation].append(metric.duration)
        
        if len(operation_durations) == 1:
            # Single operation - histogram
            operation = list(operation_durations.keys())[0]
            durations = operation_durations[operation]
            
            plt.figure(figsize=(10, 6))
            plt.hist(durations, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Frequency')
            plt.title(f'Duration Distribution for {operation}')
            plt.axvline(np.mean(durations), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(durations):.3f}s')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Multiple operations - box plot
            operations = list(operation_durations.keys())
            durations_list = [operation_durations[op] for op in operations]
            
            plt.figure(figsize=(12, 8))
            plt.boxplot(durations_list, labels=operations)
            plt.xlabel('Operation')
            plt.ylabel('Duration (seconds)')
            plt.title('Duration Distribution by Operation')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_resource_usage(self, save_path: Optional[str] = None) -> None:
        """
        Plot system resource usage over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.resource_history['timestamps']:
            print("No resource monitoring data available")
            return
        
        timestamps = list(self.resource_history['timestamps'])
        cpu_data = list(self.resource_history['cpu_percent'])
        memory_data = list(self.resource_history['memory_percent'])
        
        # Convert timestamps to relative time
        start_time = timestamps[0]
        relative_times = [(t - start_time) / 60 for t in timestamps]  # Convert to minutes
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # CPU Usage
        ax1.plot(relative_times, cpu_data, 'b-', alpha=0.7, linewidth=1)
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('System Resource Usage Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # Memory Usage
        ax2.plot(relative_times, memory_data, 'r-', alpha=0.7, linewidth=1)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_throughput_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Plot throughput analysis.
        
        Args:
            save_path: Optional path to save the plot
        """
        throughput_metrics = self.calculate_throughput_metrics()
        
        if not throughput_metrics:
            print("No throughput data available")
            return
        
        operations = list(throughput_metrics.keys())
        ops_per_second = [throughput_metrics[op]['operations_per_second'] for op in operations]
        avg_processing_time = [throughput_metrics[op]['average_processing_time'] for op in operations]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Operations per second
        ax1.bar(operations, ops_per_second, alpha=0.7)
        ax1.set_xlabel('Operation')
        ax1.set_ylabel('Operations per Second')
        ax1.set_title('Throughput by Operation')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Average processing time
        ax2.bar(operations, avg_processing_time, alpha=0.7, color='orange')
        ax2.set_xlabel('Operation')
        ax2.set_ylabel('Average Processing Time (seconds)')
        ax2.set_title('Average Processing Time by Operation')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def benchmark_function(func: Callable, *args, iterations: int = 10, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function's performance.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Dictionary with benchmark results
    """
    durations = []
    memory_usage = []
    
    process = psutil.Process()
    
    for _ in range(iterations):
        # Collect garbage before each run
        gc.collect()
        
        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Time the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
        end_time = time.time()
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024
        
        durations.append(end_time - start_time)
        memory_usage.append(memory_after - memory_before)
    
    return {
        'mean_duration': np.mean(durations),
        'median_duration': np.median(durations),
        'std_duration': np.std(durations),
        'min_duration': np.min(durations),
        'max_duration': np.max(durations),
        'mean_memory_delta': np.mean(memory_usage),
        'max_memory_delta': np.max(memory_usage),
        'iterations': iterations,
        'success_rate': success if 'success' in locals() else True
    }
