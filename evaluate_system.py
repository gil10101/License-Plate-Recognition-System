#!/usr/bin/env python3
"""
Comprehensive system evaluation script for License Plate Recognition System.

This script demonstrates how to use the enhanced metrics module to evaluate
the license plate recognition system performance and export data for BI tools.

Usage:
    python evaluate_system.py --csv results.csv --output-dir metrics_export
    python evaluate_system.py --csv enhanced_results.csv --format tableau
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Import the new metrics module
from metrics import EvaluationMetrics, export_metrics_for_bi


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate License Plate Recognition System')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to results CSV file')
    parser.add_argument('--images-dir', type=str, default='',
                       help='Directory containing test images')
    parser.add_argument('--output-dir', type=str, default='metrics_export',
                       help='Output directory for exported metrics')
    parser.add_argument('--format', type=str, choices=['powerbi', 'tableau', 'sqlite'],
                       default='powerbi', help='Export format for BI tools')
    parser.add_argument('--save-report', type=str, default='evaluation_report.json',
                       help='Path to save comprehensive report')
    parser.add_argument('--save-results', type=str, default='detailed_results.csv',
                       help='Path to save detailed results CSV')
    
    args = parser.parse_args()
    
    # Validate input files
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print("License Plate Recognition System Evaluation")
    print("=" * 50)
    print(f"Input CSV: {csv_path}")
    print(f"Export Format: {args.format}")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Initialize evaluation metrics
    print("1. Loading and processing results...")
    evaluation = EvaluationMetrics()
    
    try:
        evaluation.evaluate_from_results_csv(str(csv_path), args.images_dir)
        print(f"   ✓ Loaded {len(evaluation.end_to_end_results)} samples")
    except Exception as e:
        print(f"   ✗ Error loading CSV: {e}")
        sys.exit(1)
    
    # Generate comprehensive report
    print("2. Calculating comprehensive metrics...")
    try:
        report = evaluation.get_comprehensive_report()
        print("   ✓ Generated comprehensive report")
        
        # Display key metrics
        end_to_end_metrics = report.get('end_to_end_metrics', {}).get('overall_metrics', {})
        quality_score = report.get('quality_assessment', {}).get('overall_quality_score', 0)
        
        print(f"   → End-to-End Success Rate: {end_to_end_metrics.get('end_to_end_success_rate', 0):.1%}")
        print(f"   → Detection Success Rate: {end_to_end_metrics.get('detection_success_rate', 0):.1%}")
        print(f"   → OCR Success Rate: {end_to_end_metrics.get('ocr_success_rate', 0):.1%}")
        print(f"   → Overall Quality Score: {quality_score:.3f}")
        
    except Exception as e:
        print(f"   ✗ Error calculating metrics: {e}")
        sys.exit(1)
    
    # Export for BI tools
    print(f"3. Exporting data for {args.format.upper()}...")
    try:
        exported_files = export_metrics_for_bi(
            evaluation, 
            output_format=args.format,
            output_dir=args.output_dir
        )
        print(f"   ✓ Exported {len(exported_files)} files")
        for data_type, filepath in exported_files.items():
            print(f"     - {data_type}: {filepath}")
            
    except Exception as e:
        print(f"   ✗ Error exporting data: {e}")
        sys.exit(1)
    
    # Save comprehensive report
    print("4. Saving detailed reports...")
    try:
        evaluation.save_comprehensive_report(args.save_report)
        print(f"   ✓ Saved comprehensive report: {args.save_report}")
        
        evaluation.save_results_csv(args.save_results)
        print(f"   ✓ Saved detailed results: {args.save_results}")
        
    except Exception as e:
        print(f"   ✗ Error saving reports: {e}")
        sys.exit(1)
    
    # Display recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print("\n5. System Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Display failure analysis
    failure_analysis = report.get('failure_analysis', {})
    if failure_analysis:
        print("\n6. Failure Analysis Summary:")
        det_failures = failure_analysis.get('detection_failures', {})
        ocr_failures = failure_analysis.get('ocr_failures', {})
        text_failures = failure_analysis.get('text_accuracy_failures', {})
        
        print(f"   → Detection Failures: {det_failures.get('count', 0)} ({det_failures.get('percentage', 0):.1f}%)")
        print(f"   → OCR Failures: {ocr_failures.get('count', 0)} ({ocr_failures.get('percentage', 0):.1f}%)")
        print(f"   → Text Accuracy Failures: {text_failures.get('count', 0)} ({text_failures.get('percentage', 0):.1f}%)")
    
    print(f"\n✓ Evaluation completed successfully!")
    print(f"📊 Import the files in '{args.output_dir}' into {args.format.upper()} for visualization")
    
    if args.format == 'powerbi':
        print("\n📝 Power BI Setup Instructions:")
        print("   1. Open Power BI Desktop")
        print("   2. Get Data → Text/CSV")
        print(f"   3. Import 'license_plate_results.csv' as the main table")
        print("   4. Import other CSV files as supporting tables")
        print("   5. Create relationships based on common columns")
        print("   6. Use 'kpi_summary.csv' for dashboard KPI tiles")


def show_sample_analysis(csv_path: str):
    """Show a quick sample analysis of the CSV data."""
    try:
        df = pd.read_csv(csv_path)
        print(f"\nQuick Analysis of {csv_path}:")
        print(f"  - Total samples: {len(df)}")
        
        if 'predicted' in df.columns and 'expected' in df.columns:
            # Calculate basic accuracy
            df['match'] = df['predicted'].str.upper().str.strip() == df['expected'].str.upper().str.strip()
            accuracy = df['match'].mean()
            print(f"  - Exact match accuracy: {accuracy:.1%}")
        
        if 'confidence' in df.columns:
            mean_conf = df['confidence'].mean()
            print(f"  - Average confidence: {mean_conf:.3f}")
        
    except Exception as e:
        print(f"  - Error analyzing CSV: {e}")


if __name__ == '__main__':
    main()
