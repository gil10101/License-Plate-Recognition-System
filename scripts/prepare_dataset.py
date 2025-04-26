#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def main():
    """
    Prepare the license plate dataset for YOLOv5 training.
    This script only handles the dataset preparation step.
    """
    print("License Plate Dataset Preparation")
    print("=================================")
    
    # Prepare the dataset
    try:
        sys.path.append('data')
        from data.prepare_license_data import create_yolo_dataset
        os.chdir('data')
        create_yolo_dataset()
        os.chdir('..')
        
        data_yaml_path = 'data/license_plate_data/dataset.yaml'
        if not os.path.exists(data_yaml_path):
            print(f"Error: Data YAML file not found at {data_yaml_path}")
            sys.exit(1)
        
        print("\nDataset preparation completed successfully.")
        print(f"Data YAML file created at: {data_yaml_path}")
        print("You can now run train_only.py to train the model.")
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 