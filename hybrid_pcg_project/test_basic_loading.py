"""
Simple test for data loading functionality
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_basic_loading():
    """Test basic data loading without complex imports"""
    
    # Get dataset path
    dataset_path = Path(__file__).parent.parent / 'training_data'
    metadata_file = Path(__file__).parent.parent / 'training_data.csv'
    
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset exists: {dataset_path.exists()}")
    print(f"Metadata file: {metadata_file}")
    print(f"Metadata exists: {metadata_file.exists()}")
    
    if not metadata_file.exists():
        print("Error: Metadata file not found!")
        return
    
    # Load metadata
    try:
        metadata = pd.read_csv(metadata_file)
        print(f"\nMetadata loaded successfully!")
        print(f"Number of patients: {len(metadata)}")
        print(f"Columns: {list(metadata.columns)}")
        
        # Show first few rows
        print(f"\nFirst 3 patients:")
        print(metadata.head(3))
        
        # Check outcome distribution
        if 'Outcome' in metadata.columns:
            outcome_dist = metadata['Outcome'].value_counts()
            print(f"\nOutcome distribution:")
            print(outcome_dist)
        
        # Check recording locations
        if 'Recording locations:' in metadata.columns:
            locations = metadata['Recording locations:'].unique()
            print(f"\nUnique recording location combinations:")
            for loc in locations[:5]:  # Show first 5
                print(f"  {loc}")
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    # Test audio file access
    print(f"\nTesting audio file access...")
    
    # Get first few patient IDs
    patient_ids = metadata['Patient ID'].astype(str).tolist()[:3]
    
    for patient_id in patient_ids:
        print(f"\nChecking patient {patient_id}:")
        
        # Check what files exist
        files_found = []
        locations = ['AV', 'PV', 'TV', 'MV']
        
        for location in locations:
            wav_file = dataset_path / f"{patient_id}_{location}.wav"
            hea_file = dataset_path / f"{patient_id}_{location}.hea"
            tsv_file = dataset_path / f"{patient_id}_{location}.tsv"
            
            if wav_file.exists():
                files_found.append(f"{location} (wav: {wav_file.stat().st_size} bytes)")
                
                if hea_file.exists():
                    files_found[-1] += " + hea"
                if tsv_file.exists():
                    files_found[-1] += " + tsv"
        
        if files_found:
            print(f"  Files found: {files_found}")
        else:
            print(f"  No audio files found for patient {patient_id}")
    
    print(f"\nBasic data loading test completed!")

if __name__ == "__main__":
    test_basic_loading()
