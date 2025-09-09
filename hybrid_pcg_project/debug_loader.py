#!/usr/bin/env python3
"""Debug script to test data loading"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.preprocessing.hybrid_data_loader import HybridDataLoader
    print("✓ Import successful")
    
    # Set correct paths
    circor_path = r"d:\WORKSPACE\PYTHON\the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    yaseen_path = r"d:\WORKSPACE\PYTHON\the-circor-digiscope-phonocardiogram-dataset-1.0.3\yaseen"
    
    loader = HybridDataLoader(circor_path=circor_path, yaseen_path=yaseen_path)
    print("✓ Loader created")
    
    data = loader.load_circor_dataset()
    print(f"✓ CirCor data loaded: {len(data)} patients")
    
    # Show first few patients
    for i, (patient_id, patient_data) in enumerate(list(data.items())[:3]):
        print(f"Patient {patient_id}: {len(patient_data['recordings'])} recordings")
        for location, recording in patient_data['recordings'].items():
            print(f"  - {location}: {os.path.basename(recording['audio_file'])}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
