"""
Quick test script to run preprocessing on a small subset of data
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from preprocessing import PCGDataLoader, PCGPreprocessor, PCGFeatureExtractor

def test_preprocessing():
    """Test the preprocessing pipeline with a small dataset sample"""
    
    # Load configuration
    config_path = project_root / 'configs' / 'preprocessing.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths to be relative to dataset location
    config['dataset']['source_path'] = str(project_root.parent / 'training_data')
    config['dataset']['metadata_file'] = str(project_root.parent / 'training_data.csv')
    
    print("Testing PCG Data Loader...")
    data_loader = PCGDataLoader(config)
    
    # Get dataset statistics
    stats = data_loader.get_dataset_statistics()
    print(f"Dataset Statistics:")
    print(f"  Total patients: {stats['total_patients']}")
    print(f"  Total recordings: {stats['total_recordings']}")
    print(f"  Location distribution: {stats['locations_count']}")
    print(f"  Duration stats: Mean={stats['duration_stats']['mean']:.2f}s, Max={stats['duration_stats']['max']:.2f}s")
    
    # Test with a few patients
    patient_ids = data_loader.get_patient_ids()[:3]  # Test with first 3 patients
    
    print(f"\nTesting with patients: {patient_ids}")
    
    preprocessor = PCGPreprocessor(config)
    feature_extractor = PCGFeatureExtractor(config)
    
    for patient_id in patient_ids:
        print(f"\nProcessing patient {patient_id}...")
        
        # Get recordings for this patient
        recordings = data_loader.get_patient_recordings(patient_id)
        patient_metadata = data_loader.get_patient_metadata(patient_id)
        
        print(f"  Patient metadata: Age={patient_metadata.get('age')}, Sex={patient_metadata.get('sex')}, Outcome={patient_metadata.get('outcome')}")
        print(f"  Available recordings: {list(recordings.keys())}")
        
        for location, recording in recordings.items():
            print(f"    Processing {location} recording...")
            
            # Get audio data
            audio = recording['audio']
            sample_rate = recording['sample_rate']
            annotations = recording['annotations']
            
            print(f"      Original: {len(audio)} samples at {sample_rate} Hz ({len(audio)/sample_rate:.2f}s)")
            print(f"      Annotations: {len(annotations)} segments")
            
            # Quality check
            quality_checks = preprocessor.quality_check(audio, sample_rate)
            print(f"      Quality checks: {quality_checks}")
            
            # Preprocess audio
            processed_audio, processed_sr = preprocessor.preprocess_audio(audio, sample_rate)
            print(f"      Processed: {len(processed_audio)} samples at {processed_sr} Hz ({len(processed_audio)/processed_sr:.2f}s)")
            
            # Extract features
            features = feature_extractor.extract_all_features(processed_audio, processed_sr)
            numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
            print(f"      Extracted {len(numeric_features)} numeric features")
            
            # Segment audio
            segments = preprocessor.segment_audio(processed_audio, processed_sr, annotations)
            print(f"      Created {len(segments)} segments")
            
            for seg in segments[:3]:  # Show first 3 segments
                print(f"        Segment: {seg['label_name']} ({seg['duration']:.2f}s)")
            
            # Test augmentation
            augmented = preprocessor.augment_audio(processed_audio, processed_sr)
            print(f"      Generated {len(augmented)} augmented versions")
            
            break  # Only process first location for testing
        
        break  # Only process first patient for full testing

if __name__ == "__main__":
    test_preprocessing()
