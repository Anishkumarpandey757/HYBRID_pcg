"""
Main preprocessing script for CirCor DigiScope dataset
Orchestrates the complete preprocessing pipeline
"""

import os
import sys
import yaml
import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader import PCGDataLoader
from audio_processor import PCGPreprocessor
from feature_extractor import PCGFeatureExtractor

class PCGDatasetPreprocessor:
    """
    Main preprocessing pipeline for PCG dataset
    """
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_loader = PCGDataLoader(self.config)
        self.audio_processor = PCGPreprocessor(self.config)
        self.feature_extractor = PCGFeatureExtractor(self.config)
        
        # Set up paths
        self.output_path = Path(self.config['dataset']['output_path'])
        self.features_path = Path(self.config['dataset']['features_path'])
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_path / 'preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def preprocess_dataset(self):
        """
        Run complete preprocessing pipeline
        """
        self.logger.info("Starting dataset preprocessing...")
        
        # Get all recordings
        all_recordings = self.data_loader.get_all_recordings()
        self.logger.info(f"Found {len(all_recordings)} recordings")
        
        # Initialize data containers
        processed_data = []
        feature_data = []
        segment_data = []
        
        # Process each recording
        for i, recording in enumerate(tqdm(all_recordings, desc="Processing recordings")):
            try:
                result = self._process_single_recording(recording)
                if result:
                    processed_data.append(result['processed'])
                    feature_data.extend(result['features'])
                    segment_data.extend(result['segments'])
                    
            except Exception as e:
                self.logger.error(f"Error processing recording {recording.get('patient_id', 'unknown')}: {e}")
                continue
        
        # Save processed data
        self._save_processed_data(processed_data, feature_data, segment_data)
        
        # Generate dataset statistics
        self._generate_statistics(processed_data, feature_data, segment_data)
        
        self.logger.info("Preprocessing completed successfully!")
    
    def _process_single_recording(self, recording: Dict) -> Dict:
        """
        Process a single recording
        
        Args:
            recording: Recording dictionary from data loader
            
        Returns:
            Dictionary with processed data, features, and segments
        """
        patient_id = recording['patient_id']
        location = recording['location']
        
        # Get audio and metadata
        audio = recording['audio']
        sample_rate = recording['sample_rate']
        annotations = recording['annotations']
        patient_metadata = recording['patient_metadata']
        
        # Quality check
        quality_checks = self.audio_processor.quality_check(audio, sample_rate)
        if not quality_checks['overall_quality']:
            self.logger.warning(f"Recording {patient_id}_{location} failed quality checks")
        
        # Preprocess audio
        processed_audio, processed_sr = self.audio_processor.preprocess_audio(audio, sample_rate)
        
        # Extract full recording features
        recording_features = self.feature_extractor.extract_all_features(processed_audio, processed_sr)
        
        # Add metadata to features
        recording_features.update({
            'patient_id': patient_id,
            'location': location,
            'original_duration': len(audio) / sample_rate,
            'processed_duration': len(processed_audio) / processed_sr,
            'sample_rate': processed_sr,
            'quality_score': sum(quality_checks.values()) / len(quality_checks)
        })
        
        # Add patient metadata
        recording_features.update(patient_metadata)
        
        # Segment audio and extract segment features
        segments = self.audio_processor.segment_audio(processed_audio, processed_sr, annotations)
        segment_features = []
        
        for segment in segments:
            seg_features = self.feature_extractor.extract_all_features(segment['audio'], processed_sr)
            seg_features.update({
                'patient_id': patient_id,
                'location': location,
                'segment_start': segment['start_time'],
                'segment_end': segment['end_time'],
                'segment_duration': segment['duration'],
                'segment_label': segment['label_name'],
                'segment_label_id': segment['label_id']
            })
            seg_features.update(patient_metadata)
            segment_features.append(seg_features)
        
        # Extract heart cycles if annotations available
        if annotations:
            heart_cycles = self.audio_processor.extract_heart_cycles(processed_audio, processed_sr, annotations)
            for cycle in heart_cycles:
                cycle_features = self.feature_extractor.extract_all_features(cycle['audio'], processed_sr)
                cycle_features.update({
                    'patient_id': patient_id,
                    'location': location,
                    'cycle_start': cycle['start_time'],
                    'cycle_end': cycle['end_time'],
                    'cycle_duration': cycle['duration'],
                    'cycle_index': cycle['cycle_index'],
                    'segment_label': 'heart_cycle',
                    'segment_label_id': 99
                })
                cycle_features.update(patient_metadata)
                segment_features.append(cycle_features)
        
        # Data augmentation if enabled
        augmented_audio_list = self.audio_processor.augment_audio(processed_audio, processed_sr)
        augmented_features = []
        
        for aug_idx, aug_audio in enumerate(augmented_audio_list[1:]):  # Skip original
            aug_features = self.feature_extractor.extract_all_features(aug_audio, processed_sr)
            aug_features.update({
                'patient_id': f"{patient_id}_aug_{aug_idx}",
                'location': location,
                'original_patient_id': patient_id,
                'augmentation_index': aug_idx,
                'is_augmented': True
            })
            aug_features.update(patient_metadata)
            augmented_features.append(aug_features)
        
        return {
            'processed': {
                'patient_id': patient_id,
                'location': location,
                'audio': processed_audio,
                'sample_rate': processed_sr,
                'annotations': annotations,
                'quality_checks': quality_checks,
                'metadata': patient_metadata
            },
            'features': [recording_features] + augmented_features,
            'segments': segment_features
        }
    
    def _save_processed_data(self, processed_data: List[Dict], 
                           feature_data: List[Dict], 
                           segment_data: List[Dict]):
        """
        Save all processed data to files
        """
        self.logger.info("Saving processed data...")
        
        # Save processed audio data
        with h5py.File(self.output_path / 'processed_audio.h5', 'w') as f:
            for i, data in enumerate(processed_data):
                group = f.create_group(f"recording_{i}")
                group.create_dataset('audio', data=data['audio'])
                group.attrs['patient_id'] = data['patient_id']
                group.attrs['location'] = data['location']
                group.attrs['sample_rate'] = data['sample_rate']
                
                # Save annotations
                if data['annotations']:
                    ann_group = group.create_group('annotations')
                    for j, ann in enumerate(data['annotations']):
                        ann_subgroup = ann_group.create_group(f"annotation_{j}")
                        for key, value in ann.items():
                            ann_subgroup.attrs[key] = value
        
        # Save features as CSV
        if feature_data:
            # Convert to DataFrame and save
            feature_df = pd.DataFrame(feature_data)
            feature_df.to_csv(self.features_path / 'recording_features.csv', index=False)
            
            # Save feature matrix for ML
            numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
            feature_matrix = feature_df[numeric_columns].values
            feature_names = numeric_columns.tolist()
            
            np.save(self.features_path / 'feature_matrix.npy', feature_matrix)
            with open(self.features_path / 'feature_names.json', 'w') as f:
                json.dump(feature_names, f)
        
        # Save segment features
        if segment_data:
            segment_df = pd.DataFrame(segment_data)
            segment_df.to_csv(self.features_path / 'segment_features.csv', index=False)
            
            # Save segment feature matrix
            numeric_columns = segment_df.select_dtypes(include=[np.number]).columns
            segment_matrix = segment_df[numeric_columns].values
            segment_names = numeric_columns.tolist()
            
            np.save(self.features_path / 'segment_feature_matrix.npy', segment_matrix)
            with open(self.features_path / 'segment_feature_names.json', 'w') as f:
                json.dump(segment_names, f)
        
        self.logger.info(f"Saved {len(processed_data)} processed recordings")
        self.logger.info(f"Saved {len(feature_data)} feature vectors")
        self.logger.info(f"Saved {len(segment_data)} segment features")
    
    def _generate_statistics(self, processed_data: List[Dict], 
                           feature_data: List[Dict], 
                           segment_data: List[Dict]):
        """
        Generate and save dataset statistics
        """
        self.logger.info("Generating dataset statistics...")
        
        stats = {
            'dataset_info': {
                'total_recordings': len(processed_data),
                'total_features': len(feature_data),
                'total_segments': len(segment_data),
                'processing_config': self.config
            },
            'recording_stats': {},
            'feature_stats': {},
            'segment_stats': {}
        }
        
        # Recording statistics
        locations = [data['location'] for data in processed_data]
        patients = [data['patient_id'] for data in processed_data]
        
        stats['recording_stats'] = {
            'unique_patients': len(set(patients)),
            'location_distribution': {loc: locations.count(loc) for loc in set(locations)},
            'average_duration': np.mean([len(data['audio']) / data['sample_rate'] for data in processed_data]),
            'quality_scores': [np.mean(list(data['quality_checks'].values())) for data in processed_data]
        }
        
        # Feature statistics
        if feature_data:
            feature_df = pd.DataFrame(feature_data)
            numeric_df = feature_df.select_dtypes(include=[np.number])
            
            stats['feature_stats'] = {
                'num_features': len(numeric_df.columns),
                'feature_means': numeric_df.mean().to_dict(),
                'feature_stds': numeric_df.std().to_dict(),
                'missing_values': numeric_df.isnull().sum().to_dict()
            }
        
        # Segment statistics
        if segment_data:
            segment_df = pd.DataFrame(segment_data)
            
            stats['segment_stats'] = {
                'total_segments': len(segment_df),
                'segment_types': segment_df['segment_label'].value_counts().to_dict(),
                'average_segment_duration': segment_df['segment_duration'].mean() if 'segment_duration' in segment_df.columns else None,
                'segments_per_recording': len(segment_df) / len(processed_data)
            }
        
        # Save statistics
        with open(self.output_path / 'dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info("Statistics saved to dataset_statistics.json")
    
    def create_train_val_test_splits(self):
        """
        Create train/validation/test splits
        """
        self.logger.info("Creating train/validation/test splits...")
        
        validation_config = self.config.get('validation', {})
        train_split = validation_config.get('train_split', 0.7)
        val_split = validation_config.get('val_split', 0.15)
        test_split = validation_config.get('test_split', 0.15)
        random_seed = validation_config.get('random_seed', 42)
        
        # Load feature data
        feature_df = pd.read_csv(self.features_path / 'recording_features.csv')
        
        # Get unique patients for stratification
        unique_patients = feature_df['patient_id'].unique()
        np.random.seed(random_seed)
        np.random.shuffle(unique_patients)
        
        # Split patients
        n_patients = len(unique_patients)
        train_end = int(train_split * n_patients)
        val_end = int((train_split + val_split) * n_patients)
        
        train_patients = unique_patients[:train_end]
        val_patients = unique_patients[train_end:val_end]
        test_patients = unique_patients[val_end:]
        
        # Create splits
        train_df = feature_df[feature_df['patient_id'].isin(train_patients)]
        val_df = feature_df[feature_df['patient_id'].isin(val_patients)]
        test_df = feature_df[feature_df['patient_id'].isin(test_patients)]
        
        # Save splits
        train_df.to_csv(self.features_path / 'train_features.csv', index=False)
        val_df.to_csv(self.features_path / 'val_features.csv', index=False)
        test_df.to_csv(self.features_path / 'test_features.csv', index=False)
        
        # Save patient splits
        split_info = {
            'train_patients': train_patients.tolist(),
            'val_patients': val_patients.tolist(),
            'test_patients': test_patients.tolist(),
            'split_ratios': {
                'train': len(train_df) / len(feature_df),
                'val': len(val_df) / len(feature_df),
                'test': len(test_df) / len(feature_df)
            }
        }
        
        with open(self.features_path / 'data_splits.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        self.logger.info(f"Created splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess CirCor DigiScope dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to preprocessing configuration file')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create train/validation/test splits')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = PCGDatasetPreprocessor(args.config)
    
    # Run preprocessing
    preprocessor.preprocess_dataset()
    
    # Create data splits if requested
    if args.create_splits:
        preprocessor.create_train_val_test_splits()

if __name__ == "__main__":
    main()
