"""
Hybrid Preprocessing Pipeline for CirCor and Yaseen datasets
Implements the complete blueprint preprocessing workflow
"""

import numpy as np
import pandas as pd
import h5py
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .hybrid_data_loader import HybridDataLoader, CIRCOR_CONFIG, YASEEN_CONFIG
from .hybrid_audio_processor import HybridAudioProcessor

class HybridPreprocessor:
    """
    Main preprocessing pipeline that handles both datasets according to blueprint
    """
    
    def __init__(self, 
                 circor_path: str = None,
                 yaseen_path: str = None,
                 output_dir: str = "data",
                 use_dwt: bool = True,
                 yaseen_target_sr: int = 4000):  # Can be 22050 or 4000
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.data_loader = HybridDataLoader(circor_path, yaseen_path)
        self.audio_processor = HybridAudioProcessor(
            yaseen_sr=yaseen_target_sr,
            use_dwt=use_dwt
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output subdirectories
        self.preprocessed_dir = self.output_dir / "preprocessed"
        self.features_dir = self.output_dir / "features"
        self.splits_dir = self.output_dir / "splits"
        
        for dir_path in [self.preprocessed_dir, self.features_dir, self.splits_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def preprocess_circor_dataset(self, max_patients: Optional[int] = None) -> Dict:
        """
        Preprocess CirCor dataset following blueprint:
        - Group by patient
        - Process each site recording
        - Save patient-level data for multi-location fusion
        """
        self.logger.info("Starting CirCor preprocessing...")
        
        # Load dataset
        patients = self.data_loader.load_circor_dataset()
        
        if max_patients:
            patient_ids = list(patients.keys())[:max_patients]
            patients = {pid: patients[pid] for pid in patient_ids}
        
        processed_data = {}
        patient_features = []
        
        # HDF5 file for processed spectrograms
        circor_h5_path = self.preprocessed_dir / "circor_spectrograms.h5"
        
        with h5py.File(circor_h5_path, 'w') as h5f:
            
            for patient_id, patient_data in tqdm(patients.items(), desc="Processing CirCor patients"):
                try:
                    patient_spectrograms = {}
                    patient_metadata = {
                        'patient_id': patient_id,
                        'age': patient_data.get('age'),
                        'sex': patient_data.get('sex'),
                        'murmur': patient_data.get('murmur', 'Unknown'),
                        'outcome': patient_data.get('outcome', 'Unknown'),
                        'campaign': patient_data.get('campaign'),
                        'available_locations': list(patient_data['recordings'].keys())
                    }
                    
                    # Process each location
                    for location, recording_info in patient_data['recordings'].items():
                        try:
                            # Load audio
                            audio, sr = self.data_loader.load_audio_circor(patient_id, location)
                            
                            # Process to 224x224 spectrogram
                            spectrogram = self.audio_processor.process_circor_audio(
                                audio, sr, augment=False
                            )
                            
                            # Store in HDF5
                            grp_name = f"{patient_id}/{location}"
                            h5f.create_dataset(grp_name, data=spectrogram)
                            
                            # Store metadata
                            patient_spectrograms[location] = {
                                'shape': spectrogram.shape,
                                'h5_path': grp_name
                            }
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to process {patient_id}_{location}: {e}")
                            continue
                    
                    if patient_spectrograms:
                        processed_data[patient_id] = {
                            'metadata': patient_metadata,
                            'spectrograms': patient_spectrograms
                        }
                        
                        # Collect features for CSV
                        patient_features.append({
                            'patient_id': patient_id,
                            'dataset': 'circor',
                            'num_locations': len(patient_spectrograms),
                            'locations': ','.join(patient_spectrograms.keys()),
                            **patient_metadata
                        })
                
                except Exception as e:
                    self.logger.error(f"Error processing patient {patient_id}: {e}")
                    continue
        
        # Save processed data info
        circor_info_path = self.preprocessed_dir / "circor_processed_info.json"
        with open(circor_info_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        # Save features CSV
        if patient_features:
            features_df = pd.DataFrame(patient_features)
            features_df.to_csv(self.features_dir / "circor_patient_features.csv", index=False)
        
        self.logger.info(f"CirCor preprocessing completed. Processed {len(processed_data)} patients")
        return processed_data
    
    def preprocess_yaseen_dataset(self, max_recordings: Optional[int] = None) -> Dict:
        """
        Preprocess Yaseen dataset following blueprint:
        - Single-recording per file
        - 5-class classification
        - Save recording-level data
        """
        self.logger.info("Starting Yaseen preprocessing...")
        
        # Load dataset
        recordings = self.data_loader.load_yaseen_dataset()
        
        if max_recordings:
            record_ids = list(recordings.keys())[:max_recordings]
            recordings = {rid: recordings[rid] for rid in record_ids}
        
        processed_data = {}
        recording_features = []
        
        # HDF5 file for processed spectrograms
        yaseen_h5_path = self.preprocessed_dir / "yaseen_spectrograms.h5"
        
        with h5py.File(yaseen_h5_path, 'w') as h5f:
            
            for record_id, recording_info in tqdm(recordings.items(), desc="Processing Yaseen recordings"):
                try:
                    # Load audio
                    audio, sr = self.data_loader.load_audio_yaseen(record_id)
                    
                    # Check for corrupted audio
                    if not np.isfinite(audio).all():
                        self.logger.warning(f"Skipping {record_id}: contains non-finite values")
                        continue
                    
                    # Process to 224x224 spectrogram
                    spectrogram = self.audio_processor.process_yaseen_audio(
                        audio, sr, augment=False
                    )
                    
                    # Store in HDF5
                    h5f.create_dataset(record_id, data=spectrogram)
                    
                    # Store metadata
                    processed_data[record_id] = {
                        'metadata': {
                            'record_id': record_id,
                            'class_label': recording_info['class_label'],
                            'filename': recording_info['filename'],
                            'duration': recording_info.get('duration'),
                            'original_sr': recording_info['sample_rate']
                        },
                        'spectrogram': {
                            'shape': spectrogram.shape,
                            'h5_path': record_id
                        }
                    }
                    
                    # Collect features for CSV
                    recording_features.append({
                        'record_id': record_id,
                        'dataset': 'yaseen',
                        'class_label': recording_info['class_label'],
                        'filename': recording_info['filename'],
                        'duration': recording_info.get('duration', 0),
                        'original_sr': recording_info['sample_rate'],
                        'processed_sr': self.audio_processor.yaseen_sr
                    })
                
                except Exception as e:
                    self.logger.error(f"Error processing recording {record_id}: {e}")
                    continue
        
        # Save processed data info
        yaseen_info_path = self.preprocessed_dir / "yaseen_processed_info.json"
        with open(yaseen_info_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        # Save features CSV
        if recording_features:
            features_df = pd.DataFrame(recording_features)
            features_df.to_csv(self.features_dir / "yaseen_recording_features.csv", index=False)
        
        self.logger.info(f"Yaseen preprocessing completed. Processed {len(processed_data)} recordings")
        return processed_data
    
    def create_data_splits(self, circor_data: Dict = None, yaseen_data: Dict = None):
        """
        Create data splits following blueprint specifications:
        - CirCor: StratifiedGroupKFold by patient (5-fold)
        - Yaseen: StratifiedKFold by class (5-fold)
        """
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
        
        self.logger.info("Creating data splits...")
        
        splits = {}
        
        # CirCor splits (patient-level, stratified by murmur)
        if circor_data:
            patients = list(circor_data.keys())
            murmur_labels = [circor_data[p]['metadata']['murmur'] for p in patients]
            
            # Convert murmur labels to numeric
            murmur_map = {'Present': 0, 'Absent': 1, 'Unknown': 2}
            murmur_numeric = [murmur_map.get(m, 2) for m in murmur_labels]
            
            # 5-fold cross-validation
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            
            circor_splits = []
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(patients, murmur_numeric, patients)):
                train_patients = [patients[i] for i in train_idx]
                val_patients = [patients[i] for i in val_idx]
                
                circor_splits.append({
                    'fold': fold,
                    'train_patients': train_patients,
                    'val_patients': val_patients
                })
            
            splits['circor'] = circor_splits
        
        # Yaseen splits (recording-level, stratified by class)
        if yaseen_data:
            recordings = list(yaseen_data.keys())
            class_labels = [yaseen_data[r]['metadata']['class_label'] for r in recordings]
            
            # Convert class labels to numeric
            class_map = {'Normal': 0, 'AS': 1, 'MS': 2, 'MR': 3, 'MVP': 4}
            class_numeric = [class_map.get(c, 0) for c in class_labels]
            
            # 5-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            yaseen_splits = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(recordings, class_numeric)):
                train_recordings = [recordings[i] for i in train_idx]
                val_recordings = [recordings[i] for i in val_idx]
                
                yaseen_splits.append({
                    'fold': fold,
                    'train_recordings': train_recordings,
                    'val_recordings': val_recordings
                })
            
            splits['yaseen'] = yaseen_splits
        
        # Save splits
        splits_path = self.splits_dir / "data_splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        self.logger.info(f"Data splits saved to {splits_path}")
        return splits
    
    def run_full_preprocessing(self, 
                             max_circor_patients: Optional[int] = None,
                             max_yaseen_recordings: Optional[int] = None):
        """
        Run complete preprocessing pipeline for both datasets
        """
        self.logger.info("=== Starting Hybrid Preprocessing Pipeline ===")
        
        # Save processor config
        config_path = self.preprocessed_dir / "preprocessing_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.audio_processor.get_config(), f, indent=2)
        
        # Process datasets
        circor_data = None
        yaseen_data = None
        
        if self.data_loader.circor_path:
            circor_data = self.preprocess_circor_dataset(max_circor_patients)
        
        if self.data_loader.yaseen_path:
            yaseen_data = self.preprocess_yaseen_dataset(max_yaseen_recordings)
        
        # Create data splits
        splits = self.create_data_splits(circor_data, yaseen_data)
        
        # Save dataset statistics
        self.data_loader.save_dataset_info(str(self.preprocessed_dir))
        
        self.logger.info("=== Preprocessing pipeline completed successfully ===")
        
        return {
            'circor_data': circor_data,
            'yaseen_data': yaseen_data,
            'splits': splits,
            'output_dir': str(self.output_dir)
        }

def main():
    """Main preprocessing script"""
    parser = argparse.ArgumentParser(description='Hybrid PCG Preprocessing Pipeline')
    parser.add_argument('--circor_path', type=str, default='../training_data',
                       help='Path to CirCor dataset')
    parser.add_argument('--yaseen_path', type=str, default='../yaseen',
                       help='Path to Yaseen dataset')
    parser.add_argument('--output_dir', type=str, default='../data',
                       help='Output directory for processed data')
    parser.add_argument('--max_circor_patients', type=int, default=None,
                       help='Maximum number of CirCor patients to process (for testing)')
    parser.add_argument('--max_yaseen_recordings', type=int, default=None,
                       help='Maximum number of Yaseen recordings to process (for testing)')
    parser.add_argument('--use_dwt', action='store_true', default=True,
                       help='Enable DWT denoising')
    parser.add_argument('--yaseen_sr', type=int, default=4000, choices=[4000, 22050],
                       help='Target sampling rate for Yaseen dataset')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = HybridPreprocessor(
        circor_path=args.circor_path,
        yaseen_path=args.yaseen_path,
        output_dir=args.output_dir,
        use_dwt=args.use_dwt,
        yaseen_target_sr=args.yaseen_sr
    )
    
    # Run preprocessing
    results = preprocessor.run_full_preprocessing(
        max_circor_patients=args.max_circor_patients,
        max_yaseen_recordings=args.max_yaseen_recordings
    )
    
    print(f"\n=== Preprocessing Results ===")
    if results['circor_data']:
        print(f"CirCor: {len(results['circor_data'])} patients processed")
    if results['yaseen_data']:
        print(f"Yaseen: {len(results['yaseen_data'])} recordings processed")
    print(f"Output directory: {results['output_dir']}")

if __name__ == "__main__":
    main()
