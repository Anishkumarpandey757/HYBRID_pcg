"""
Hybrid Data Loader for CirCor and Yaseen datasets
Follows the exact blueprint specifications
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import wfdb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DatasetConfig:
    """Configuration for each dataset"""
    name: str
    sample_rate: int
    clip_length: float  # in seconds
    task_type: str  # 'multi_task' or 'single_task'
    classes: List[str]
    locations: Optional[List[str]] = None

# Dataset configurations following the blueprint
CIRCOR_CONFIG = DatasetConfig(
    name="circor",
    sample_rate=4000,  # Will read from .hea files
    clip_length=12.0,  # 12-12.5s
    task_type="multi_task",
    classes=["Present", "Absent", "Unknown"],  # Murmur classes
    locations=["AV", "MV", "PV", "TV", "Phc"]
)

YASEEN_CONFIG = DatasetConfig(
    name="yaseen", 
    sample_rate=22050,  # Native or downsample to 4kHz
    clip_length=3.0,   # 3s clips
    task_type="single_task",
    classes=["Normal", "AS", "MS", "MR", "MVP"]  # 5-class disease classification
)

class HybridDataLoader:
    """
    Unified data loader for both CirCor and Yaseen datasets
    Implements the exact preprocessing from the hybrid blueprint
    """
    
    def __init__(self, circor_path: str = None, yaseen_path: str = None):
        self.circor_path = Path(circor_path) if circor_path else None
        self.yaseen_path = Path(yaseen_path) if yaseen_path else None
        
        # Initialize dataset info
        self.circor_data = {}
        self.yaseen_data = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_circor_dataset(self) -> Dict:
        """
        Load CirCor dataset following blueprint specifications:
        - Group by patient
        - Collect sites {AV, MV, PV, TV, Phc} per patient
        - Multi-task labels (murmur + outcome)
        """
        if not self.circor_path or not self.circor_path.exists():
            self.logger.error(f"CirCor path not found: {self.circor_path}")
            return {}
            
        self.logger.info("Loading CirCor dataset...")
        
        # Load metadata - check multiple possible locations
        metadata_file = self.circor_path / "training_data.csv"
        if not metadata_file.exists():
            # Try parent directory (common case)
            metadata_file = self.circor_path.parent / "training_data.csv"
        if not metadata_file.exists():
            # Try root level path
            metadata_file = Path("../training_data.csv")
        
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
            self.logger.info(f"Found metadata file: {metadata_file}")
        else:
            self.logger.error("training_data.csv not found in any expected location")
            self.logger.error(f"Tried: {self.circor_path}/training_data.csv")
            self.logger.error(f"Tried: {self.circor_path.parent}/training_data.csv") 
            self.logger.error("Tried: ../training_data.csv")
            return {}
        
        training_data_dir = self.circor_path / "training_data"
        
        # Group by patient
        patients = {}
        
        for _, row in metadata.iterrows():
            patient_id = str(row['Patient ID'])
            
            if patient_id not in patients:
                patients[patient_id] = {
                    'patient_id': patient_id,
                    'age': row.get('Age', None),
                    'sex': row.get('Sex', None),
                    'height': row.get('Height', None),
                    'weight': row.get('Weight', None),
                    'pregnancy_status': row.get('Pregnancy status', None),
                    'murmur': row.get('Murmur', 'Unknown'),
                    'outcome': row.get('Outcome', 'Unknown'),
                    'campaign': row.get('Campaign', None),
                    'recordings': {}
                }
            
            # Find all recordings for this patient
            patient_files = list(training_data_dir.glob(f"{patient_id}_*.wav"))
            
            for audio_file in patient_files:
                # Extract location from filename
                filename = audio_file.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    location = parts[1]  # AV, MV, PV, TV, etc.
                    
                    # Read sampling rate from .hea file
                    hea_file = audio_file.with_suffix('.hea')
                    sample_rate = 4000  # default
                    if hea_file.exists():
                        try:
                            record = wfdb.rdheader(str(hea_file.with_suffix('')))
                            sample_rate = record.fs
                        except:
                            pass
                    
                    # Read TSV annotations if available
                    tsv_file = audio_file.with_suffix('.tsv')
                    annotations = []
                    if tsv_file.exists():
                        try:
                            ann_df = pd.read_csv(tsv_file, sep='\t')
                            for _, ann_row in ann_df.iterrows():
                                annotations.append({
                                    'start_time': ann_row.get('Start', 0),
                                    'end_time': ann_row.get('End', 0),
                                    'label_name': ann_row.get('Label', 'unlabeled')
                                })
                        except:
                            pass
                    
                    patients[patient_id]['recordings'][location] = {
                        'audio_file': str(audio_file),
                        'sample_rate': sample_rate,
                        'annotations': annotations,
                        'location': location
                    }
        
        self.circor_data = patients
        self.logger.info(f"Loaded {len(patients)} CirCor patients")
        
        return patients
    
    def load_yaseen_dataset(self) -> Dict:
        """
        Load Yaseen dataset following blueprint specifications:
        - Single-recording per file
        - 5-class classification {Normal, AS, MS, MR, MVP}
        - Per file processing
        """
        if not self.yaseen_path or not self.yaseen_path.exists():
            self.logger.error(f"Yaseen path not found: {self.yaseen_path}")
            return {}
            
        self.logger.info("Loading Yaseen dataset...")
        
        # Class mapping from folder names
        class_mapping = {
            'N_New_3주기': 'Normal',
            'AS_New_3주기': 'AS',
            'MS_New_3주기': 'MS', 
            'MR_New_3주기': 'MR',
            'MVP_New_3주기': 'MVP'
        }
        
        recordings = {}
        record_id = 0
        
        for folder_name, class_label in class_mapping.items():
            class_folder = self.yaseen_path / folder_name
            
            if not class_folder.exists():
                self.logger.warning(f"Class folder not found: {class_folder}")
                continue
                
            # Get all WAV files in this class folder
            wav_files = list(class_folder.glob("*.wav"))
            
            for wav_file in wav_files:
                # Get audio info
                try:
                    info = sf.info(str(wav_file))
                    sample_rate = info.samplerate
                    duration = info.duration
                    
                    recordings[f"yaseen_{record_id:04d}"] = {
                        'record_id': f"yaseen_{record_id:04d}",
                        'audio_file': str(wav_file),
                        'sample_rate': sample_rate,
                        'duration': duration,
                        'class_label': class_label,
                        'filename': wav_file.name
                    }
                    record_id += 1
                    
                except Exception as e:
                    self.logger.warning(f"Could not read {wav_file}: {e}")
                    continue
        
        self.yaseen_data = recordings
        self.logger.info(f"Loaded {len(recordings)} Yaseen recordings")
        
        return recordings
    
    def get_circor_patient_recordings(self, patient_id: str) -> Dict:
        """Get all recordings for a CirCor patient"""
        return self.circor_data.get(patient_id, {}).get('recordings', {})
    
    def get_yaseen_recording(self, record_id: str) -> Dict:
        """Get a single Yaseen recording"""
        return self.yaseen_data.get(record_id, {})
    
    def load_audio_circor(self, patient_id: str, location: str) -> Tuple[np.ndarray, int]:
        """Load audio for CirCor recording"""
        recordings = self.get_circor_patient_recordings(patient_id)
        
        if location not in recordings:
            raise ValueError(f"Location {location} not found for patient {patient_id}")
            
        audio_file = recordings[location]['audio_file']
        sample_rate = recordings[location]['sample_rate']
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=None)
        
        return audio, sr
    
    def load_audio_yaseen(self, record_id: str) -> Tuple[np.ndarray, int]:
        """Load audio for Yaseen recording"""
        recording = self.get_yaseen_recording(record_id)
        
        if not recording:
            raise ValueError(f"Recording {record_id} not found")
            
        audio_file = recording['audio_file']
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=None)
        
        return audio, sr
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive statistics for both datasets"""
        stats = {
            'circor': {
                'num_patients': len(self.circor_data),
                'num_recordings': sum(len(p['recordings']) for p in self.circor_data.values()),
                'locations': {},
                'murmur_distribution': {},
                'outcome_distribution': {},
                'campaign_distribution': {}
            },
            'yaseen': {
                'num_recordings': len(self.yaseen_data),
                'class_distribution': {},
                'sample_rates': {},
                'durations': []
            }
        }
        
        # CirCor statistics
        for patient in self.circor_data.values():
            # Count locations
            for location in patient['recordings'].keys():
                stats['circor']['locations'][location] = stats['circor']['locations'].get(location, 0) + 1
            
            # Count murmur labels
            murmur = patient.get('murmur', 'Unknown')
            stats['circor']['murmur_distribution'][murmur] = stats['circor']['murmur_distribution'].get(murmur, 0) + 1
            
            # Count outcomes
            outcome = patient.get('outcome', 'Unknown')
            stats['circor']['outcome_distribution'][outcome] = stats['circor']['outcome_distribution'].get(outcome, 0) + 1
            
            # Count campaigns
            campaign = patient.get('campaign', 'Unknown')
            stats['circor']['campaign_distribution'][campaign] = stats['circor']['campaign_distribution'].get(campaign, 0) + 1
        
        # Yaseen statistics
        for recording in self.yaseen_data.values():
            # Count classes
            class_label = recording['class_label']
            stats['yaseen']['class_distribution'][class_label] = stats['yaseen']['class_distribution'].get(class_label, 0) + 1
            
            # Count sample rates
            sr = recording['sample_rate']
            stats['yaseen']['sample_rates'][sr] = stats['yaseen']['sample_rates'].get(sr, 0) + 1
            
            # Collect durations
            stats['yaseen']['durations'].append(recording.get('duration', 0))
        
        return stats
    
    def save_dataset_info(self, output_dir: str):
        """Save dataset information to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save CirCor patient info
        if self.circor_data:
            circor_file = output_path / 'circor_patients.json'
            with open(circor_file, 'w') as f:
                json.dump(self.circor_data, f, indent=2)
        
        # Save Yaseen recording info
        if self.yaseen_data:
            yaseen_file = output_path / 'yaseen_recordings.json'
            with open(yaseen_file, 'w') as f:
                json.dump(self.yaseen_data, f, indent=2)
        
        # Save statistics
        stats = self.get_dataset_statistics()
        stats_file = output_path / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Dataset info saved to {output_dir}")

def main():
    """Test the hybrid data loader"""
    # Paths
    circor_path = "../training_data"
    yaseen_path = "../yaseen" 
    
    # Initialize loader
    loader = HybridDataLoader(circor_path, yaseen_path)
    
    # Load datasets
    circor_patients = loader.load_circor_dataset()
    yaseen_recordings = loader.load_yaseen_dataset()
    
    # Print statistics
    stats = loader.get_dataset_statistics()
    print("=== Dataset Statistics ===")
    print(f"CirCor: {stats['circor']['num_patients']} patients, {stats['circor']['num_recordings']} recordings")
    print(f"Yaseen: {stats['yaseen']['num_recordings']} recordings")
    
    # Save dataset info
    loader.save_dataset_info("../data/dataset_info")

if __name__ == "__main__":
    main()
