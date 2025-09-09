"""
Data Loader for CirCor DigiScope Phonocardiogram Dataset
Handles loading audio files, annotations, and metadata
"""

import os
import pandas as pd
import numpy as np
import soundfile as sf
import wfdb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

class PCGDataLoader:
    """
    Data loader for CirCor DigiScope PCG dataset
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_config = config['dataset']
        self.audio_config = config['audio']
        
        self.source_path = Path(self.dataset_config['source_path'])
        self.metadata_file = Path(self.dataset_config['metadata_file'])
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Heart valve locations
        self.locations = ['AV', 'PV', 'TV', 'MV']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load patient metadata from CSV file"""
        try:
            metadata = pd.read_csv(self.metadata_file)
            self.logger.info(f"Loaded metadata for {len(metadata)} patients")
            return metadata
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            raise
    
    def get_patient_ids(self) -> List[str]:
        """Get list of all patient IDs"""
        return self.metadata['Patient ID'].astype(str).tolist()
    
    def get_patient_recordings(self, patient_id: str) -> Dict[str, Dict]:
        """
        Get all recordings for a specific patient
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary with recording locations as keys and recording info as values
        """
        recordings = {}
        
        # Check patient metadata
        patient_meta = self.metadata[self.metadata['Patient ID'] == int(patient_id)]
        if patient_meta.empty:
            self.logger.warning(f"No metadata found for patient {patient_id}")
            return recordings
        
        patient_row = patient_meta.iloc[0]
        available_locations = patient_row['Recording locations:'].split('+')
        
        for location in available_locations:
            location = location.strip()
            if location in self.locations:
                recording_info = self._load_recording(patient_id, location)
                if recording_info:
                    recordings[location] = recording_info
        
        return recordings
    
    def _load_recording(self, patient_id: str, location: str) -> Optional[Dict]:
        """
        Load a single recording (audio + annotations)
        
        Args:
            patient_id: Patient identifier
            location: Recording location (AV, PV, TV, MV)
            
        Returns:
            Dictionary containing audio data, annotations, and metadata
        """
        base_filename = f"{patient_id}_{location}"
        
        # File paths
        wav_file = self.source_path / f"{base_filename}.wav"
        hea_file = self.source_path / f"{base_filename}.hea"
        tsv_file = self.source_path / f"{base_filename}.tsv"
        
        if not wav_file.exists():
            self.logger.warning(f"Audio file not found: {wav_file}")
            return None
        
        try:
            # Load audio
            audio_data, sample_rate = sf.read(wav_file)
            
            # Load header information
            header_info = self._parse_header(hea_file) if hea_file.exists() else {}
            
            # Load annotations
            annotations = self._load_annotations(tsv_file) if tsv_file.exists() else []
            
            return {
                'audio': audio_data,
                'sample_rate': sample_rate,
                'header': header_info,
                'annotations': annotations,
                'location': location,
                'patient_id': patient_id,
                'file_path': str(wav_file)
            }
            
        except Exception as e:
            self.logger.error(f"Error loading recording {base_filename}: {e}")
            return None
    
    def _parse_header(self, hea_file: Path) -> Dict:
        """Parse WFDB header file"""
        try:
            header_info = {}
            with open(hea_file, 'r') as f:
                lines = f.readlines()
                
            if lines:
                # First line contains record info
                first_line = lines[0].strip().split()
                if len(first_line) >= 4:
                    header_info['record_name'] = first_line[0]
                    header_info['n_signals'] = int(first_line[1])
                    header_info['sampling_frequency'] = float(first_line[2])
                    header_info['n_samples'] = int(first_line[3])
                
                # Second line contains signal info
                if len(lines) > 1:
                    signal_line = lines[1].strip().split()
                    if len(signal_line) >= 8:
                        header_info['format'] = signal_line[1]
                        header_info['gain'] = float(signal_line[2])
                        header_info['baseline'] = int(signal_line[3])
                        header_info['units'] = signal_line[4]
                        header_info['description'] = signal_line[-1]
            
            return header_info
            
        except Exception as e:
            self.logger.error(f"Error parsing header {hea_file}: {e}")
            return {}
    
    def _load_annotations(self, tsv_file: Path) -> List[Dict]:
        """
        Load segmentation annotations from TSV file
        
        Args:
            tsv_file: Path to TSV annotation file
            
        Returns:
            List of annotation dictionaries
        """
        try:
            annotations = []
            
            with open(tsv_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        label = int(parts[2])
                        
                        # Convert label to semantic meaning
                        label_map = {
                            0: 'unlabeled',
                            1: 'S1',
                            2: 'systolic',
                            3: 'S2',
                            4: 'diastolic'
                        }
                        
                        annotations.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time,
                            'label_id': label,
                            'label_name': label_map.get(label, 'unknown')
                        })
            
            return annotations
            
        except Exception as e:
            self.logger.error(f"Error loading annotations {tsv_file}: {e}")
            return []
    
    def get_patient_metadata(self, patient_id: str) -> Dict:
        """Get metadata for a specific patient"""
        patient_meta = self.metadata[self.metadata['Patient ID'] == int(patient_id)]
        if patient_meta.empty:
            return {}
        
        patient_row = patient_meta.iloc[0]
        return {
            'patient_id': patient_id,
            'age': patient_row.get('Age', None),
            'sex': patient_row.get('Sex', None),
            'height': patient_row.get('Height', None),
            'weight': patient_row.get('Weight', None),
            'pregnancy_status': patient_row.get('Pregnancy status', None),
            'murmur': patient_row.get('Murmur', None),
            'murmur_locations': patient_row.get('Murmur locations', None),
            'outcome': patient_row.get('Outcome', None),
            'campaign': patient_row.get('Campaign', None),
            'recording_locations': patient_row.get('Recording locations:', None)
        }
    
    def get_all_recordings(self) -> List[Dict]:
        """Get all recordings from all patients"""
        all_recordings = []
        patient_ids = self.get_patient_ids()
        
        for patient_id in patient_ids:
            recordings = self.get_patient_recordings(patient_id)
            patient_metadata = self.get_patient_metadata(patient_id)
            
            for location, recording in recordings.items():
                recording['patient_metadata'] = patient_metadata
                all_recordings.append(recording)
        
        self.logger.info(f"Loaded {len(all_recordings)} recordings from {len(patient_ids)} patients")
        return all_recordings
    
    def get_dataset_statistics(self) -> Dict:
        """Get basic statistics about the dataset"""
        all_recordings = self.get_all_recordings()
        
        stats = {
            'total_patients': len(self.get_patient_ids()),
            'total_recordings': len(all_recordings),
            'locations_count': {},
            'duration_stats': {},
            'sample_rate_stats': {},
            'outcome_distribution': {}
        }
        
        # Location distribution
        for location in self.locations:
            count = sum(1 for r in all_recordings if r['location'] == location)
            stats['locations_count'][location] = count
        
        # Duration statistics
        durations = [len(r['audio']) / r['sample_rate'] for r in all_recordings]
        stats['duration_stats'] = {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations),
            'median': np.median(durations)
        }
        
        # Sample rate statistics
        sample_rates = [r['sample_rate'] for r in all_recordings]
        stats['sample_rate_stats'] = {
            'unique_rates': list(set(sample_rates)),
            'most_common': max(set(sample_rates), key=sample_rates.count)
        }
        
        # Outcome distribution
        outcomes = [r['patient_metadata']['outcome'] for r in all_recordings if r['patient_metadata'].get('outcome')]
        for outcome in set(outcomes):
            stats['outcome_distribution'][outcome] = outcomes.count(outcome)
        
        return stats
