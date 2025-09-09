"""
Feature Extraction Module for PCG Analysis
Extracts various audio features for machine learning
"""

import numpy as np
import librosa
import scipy.signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional
import logging

class PCGFeatureExtractor:
    """
    Feature extractor for phonocardiogram signals
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.audio_config = config['audio']
        self.features_config = config['features']
        
        self.target_sr = self.audio_config['sampling_rate']
        
        self.logger = logging.getLogger(__name__)
    
    def extract_all_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract all configured features from audio signal
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Resample if necessary
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        try:
            # MFCC features
            if self.features_config.get('extract_mfcc', False):
                features.update(self._extract_mfcc(audio, sr))
            
            # Mel spectrogram
            if self.features_config.get('extract_mel_spectrogram', False):
                features.update(self._extract_mel_spectrogram(audio, sr))
            
            # Spectral features
            if self.features_config.get('extract_spectral', False):
                features.update(self._extract_spectral_features(audio, sr))
            
            # Temporal features
            if self.features_config.get('extract_temporal', False):
                features.update(self._extract_temporal_features(audio, sr))
            
            # Wavelet features
            features.update(self._extract_wavelet_features(audio))
            
            # Statistical features
            features.update(self._extract_statistical_features(audio))
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            
        return features
    
    def _extract_mfcc(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract MFCC features"""
        mfcc_config = self.features_config['mfcc']
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=mfcc_config['n_mfcc'],
            n_fft=mfcc_config['n_fft'],
            hop_length=mfcc_config['hop_length']
        )
        
        # Calculate derivatives
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        return {
            'mfcc': mfcc,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1),
            'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
            'mfcc_delta_std': np.std(mfcc_delta, axis=1)
        }
    
    def _extract_mel_spectrogram(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract mel spectrogram features"""
        mel_config = self.features_config['mel_spectrogram']
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=mel_config['n_mels'],
            n_fft=mel_config['n_fft'],
            hop_length=mel_config['hop_length']
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            'mel_spectrogram': mel_spec,
            'log_mel_spectrogram': log_mel_spec,
            'mel_spec_mean': np.mean(log_mel_spec, axis=1),
            'mel_spec_std': np.std(log_mel_spec, axis=1)
        }
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract spectral features"""
        features = {}
        
        # Spectral centroid
        if 'spectral_centroid' in self.features_config['spectral']:
            cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spectral_centroid'] = cent
            features['spectral_centroid_mean'] = np.mean(cent)
            features['spectral_centroid_std'] = np.std(cent)
        
        # Spectral rolloff
        if 'spectral_rolloff' in self.features_config['spectral']:
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['spectral_rolloff'] = rolloff
            features['spectral_rolloff_mean'] = np.mean(rolloff)
            features['spectral_rolloff_std'] = np.std(rolloff)
        
        # Spectral contrast
        if 'spectral_contrast' in self.features_config['spectral']:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['spectral_contrast'] = contrast
            features['spectral_contrast_mean'] = np.mean(contrast, axis=1)
            features['spectral_contrast_std'] = np.std(contrast, axis=1)
        
        # Zero crossing rate
        if 'zero_crossing_rate' in self.features_config['spectral']:
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zero_crossing_rate'] = zcr
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features['spectral_bandwidth'] = bandwidth
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        features['spectral_bandwidth_std'] = np.std(bandwidth)
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio)
        features['spectral_flatness'] = flatness
        features['spectral_flatness_mean'] = np.mean(flatness)
        features['spectral_flatness_std'] = np.std(flatness)
        
        return features
    
    def _extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract temporal features"""
        features = {}
        
        # RMS energy
        if 'rms' in self.features_config['temporal']:
            rms = librosa.feature.rms(y=audio)
            features['rms'] = rms
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
        
        # Tempo
        if 'tempo' in self.features_config['temporal']:
            try:
                tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
                features['tempo'] = tempo
                features['beat_times'] = beats
            except:
                features['tempo'] = 0.0
                features['beat_times'] = np.array([])
        
        # Envelope
        envelope = np.abs(scipy.signal.hilbert(audio))
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_max'] = np.max(envelope)
        
        # Peak detection
        peaks, _ = scipy.signal.find_peaks(audio, height=np.std(audio))
        features['peak_count'] = len(peaks)
        features['peak_rate'] = len(peaks) / (len(audio) / sr)
        
        return features
    
    def _extract_wavelet_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract wavelet-based features"""
        try:
            import pywt
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(audio, 'db4', level=4)
            
            features = {}
            for i, coeff in enumerate(coeffs):
                features[f'wavelet_level_{i}_mean'] = np.mean(coeff)
                features[f'wavelet_level_{i}_std'] = np.std(coeff)
                features[f'wavelet_level_{i}_energy'] = np.sum(coeff**2)
            
            return features
            
        except ImportError:
            self.logger.warning("PyWavelets not available, skipping wavelet features")
            return {}
    
    def _extract_statistical_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract statistical features"""
        features = {
            'signal_mean': np.mean(audio),
            'signal_std': np.std(audio),
            'signal_var': np.var(audio),
            'signal_skewness': skew(audio),
            'signal_kurtosis': kurtosis(audio),
            'signal_min': np.min(audio),
            'signal_max': np.max(audio),
            'signal_range': np.max(audio) - np.min(audio),
            'signal_energy': np.sum(audio**2),
            'signal_power': np.mean(audio**2),
            'signal_rms': np.sqrt(np.mean(audio**2))
        }
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            features[f'signal_percentile_{p}'] = np.percentile(audio, p)
        
        return features
    
    def extract_segment_features(self, audio: np.ndarray, sr: int, 
                               annotations: List[Dict]) -> List[Dict]:
        """
        Extract features for each annotated segment
        
        Args:
            audio: Full audio signal
            sr: Sample rate
            annotations: List of annotation dictionaries
            
        Returns:
            List of feature dictionaries for each segment
        """
        segment_features = []
        
        for annotation in annotations:
            start_sample = int(annotation['start_time'] * sr)
            end_sample = int(annotation['end_time'] * sr)
            
            # Extract segment
            segment = audio[start_sample:end_sample]
            
            if len(segment) > 0:
                # Extract features for this segment
                features = self.extract_all_features(segment, sr)
                
                # Add segment metadata
                features['segment_start'] = annotation['start_time']
                features['segment_end'] = annotation['end_time']
                features['segment_duration'] = annotation['duration']
                features['segment_label'] = annotation['label_name']
                features['segment_label_id'] = annotation['label_id']
                
                segment_features.append(features)
        
        return segment_features
    
    def create_feature_matrix(self, feature_list: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix from list of feature dictionaries
        
        Args:
            feature_list: List of feature dictionaries
            
        Returns:
            Feature matrix and feature names
        """
        if not feature_list:
            return np.array([]), []
        
        # Get all possible feature names
        all_features = set()
        for features in feature_list:
            all_features.update(features.keys())
        
        # Filter to numeric features only
        numeric_features = []
        for feature_name in all_features:
            sample_value = feature_list[0].get(feature_name)
            if isinstance(sample_value, (int, float, np.number)):
                numeric_features.append(feature_name)
        
        numeric_features = sorted(numeric_features)
        
        # Create feature matrix
        feature_matrix = np.zeros((len(feature_list), len(numeric_features)))
        
        for i, features in enumerate(feature_list):
            for j, feature_name in enumerate(numeric_features):
                value = features.get(feature_name, 0)
                if isinstance(value, (list, np.ndarray)):
                    value = np.mean(value) if len(value) > 0 else 0
                feature_matrix[i, j] = float(value)
        
        return feature_matrix, numeric_features
