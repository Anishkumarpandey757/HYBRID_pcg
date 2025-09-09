"""
Audio Preprocessing Module for PCG Signals
Handles filtering, normalization, segmentation, and augmentation
"""

import numpy as np
import scipy.signal
import librosa
from typing import Dict, List, Tuple, Optional
import logging

class PCGPreprocessor:
    """
    Preprocessor for phonocardiogram signals
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.audio_config = config['audio']
        self.segmentation_config = config.get('segmentation', {})
        self.augmentation_config = config.get('augmentation', {})
        
        self.target_sr = self.audio_config['sampling_rate']
        self.target_duration = self.audio_config['target_duration']
        
        self.logger = logging.getLogger(__name__)
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Apply complete preprocessing pipeline to audio signal
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Preprocessed audio signal and sample rate
        """
        # Resample to target sample rate
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Apply filters
        audio = self._apply_filters(audio, sr)
        
        # Normalize
        if self.audio_config.get('normalize', True):
            audio = self._normalize_audio(audio)
        
        # Trim or pad to target duration
        audio = self._adjust_duration(audio, sr)
        
        return audio, sr
    
    def _apply_filters(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply bandpass filtering"""
        filter_config = self.audio_config.get('filter', {})
        
        if not filter_config:
            return audio
        
        try:
            low_pass = filter_config.get('low_pass')
            high_pass = filter_config.get('high_pass')
            order = filter_config.get('order', 5)
            
            # Design bandpass filter
            nyquist = sr / 2
            
            if high_pass and low_pass:
                # Bandpass filter
                low = high_pass / nyquist
                high = low_pass / nyquist
                b, a = scipy.signal.butter(order, [low, high], btype='band')
            elif high_pass:
                # High-pass filter
                high = high_pass / nyquist
                b, a = scipy.signal.butter(order, high, btype='high')
            elif low_pass:
                # Low-pass filter
                low = low_pass / nyquist
                b, a = scipy.signal.butter(order, low, btype='low')
            else:
                return audio
            
            # Apply filter
            filtered_audio = scipy.signal.filtfilt(b, a, audio)
            return filtered_audio
            
        except Exception as e:
            self.logger.warning(f"Error applying filters: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio signal"""
        # Remove DC component
        audio = audio - np.mean(audio)
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def _adjust_duration(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Adjust audio to target duration"""
        target_samples = int(self.target_duration * sr)
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        elif current_samples < target_samples:
            # Pad with zeros
            padding = target_samples - current_samples
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            # Trim to target length
            audio = audio[:target_samples]
        
        return audio
    
    def segment_audio(self, audio: np.ndarray, sr: int, 
                     annotations: List[Dict]) -> List[Dict]:
        """
        Segment audio based on annotations
        
        Args:
            audio: Audio signal
            sr: Sample rate
            annotations: List of annotation dictionaries
            
        Returns:
            List of audio segments with metadata
        """
        segments = []
        
        if not self.segmentation_config.get('use_annotations', True):
            # Return full audio as single segment
            segments.append({
                'audio': audio,
                'start_time': 0.0,
                'end_time': len(audio) / sr,
                'duration': len(audio) / sr,
                'label_name': 'full',
                'label_id': -1
            })
            return segments
        
        min_duration = self.segmentation_config.get('min_segment_duration', 0.1)
        segment_types = self.segmentation_config.get('segment_types', [])
        
        for annotation in annotations:
            # Check if this segment type should be extracted
            if segment_types and annotation['label_name'] not in segment_types:
                continue
            
            # Check minimum duration
            if annotation['duration'] < min_duration:
                continue
            
            # Extract segment
            start_sample = int(annotation['start_time'] * sr)
            end_sample = int(annotation['end_time'] * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                segment_audio = audio[start_sample:end_sample]
                
                segments.append({
                    'audio': segment_audio,
                    'start_time': annotation['start_time'],
                    'end_time': annotation['end_time'],
                    'duration': annotation['duration'],
                    'label_name': annotation['label_name'],
                    'label_id': annotation['label_id']
                })
        
        return segments
    
    def augment_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Apply data augmentation to audio signal
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            List of augmented audio signals (including original)
        """
        augmented_signals = [audio.copy()]  # Include original
        
        if not self.augmentation_config.get('enabled', False):
            return augmented_signals
        
        methods = self.augmentation_config.get('methods', {})
        
        # Time stretching
        if methods.get('time_stretch', {}).get('enabled', False):
            rate_range = methods['time_stretch']['rate_range']
            for rate in np.linspace(rate_range[0], rate_range[1], 3):
                if rate != 1.0:  # Skip original rate
                    stretched = librosa.effects.time_stretch(audio, rate=rate)
                    # Adjust to original length
                    stretched = self._adjust_duration(stretched, sr)
                    augmented_signals.append(stretched)
        
        # Pitch shifting
        if methods.get('pitch_shift', {}).get('enabled', False):
            steps_range = methods['pitch_shift']['steps_range']
            for steps in range(steps_range[0], steps_range[1] + 1):
                if steps != 0:  # Skip original pitch
                    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
                    augmented_signals.append(shifted)
        
        # Noise addition
        if methods.get('noise_addition', {}).get('enabled', False):
            noise_factor = methods['noise_addition']['noise_factor']
            noise = np.random.normal(0, noise_factor * np.std(audio), len(audio))
            noisy_audio = audio + noise
            # Ensure it stays in valid range
            noisy_audio = np.clip(noisy_audio, -1, 1)
            augmented_signals.append(noisy_audio)
        
        # Time shifting
        if methods.get('time_shift', {}).get('enabled', False):
            shift_range = methods['time_shift']['shift_range']
            max_shift_samples = int(max(abs(shift_range[0]), abs(shift_range[1])) * sr)
            
            for shift_factor in [shift_range[0], shift_range[1]]:
                if shift_factor != 0:
                    shift_samples = int(shift_factor * sr)
                    
                    if shift_samples > 0:
                        # Shift right
                        shifted = np.pad(audio, (shift_samples, 0), mode='constant')[:-shift_samples]
                    else:
                        # Shift left
                        shifted = np.pad(audio, (0, -shift_samples), mode='constant')[-shift_samples:]
                    
                    augmented_signals.append(shifted)
        
        return augmented_signals
    
    def extract_heart_cycles(self, audio: np.ndarray, sr: int, 
                           annotations: List[Dict]) -> List[Dict]:
        """
        Extract complete heart cycles (S1 to next S1)
        
        Args:
            audio: Audio signal
            sr: Sample rate
            annotations: List of annotation dictionaries
            
        Returns:
            List of heart cycle segments
        """
        cycles = []
        
        # Find S1 beats
        s1_annotations = [ann for ann in annotations if ann['label_name'] == 'S1']
        s1_annotations = sorted(s1_annotations, key=lambda x: x['start_time'])
        
        # Extract cycles between consecutive S1s
        for i in range(len(s1_annotations) - 1):
            cycle_start = s1_annotations[i]['start_time']
            cycle_end = s1_annotations[i + 1]['start_time']
            cycle_duration = cycle_end - cycle_start
            
            # Skip very short or very long cycles
            if 0.4 <= cycle_duration <= 1.5:  # Reasonable heart cycle duration
                start_sample = int(cycle_start * sr)
                end_sample = int(cycle_end * sr)
                
                if start_sample < len(audio) and end_sample <= len(audio):
                    cycle_audio = audio[start_sample:end_sample]
                    
                    cycles.append({
                        'audio': cycle_audio,
                        'start_time': cycle_start,
                        'end_time': cycle_end,
                        'duration': cycle_duration,
                        'label_name': 'heart_cycle',
                        'label_id': 99,  # Special ID for heart cycles
                        'cycle_index': i
                    })
        
        return cycles
    
    def detect_envelope_peaks(self, audio: np.ndarray, sr: int) -> List[float]:
        """
        Detect envelope peaks which may correspond to heart sounds
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of peak times in seconds
        """
        # Calculate envelope
        envelope = np.abs(scipy.signal.hilbert(audio))
        
        # Smooth envelope
        window_size = int(0.02 * sr)  # 20ms window
        envelope_smooth = scipy.signal.savgol_filter(envelope, window_size, 3)
        
        # Find peaks
        min_distance = int(0.2 * sr)  # Minimum 200ms between peaks
        height_threshold = np.mean(envelope_smooth) + 2 * np.std(envelope_smooth)
        
        peaks, _ = scipy.signal.find_peaks(
            envelope_smooth,
            height=height_threshold,
            distance=min_distance
        )
        
        # Convert to time
        peak_times = peaks / sr
        
        return peak_times.tolist()
    
    def quality_check(self, audio: np.ndarray, sr: int) -> Dict[str, bool]:
        """
        Perform quality checks on audio signal
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of quality check results
        """
        checks = {}
        
        # Check for clipping
        clipping_threshold = 0.99
        checks['no_clipping'] = np.max(np.abs(audio)) < clipping_threshold
        
        # Check for silence
        rms = np.sqrt(np.mean(audio**2))
        silence_threshold = 0.001
        checks['not_silent'] = rms > silence_threshold
        
        # Check for reasonable dynamic range
        dynamic_range = np.max(audio) - np.min(audio)
        checks['good_dynamic_range'] = dynamic_range > 0.1
        
        # Check for reasonable frequency content
        freqs, psd = scipy.signal.welch(audio, sr, nperseg=min(len(audio), 1024))
        relevant_freq_range = (freqs >= 20) & (freqs <= 1000)  # Heart sound frequency range
        energy_in_range = np.sum(psd[relevant_freq_range])
        total_energy = np.sum(psd)
        
        checks['good_frequency_content'] = (energy_in_range / total_energy) > 0.5
        
        # Overall quality
        checks['overall_quality'] = all(checks.values())
        
        return checks
