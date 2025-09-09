"""
Hybrid Audio Processor implementing the exact blueprint specifications
Shared front-end for both CirCor and Yaseen datasets
"""

import numpy as np
import librosa
import cv2
from scipy import signal
import pywt
from typing import Tuple, Optional, Dict, Any
import logging
import warnings
warnings.filterwarnings('ignore')

class HybridAudioProcessor:
    """
    Implements the exact preprocessing pipeline from the hybrid blueprint:
    
    Shared front-end (both datasets):
    • Resample: CirCor ≈ 4 kHz from .hea; Yaseen 22.05 kHz or downsample to 4kHz
    • Window: fixed clip length (CirCor 12–12.5 s; Yaseen 3 s). Pad or crop as needed
    • (Optional) DWT denoise: db4, level-5; soft-threshold detail coeffs (BayesShrink/VisuShrink)
    • Time–frequency: 128-mel, 25 ms win / 10 ms hop → log-mel → per-sample z-score
    • Resize/crop to 224×224 for PaSST/Vision backbones
    • Aug (train-only): Waveform + SpecAugment
    """
    
    def __init__(self, 
                 circor_sr: int = 4000,
                 yaseen_sr: int = 4000,  # Can be 22050 or downsample to 4000
                 circor_clip_length: float = 12.0,
                 yaseen_clip_length: float = 3.0,
                 mel_bins: int = 128,
                 win_length_ms: float = 25.0,
                 hop_length_ms: float = 10.0,
                 target_size: int = 224,
                 use_dwt: bool = True,
                 dwt_wavelet: str = 'db4',
                 dwt_levels: int = 5):
        
        self.circor_sr = circor_sr
        self.yaseen_sr = yaseen_sr
        self.circor_clip_length = circor_clip_length
        self.yaseen_clip_length = yaseen_clip_length
        self.mel_bins = mel_bins
        self.target_size = target_size
        self.use_dwt = use_dwt
        self.dwt_wavelet = dwt_wavelet
        self.dwt_levels = dwt_levels
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Calculate window and hop lengths in samples for each dataset
        self.circor_win_length = int(win_length_ms * circor_sr / 1000)
        self.circor_hop_length = int(hop_length_ms * circor_sr / 1000)
        
        self.yaseen_win_length = int(win_length_ms * yaseen_sr / 1000)
        self.yaseen_hop_length = int(hop_length_ms * yaseen_sr / 1000)
        
        # Target lengths in samples
        self.circor_target_length = int(circor_clip_length * circor_sr)
        self.yaseen_target_length = int(yaseen_clip_length * yaseen_sr)
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sampling rate"""
        # Check for invalid audio data
        if not np.isfinite(audio).all():
            self.logger.warning("Audio contains non-finite values, cleaning...")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return audio
    
    def window_audio(self, audio: np.ndarray, target_length: int, dataset_type: str = "circor") -> np.ndarray:
        """
        Window audio to fixed clip length. Pad or crop as needed.
        CirCor: 12-12.5s, Yaseen: 3s
        """
        current_length = len(audio)
        
        if current_length == target_length:
            return audio
        elif current_length < target_length:
            # Pad with zeros (can also use reflection padding)
            pad_length = target_length - current_length
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
        else:
            # Crop from center or random location
            start_idx = (current_length - target_length) // 2
            audio = audio[start_idx:start_idx + target_length]
        
        return audio
    
    def dwt_denoise(self, audio: np.ndarray, threshold_mode: str = 'bayes') -> np.ndarray:
        """
        DWT denoising following blueprint specifications:
        - db4 wavelet, level-5
        - Soft-threshold detail coefficients (BayesShrink/VisuShrink)
        """
        if not self.use_dwt:
            return audio
        
        # Decompose signal
        coeffs = pywt.wavedec(audio, self.dwt_wavelet, level=self.dwt_levels)
        
        # Estimate noise level (sigma) from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Apply thresholding to detail coefficients
        coeffs_thresh = list(coeffs)
        
        for i in range(1, len(coeffs)):  # Skip approximation coefficients
            if threshold_mode == 'bayes':
                # BayesShrink threshold
                detail_var = np.var(coeffs[i])
                if detail_var > sigma**2:
                    threshold = sigma**2 / np.sqrt(detail_var)
                else:
                    threshold = sigma
            else:
                # VisuShrink threshold
                threshold = sigma * np.sqrt(2 * np.log(len(audio)))
            
            # Soft thresholding
            coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        
        # Reconstruct signal
        audio_denoised = pywt.waverec(coeffs_thresh, self.dwt_wavelet)
        
        # Ensure same length as input
        if len(audio_denoised) != len(audio):
            audio_denoised = audio_denoised[:len(audio)]
        
        return audio_denoised
    
    def compute_mel_spectrogram(self, audio: np.ndarray, sr: int, dataset_type: str = "circor") -> np.ndarray:
        """
        Compute mel spectrogram following blueprint:
        - 128-mel bins
        - 25 ms window / 10 ms hop
        - Log-mel transformation
        """
        if dataset_type == "circor":
            win_length = self.circor_win_length
            hop_length = self.circor_hop_length
        else:
            win_length = self.yaseen_win_length
            hop_length = self.yaseen_hop_length
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.mel_bins,
            n_fft=2048,
            win_length=win_length,
            hop_length=hop_length,
            window='hann',
            fmin=0,
            fmax=sr // 2
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel
    
    def normalize_spectrogram(self, log_mel: np.ndarray) -> np.ndarray:
        """
        Per-sample z-score normalization as specified in blueprint
        """
        mean = np.mean(log_mel)
        std = np.std(log_mel)
        
        if std > 0:
            log_mel_norm = (log_mel - mean) / std
        else:
            log_mel_norm = log_mel - mean
        
        return log_mel_norm
    
    def resize_spectrogram(self, spectrogram: np.ndarray, target_size: int = 224) -> np.ndarray:
        """
        Resize/crop spectrogram to 224×224 for PaSST/Vision backbones
        """
        h, w = spectrogram.shape
        
        # Resize using OpenCV
        resized = cv2.resize(spectrogram.astype(np.float32), 
                           (target_size, target_size), 
                           interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def apply_waveform_augmentation(self, audio: np.ndarray, sr: int, augment: bool = False) -> np.ndarray:
        """
        Apply waveform augmentations (train-only):
        - Small Gaussian noise
        - ±10% time-stretch
        - Random time-shift
        """
        if not augment:
            return audio
        
        # Gaussian noise
        noise_factor = np.random.uniform(0.001, 0.01)
        audio = audio + noise_factor * np.random.randn(len(audio))
        
        # Time stretching (±10%)
        stretch_factor = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
        
        # Random time shift
        shift_samples = int(np.random.uniform(-0.1, 0.1) * len(audio))
        if shift_samples > 0:
            audio = np.pad(audio, (shift_samples, 0), mode='constant')[:-shift_samples]
        elif shift_samples < 0:
            audio = np.pad(audio, (0, -shift_samples), mode='constant')[-shift_samples:]
        
        return audio
    
    def apply_spec_augment(self, spectrogram: np.ndarray, augment: bool = False) -> np.ndarray:
        """
        Apply SpecAugment (train-only):
        - 1-2 time masks & 1-2 freq masks (≤10% each)
        """
        if not augment:
            return spectrogram
        
        spec_aug = spectrogram.copy()
        h, w = spec_aug.shape
        
        # Time masks (1-2 masks, ≤10% each)
        num_time_masks = np.random.randint(1, 3)
        for _ in range(num_time_masks):
            mask_width = int(np.random.uniform(0.01, 0.1) * w)
            mask_start = np.random.randint(0, max(1, w - mask_width))
            spec_aug[:, mask_start:mask_start + mask_width] = 0
        
        # Frequency masks (1-2 masks, ≤10% each)
        num_freq_masks = np.random.randint(1, 3)
        for _ in range(num_freq_masks):
            mask_height = int(np.random.uniform(0.01, 0.1) * h)
            mask_start = np.random.randint(0, max(1, h - mask_height))
            spec_aug[mask_start:mask_start + mask_height, :] = 0
        
        return spec_aug
    
    def process_circor_audio(self, audio: np.ndarray, orig_sr: int, augment: bool = False) -> np.ndarray:
        """
        Complete CirCor preprocessing pipeline following blueprint
        """
        # Step 1: Resample to target SR (usually keep native ~4kHz)
        audio = self.resample_audio(audio, orig_sr, self.circor_sr)
        
        # Step 2: Apply waveform augmentation (train-only)
        audio = self.apply_waveform_augmentation(audio, self.circor_sr, augment)
        
        # Step 3: Window to 12-12.5s
        audio = self.window_audio(audio, self.circor_target_length, "circor")
        
        # Step 4: DWT denoising (optional)
        audio = self.dwt_denoise(audio)
        
        # Step 5: Re-pad after DWT if needed
        audio = self.window_audio(audio, self.circor_target_length, "circor")
        
        # Step 6: Compute mel spectrogram
        log_mel = self.compute_mel_spectrogram(audio, self.circor_sr, "circor")
        
        # Step 7: Per-sample z-score normalization
        log_mel_norm = self.normalize_spectrogram(log_mel)
        
        # Step 8: Resize to 224×224
        spectrogram_224 = self.resize_spectrogram(log_mel_norm, self.target_size)
        
        # Step 9: Apply SpecAugment (train-only)
        spectrogram_final = self.apply_spec_augment(spectrogram_224, augment)
        
        # Return as (1, 224, 224) - treat as 1-channel image
        return spectrogram_final[np.newaxis, :, :]
    
    def process_yaseen_audio(self, audio: np.ndarray, orig_sr: int, augment: bool = False) -> np.ndarray:
        """
        Complete Yaseen preprocessing pipeline following blueprint
        """
        # Step 1: Resample (keep 22.05 kHz or downsample to 4kHz for consistency)
        audio = self.resample_audio(audio, orig_sr, self.yaseen_sr)
        
        # Step 2: Apply waveform augmentation (train-only)
        audio = self.apply_waveform_augmentation(audio, self.yaseen_sr, augment)
        
        # Step 3: Window to 3s
        audio = self.window_audio(audio, self.yaseen_target_length, "yaseen")
        
        # Step 4: DWT denoising (optional)
        audio = self.dwt_denoise(audio)
        
        # Step 5: Re-pad after DWT if needed
        audio = self.window_audio(audio, self.yaseen_target_length, "yaseen")
        
        # Step 6: Compute mel spectrogram
        log_mel = self.compute_mel_spectrogram(audio, self.yaseen_sr, "yaseen")
        
        # Step 7: Per-sample z-score normalization
        log_mel_norm = self.normalize_spectrogram(log_mel)
        
        # Step 8: Resize to 224×224
        spectrogram_224 = self.resize_spectrogram(log_mel_norm, self.target_size)
        
        # Step 9: Apply SpecAugment (train-only)
        spectrogram_final = self.apply_spec_augment(spectrogram_224, augment)
        
        # Return as (1, 224, 224) - treat as 1-channel image
        return spectrogram_final[np.newaxis, :, :]
    
    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration"""
        return {
            'circor_sr': self.circor_sr,
            'yaseen_sr': self.yaseen_sr,
            'circor_clip_length': self.circor_clip_length,
            'yaseen_clip_length': self.yaseen_clip_length,
            'mel_bins': self.mel_bins,
            'target_size': self.target_size,
            'use_dwt': self.use_dwt,
            'dwt_wavelet': self.dwt_wavelet,
            'dwt_levels': self.dwt_levels,
            'circor_target_length': self.circor_target_length,
            'yaseen_target_length': self.yaseen_target_length
        }

def test_processor():
    """Test the hybrid audio processor"""
    processor = HybridAudioProcessor()
    
    # Create dummy audio
    circor_audio = np.random.randn(48000)  # 12s at 4kHz
    yaseen_audio = np.random.randn(66150)  # 3s at 22.05kHz
    
    # Process
    circor_spec = processor.process_circor_audio(circor_audio, 4000, augment=False)
    yaseen_spec = processor.process_yaseen_audio(yaseen_audio, 22050, augment=False)
    
    print(f"CirCor output shape: {circor_spec.shape}")  # Should be (1, 224, 224)
    print(f"Yaseen output shape: {yaseen_spec.shape}")  # Should be (1, 224, 224)
    
    print("Processor test completed successfully!")

if __name__ == "__main__":
    test_processor()
