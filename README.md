# Hybrid CNN14+PaSST PCG Preprocessing Pipeline

A complete preprocessing pipeline for phonocardiogram (PCG) analysis using hybrid CNN14+PaSST architecture, implementing exact blueprint specifications for CirCor DigiScope and Yaseen datasets.

## 🎯 Overview

This project implements a state-of-the-art hybrid preprocessing pipeline that combines:
- **CNN14**: Convolutional neural network for local feature extraction
- **PaSST**: Patchout faSt Spectrogram Transformer for global pattern recognition
- **DWT Denoising**: Daubechies db4 level-5 wavelet denoising
- **Mel-Spectrograms**: 128-mel filter banks with 25ms/10ms frame parameters
- **Multi-Dataset Support**: Unified processing for CirCor and Yaseen datasets

## 📊 Dataset Support

### CirCor DigiScope Dataset
- **942 patients** with multi-location recordings
- **Locations**: AV, MV, PV, TV, Phc (5 cardiac locations)
- **Multi-task learning**: Murmur detection + Outcome prediction
- **Sample rate**: 4000 Hz
- **Clip length**: 12-12.5 seconds

### Yaseen Dataset  
- **990 recordings** (5-class disease classification)
- **Classes**: Normal, AS, MS, MR, MVP
- **Sample rate**: 22050 Hz → downsampled to 4000 Hz
- **Clip length**: 3 seconds

## 🚀 Key Features

### ✅ Blueprint-Compliant Preprocessing
- **DWT Denoising**: db4 wavelet, level-5 decomposition
- **Mel-Spectrograms**: 128 mel-filter banks, 25ms frames, 10ms hop
- **Log Transform**: Natural logarithm with epsilon stabilization
- **Z-Score Normalization**: Per-spectrogram standardization
- **Resizing**: 224×224 for CNN14+PaSST compatibility

### ✅ Advanced Data Management
- **HDF5 Storage**: Efficient binary format for large datasets
- **Stratified 5-Fold CV**: Balanced cross-validation splits
- **Memory Optimization**: Batch processing with configurable limits
- **Error Handling**: Robust processing with detailed logging

### ✅ Multi-Task Architecture Support
- **CirCor**: Simultaneous murmur + outcome prediction
- **Yaseen**: Single-task disease classification
- **Unified Pipeline**: Consistent preprocessing across datasets

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
Git
```

### Setup
```bash
# Clone repository
git clone https://github.com/Anishkumarpandey757/HYBRID_pcg.git
cd HYBRID_pcg

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r hybrid_pcg_project/requirements.txt
```

## 📁 Project Structure

```
HYBRID_pcg/
├── hybrid_pcg_project/
│   ├── src/
│   │   ├── preprocessing/
│   │   │   ├── hybrid_data_loader.py      # Unified dataset loading
│   │   │   ├── hybrid_audio_processor.py  # DWT + mel-spectrogram
│   │   │   └── hybrid_preprocess_main.py  # Main pipeline
│   │   └── visualization/
│   ├── configs/
│   │   └── preprocessing.yaml             # Configuration settings
│   ├── data/
│   │   ├── preprocessed/                  # Output HDF5 files
│   │   └── splits/                        # Cross-validation splits
│   └── requirements.txt
├── training_data/                         # CirCor dataset (excluded)
├── yaseen/                               # Yaseen dataset (excluded)
└── .gitignore                            # Excludes large files
```

## 🔧 Usage

### Quick Start
```bash
cd hybrid_pcg_project

# Run preprocessing with default settings
python -m src.preprocessing.hybrid_preprocess_main \
  --circor_path "/path/to/circor/dataset" \
  --yaseen_path "/path/to/yaseen/dataset" \
  --output_dir "data"
```

### Advanced Options
```bash
# Custom configuration
python -m src.preprocessing.hybrid_preprocess_main \
  --circor_path "/path/to/circor" \
  --yaseen_path "/path/to/yaseen" \
  --output_dir "data" \
  --yaseen_sr 4000 \
  --use_dwt \
  --max_circor_patients 100 \
  --max_yaseen_recordings 500
```

### Parameters
- `--circor_path`: Path to CirCor dataset directory
- `--yaseen_path`: Path to Yaseen dataset directory  
- `--output_dir`: Output directory for processed data
- `--yaseen_sr`: Target sampling rate for Yaseen (4000 or 22050)
- `--use_dwt`: Enable DWT denoising (default: True)
- `--max_circor_patients`: Limit CirCor processing (for testing)
- `--max_yaseen_recordings`: Limit Yaseen processing (for testing)

## 📈 Results

### Processing Statistics
- **CirCor**: 942 patients → ~3,500+ spectrograms
- **Yaseen**: 990 recordings → 990 spectrograms
- **Output Format**: HDF5 with (1, 224, 224) spectrograms
- **Storage**: ~200-400MB total (efficient compression)
- **Processing Time**: ~4-5 minutes for full datasets

### Quality Metrics
- **DWT Denoising**: Removes artifacts while preserving cardiac features
- **Mel-Spectrograms**: Optimized for heart sound frequency characteristics
- **Class Balance**: Stratified splits maintain original distributions
- **Validation**: 5-fold cross-validation ready

## 🔬 Technical Details

### Signal Processing Pipeline
1. **Audio Loading**: 16-bit PCM, dataset-specific sample rates
2. **DWT Denoising**: Daubechies db4, 5-level decomposition
3. **Mel-Spectrogram**: 128 mel-bins, 25ms frames, 10ms hop
4. **Log Transform**: `log(mel + 1e-8)` for stability
5. **Normalization**: Z-score per spectrogram
6. **Resizing**: Bilinear interpolation to 224×224

### Data Augmentation (Optional)
- **Time Shifting**: Random temporal shifts
- **SpecAugment**: Frequency/time masking
- **Mixup**: Sample interpolation
- **Volume Scaling**: Amplitude variation

## 🧪 Testing

```bash
# Test data loading
python debug_loader.py

# Test preprocessing pipeline
python test_preprocessing.py

# Verify outputs
python -c "
import h5py
with h5py.File('data/preprocessed/circor_spectrograms.h5', 'r') as f:
    print('CirCor shape:', f['spectrograms'].shape)
with h5py.File('data/preprocessed/yaseen_spectrograms.h5', 'r') as f:
    print('Yaseen shape:', f['spectrograms'].shape)
"
```

## 📋 Requirements

### Core Dependencies
```
numpy>=1.20.0
scipy>=1.7.0
librosa>=0.9.0
h5py>=3.6.0
pandas>=1.3.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
PyWavelets>=1.2.0
tqdm>=4.62.0
```

### Optional Dependencies
```
matplotlib>=3.5.0  # Visualization
jupyter>=1.0.0     # Notebooks
tensorboard>=2.8.0 # Logging
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🔗 References

- **CirCor DigiScope Dataset**: [PhysioNet](https://physionet.org/content/circor-heart-sound/1.0.3/)
- **Yaseen Dataset**: Heart sound classification dataset
- **CNN14**: Deep Residual Learning for Heart Sound Analysis
- **PaSST**: Patchout faSt Spectrogram Transformer

## 📧 Contact

- **Author**: Anish Kumar Pandey
- **GitHub**: [@Anishkumarpandey757](https://github.com/Anishkumarpandey757)
- **Repository**: [HYBRID_pcg](https://github.com/Anishkumarpandey757/HYBRID_pcg)

---

*Successfully preprocessing 942 CirCor patients + 990 Yaseen recordings for hybrid CNN14+PaSST training* 🎵💓
