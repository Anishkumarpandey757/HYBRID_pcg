# Hybrid PCG Classification Project

This project implements a hybrid deep learning architecture for phonocardiogram (PCG) classification using the CirCor DigiScope dataset. The approach combines CNN feature extraction with transformer attention mechanisms for robust heart sound analysis.

## Project Structure

```
hybrid_pcg_project/
├── data/
│   ├── preprocessed/     # Processed audio files
│   └── features/         # Extracted features
├── src/
│   ├── preprocessing/    # Data preprocessing modules
│   ├── models/          # Model architectures
│   ├── training/        # Training scripts
│   └── evaluation/      # Evaluation utilities
├── models/              # Saved model checkpoints
├── configs/             # Configuration files
├── utils/              # Utility functions
├── notebooks/          # Jupyter notebooks for analysis
└── requirements.txt    # Python dependencies
```

## Features

- **Multi-modal preprocessing**: Handles audio files with segmentation annotations
- **Hybrid architecture**: CNN + Transformer for optimal feature learning
- **Robust preprocessing**: Signal filtering, normalization, and augmentation
- **Flexible configuration**: YAML-based configuration system
- **Comprehensive evaluation**: Multiple metrics for model assessment

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**:
```bash
python src/preprocessing/preprocess_data.py --config configs/preprocessing.yaml
```

2. **Model Training**:
```bash
python src/training/train_model.py --config configs/training.yaml
```

3. **Evaluation**:
```bash
python src/evaluation/evaluate_model.py --model_path models/best_model.pth
```

## Dataset

The project uses the CirCor DigiScope Phonocardiogram Dataset v1.0.3, which contains:
- Heart sound recordings from multiple auscultation locations (AV, PV, TV, MV)
- Segmentation annotations for S1, S2, and murmurs
- Patient metadata including demographics and clinical information

## Model Architecture

The hybrid model combines:
1. **CNN Feature Extractor**: Captures local temporal and spectral patterns
2. **Transformer Encoder**: Models long-range dependencies and global context
3. **Classification Head**: Multi-class output for heart sound classification

## License

This project is for research purposes. Please refer to the original dataset license for data usage terms.
