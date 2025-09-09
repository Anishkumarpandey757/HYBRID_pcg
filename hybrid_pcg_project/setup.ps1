# Hybrid PCG Project Setup Script for PowerShell

# Navigate to project directory
cd 'd:\WORKSPACE\PYTHON\the-circor-digiscope-phonocardiogram-dataset-1.0.3\hybrid_pcg_project'

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, librosa, torch; print('All packages installed successfully!')"

# Run preprocessing
python src/preprocessing/preprocess_data.py

# Create visualizations
python -c "from src.visualization import create_visualization_report; create_visualization_report('../data')"
