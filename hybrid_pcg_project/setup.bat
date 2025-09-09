@echo off
REM Hybrid PCG Project Setup Script for Windows CMD

REM Navigate to project directory
cd /d "d:\WORKSPACE\PYTHON\the-circor-digiscope-phonocardiogram-dataset-1.0.3\hybrid_pcg_project"

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

REM Verify installation
python -c "import numpy, pandas, librosa, torch; print('All packages installed successfully!')"

REM Run preprocessing
python src\preprocessing\preprocess_data.py

REM Create visualizations
python -c "from src.visualization import create_visualization_report; create_visualization_report('../data')"

pause
