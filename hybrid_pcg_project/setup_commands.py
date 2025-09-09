"""
Terminal commands and setup instructions for Hybrid PCG Project
"""

SETUP_COMMANDS = {
    "windows_powershell": [
        "# Navigate to project directory",
        "cd 'd:\\WORKSPACE\\PYTHON\\the-circor-digiscope-phonocardiogram-dataset-1.0.3\\hybrid_pcg_project'",
        "",
        "# Create virtual environment",
        "python -m venv venv",
        "",
        "# Activate virtual environment",
        ".\\venv\\Scripts\\Activate.ps1",
        "",
        "# Upgrade pip",
        "python -m pip install --upgrade pip",
        "",
        "# Install requirements",
        "pip install -r requirements.txt",
        "",
        "# Verify installation",
        "python -c \"import numpy, pandas, librosa, torch; print('All packages installed successfully!')\"",
        "",
        "# Run preprocessing",
        "python src/preprocessing/preprocess_data.py",
        "",
        "# Create visualizations",
        "python -c \"from src.visualization import create_visualization_report; create_visualization_report('../data')\"",
    ],
    
    "windows_cmd": [
        "REM Navigate to project directory",
        "cd /d \"d:\\WORKSPACE\\PYTHON\\the-circor-digiscope-phonocardiogram-dataset-1.0.3\\hybrid_pcg_project\"",
        "",
        "REM Create virtual environment",
        "python -m venv venv",
        "",
        "REM Activate virtual environment",
        "venv\\Scripts\\activate.bat",
        "",
        "REM Upgrade pip",
        "python -m pip install --upgrade pip",
        "",
        "REM Install requirements",
        "pip install -r requirements.txt",
        "",
        "REM Verify installation",
        "python -c \"import numpy, pandas, librosa, torch; print('All packages installed successfully!')\"",
        "",
        "REM Run preprocessing",
        "python src\\preprocessing\\preprocess_data.py",
        "",
        "REM Create visualizations",
        "python -c \"from src.visualization import create_visualization_report; create_visualization_report('../data')\"",
    ],
    
    "linux_mac": [
        "# Navigate to project directory",
        "cd '/path/to/the-circor-digiscope-phonocardiogram-dataset-1.0.3/hybrid_pcg_project'",
        "",
        "# Create virtual environment",
        "python3 -m venv venv",
        "",
        "# Activate virtual environment",
        "source venv/bin/activate",
        "",
        "# Upgrade pip",
        "python -m pip install --upgrade pip",
        "",
        "# Install requirements",
        "pip install -r requirements.txt",
        "",
        "# Verify installation",
        "python -c \"import numpy, pandas, librosa, torch; print('All packages installed successfully!')\"",
        "",
        "# Run preprocessing",
        "python src/preprocessing/preprocess_data.py",
        "",
        "# Create visualizations",
        "python -c \"from src.visualization import create_visualization_report; create_visualization_report('../data')\"",
    ]
}

EXECUTION_COMMANDS = {
    "full_preprocessing": [
        "# Run complete preprocessing pipeline",
        "python src/preprocessing/preprocess_data.py --config configs/preprocessing.yaml --input_dir ../training_data --output_dir data",
    ],
    
    "quick_test": [
        "# Test preprocessing on subset",
        "python src/preprocessing/preprocess_data.py --config configs/preprocessing.yaml --input_dir ../training_data --output_dir data/test --max_patients 5",
    ],
    
    "feature_extraction_only": [
        "# Extract features only",
        "python -c \"from src.preprocessing import PCGFeatureExtractor; fe = PCGFeatureExtractor(); print('Feature extractor ready')\"",
    ],
    
    "create_visualizations": [
        "# Create comprehensive visualization report",
        "python -c \"from src.visualization import create_visualization_report; create_visualization_report('data', 'visualization_report')\"",
    ],
    
    "test_audio_loading": [
        "# Test audio loading functionality",
        "python -c \"from src.preprocessing import PCGDataLoader; dl = PCGDataLoader('../training_data'); print(f'Found {len(dl.get_patient_list())} patients')\"",
    ]
}

TROUBLESHOOTING = {
    "common_issues": [
        "# If torch installation fails, try:",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "",
        "# If librosa installation fails, install ffmpeg first:",
        "# On Windows: Download from https://ffmpeg.org/download.html",
        "# On Ubuntu: sudo apt install ffmpeg",
        "# On macOS: brew install ffmpeg",
        "",
        "# If memory issues during preprocessing:",
        "# Edit configs/preprocessing.yaml and reduce batch_size",
        "",
        "# If permission errors on Windows:",
        "# Run PowerShell as Administrator",
        "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser",
    ],
    
    "environment_issues": [
        "# Check Python version (requires 3.8+)",
        "python --version",
        "",
        "# Check virtual environment activation",
        "where python",  # Windows
        "which python",  # Linux/Mac
        "",
        "# Reinstall packages if needed",
        "pip uninstall -r requirements.txt -y",
        "pip install -r requirements.txt",
    ]
}

def print_setup_commands(platform="windows_powershell"):
    """Print setup commands for specified platform"""
    print(f"=== Setup Commands for {platform.upper()} ===\n")
    for cmd in SETUP_COMMANDS[platform]:
        print(cmd)
    
    print("\n=== Troubleshooting ===\n")
    for cmd in TROUBLESHOOTING["common_issues"]:
        print(cmd)

def create_setup_script(platform="windows_powershell", output_file=None):
    """Create a setup script file"""
    if output_file is None:
        ext = ".ps1" if "powershell" in platform else ".bat" if "cmd" in platform else ".sh"
        output_file = f"setup{ext}"
    
    with open(output_file, 'w') as f:
        for cmd in SETUP_COMMANDS[platform]:
            if not cmd.startswith("#") and not cmd.startswith("REM") and cmd.strip():
                f.write(cmd + "\n")
    
    print(f"Setup script created: {output_file}")

if __name__ == "__main__":
    print_setup_commands("windows_powershell")
