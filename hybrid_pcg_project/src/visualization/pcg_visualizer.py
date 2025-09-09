"""
Visualization tools for PCG data analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa
import librosa.display
from scipy import signal
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PCGVisualizer:
    """
    Visualization tools for phonocardiogram analysis
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_waveform(self, audio: np.ndarray, sr: int, title: str = "PCG Waveform",
                     annotations: List[Dict] = None, save_path: str = None):
        """
        Plot audio waveform with optional annotations
        """
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Time axis
        time = np.linspace(0, len(audio) / sr, len(audio))
        
        # Plot waveform
        ax.plot(time, audio, 'b-', linewidth=0.8, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add annotations if provided
        if annotations:
            colors = {'S1': 'red', 'S2': 'green', 'systolic': 'orange', 
                     'diastolic': 'purple', 'murmur': 'brown', 'unlabeled': 'gray'}
            
            for ann in annotations:
                start_time = ann['start_time']
                end_time = ann['end_time']
                label = ann['label_name']
                color = colors.get(label, 'black')
                
                ax.axvspan(start_time, end_time, alpha=0.3, color=color, label=label)
            
            # Remove duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spectrogram(self, audio: np.ndarray, sr: int, title: str = "PCG Spectrogram",
                        annotations: List[Dict] = None, save_path: str = None):
        """
        Plot spectrogram of audio signal
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Waveform
        time = np.linspace(0, len(audio) / sr, len(audio))
        ax1.plot(time, audio, 'b-', linewidth=0.8)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{title} - Waveform')
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax2)
        ax2.set_title(f'{title} - Spectrogram')
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        # Add annotations if provided
        if annotations:
            colors = {'S1': 'red', 'S2': 'green', 'systolic': 'orange', 
                     'diastolic': 'purple', 'murmur': 'brown', 'unlabeled': 'gray'}
            
            for ann in annotations:
                start_time = ann['start_time']
                end_time = ann['end_time']
                label = ann['label_name']
                color = colors.get(label, 'black')
                
                for ax in [ax1, ax2]:
                    ax.axvspan(start_time, end_time, alpha=0.3, color=color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_mel_spectrogram(self, audio: np.ndarray, sr: int, title: str = "Mel Spectrogram",
                           save_path: str = None):
        """
        Plot mel spectrogram
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot
        img = librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', 
                                     sr=sr, ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_mfcc(self, audio: np.ndarray, sr: int, title: str = "MFCC Features",
                 save_path: str = None):
        """
        Plot MFCC features
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Plot
        img = librosa.display.specshow(mfcc, y_axis='linear', x_axis='time', 
                                     sr=sr, ax=ax)
        ax.set_title(title)
        ax.set_ylabel('MFCC Coefficients')
        fig.colorbar(img, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_distribution(self, feature_df: pd.DataFrame, feature_cols: List[str],
                                 save_path: str = None):
        """
        Plot distribution of extracted features
        """
        n_features = len(feature_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(feature_cols):
            if i < len(axes):
                ax = axes[i]
                
                # Plot histogram
                data = feature_df[feature].dropna()
                ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'{feature}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(feature_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_outcome_distribution(self, metadata_df: pd.DataFrame, save_path: str = None):
        """
        Plot distribution of outcomes and patient characteristics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Outcome distribution
        if 'outcome' in metadata_df.columns:
            outcome_counts = metadata_df['outcome'].value_counts()
            axes[0, 0].pie(outcome_counts.values, labels=outcome_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Outcome Distribution')
        
        # Age distribution by outcome
        if 'age' in metadata_df.columns and 'outcome' in metadata_df.columns:
            sns.boxplot(data=metadata_df, x='outcome', y='age', ax=axes[0, 1])
            axes[0, 1].set_title('Age Distribution by Outcome')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sex distribution
        if 'sex' in metadata_df.columns:
            sex_counts = metadata_df['sex'].value_counts()
            axes[1, 0].bar(sex_counts.index, sex_counts.values)
            axes[1, 0].set_title('Sex Distribution')
            axes[1, 0].set_ylabel('Count')
        
        # Murmur presence
        if 'murmur' in metadata_df.columns:
            murmur_counts = metadata_df['murmur'].value_counts()
            axes[1, 1].pie(murmur_counts.values, labels=murmur_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Murmur Presence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_matrix(self, feature_df: pd.DataFrame, feature_cols: List[str],
                              save_path: str = None):
        """
        Plot correlation matrix of features
        """
        # Select numeric features
        numeric_df = feature_df[feature_cols].select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_segment_analysis(self, segment_df: pd.DataFrame, save_path: str = None):
        """
        Plot analysis of segments
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Segment type distribution
        if 'segment_label' in segment_df.columns:
            segment_counts = segment_df['segment_label'].value_counts()
            axes[0, 0].bar(segment_counts.index, segment_counts.values)
            axes[0, 0].set_title('Segment Type Distribution')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_ylabel('Count')
        
        # Segment duration distribution
        if 'segment_duration' in segment_df.columns:
            axes[0, 1].hist(segment_df['segment_duration'].dropna(), bins=50, alpha=0.7)
            axes[0, 1].set_title('Segment Duration Distribution')
            axes[0, 1].set_xlabel('Duration (s)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Duration by segment type
        if 'segment_duration' in segment_df.columns and 'segment_label' in segment_df.columns:
            sns.boxplot(data=segment_df, x='segment_label', y='segment_duration', ax=axes[1, 0])
            axes[1, 0].set_title('Duration by Segment Type')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Segments per recording location
        if 'location' in segment_df.columns:
            location_counts = segment_df['location'].value_counts()
            axes[1, 1].pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Segments by Recording Location')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_interactive_spectrogram(self, audio: np.ndarray, sr: int, 
                                     title: str = "Interactive PCG Spectrogram",
                                     annotations: List[Dict] = None):
        """
        Create interactive spectrogram using Plotly
        """
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Create figure
        fig = go.Figure()
        
        # Add spectrogram
        fig.add_trace(go.Heatmap(
            z=Sxx_db,
            x=t,
            y=f,
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)"),
            hovertemplate="Time: %{x:.3f}s<br>Frequency: %{y:.1f}Hz<br>Power: %{z:.1f}dB<extra></extra>"
        ))
        
        # Add annotations if provided
        if annotations:
            colors = {'S1': 'red', 'S2': 'green', 'systolic': 'orange', 
                     'diastolic': 'purple', 'murmur': 'brown', 'unlabeled': 'gray'}
            
            for ann in annotations:
                fig.add_vline(
                    x=ann['start_time'],
                    line=dict(color=colors.get(ann['label_name'], 'white'), width=2),
                    annotation_text=ann['label_name']
                )
                fig.add_vline(
                    x=ann['end_time'],
                    line=dict(color=colors.get(ann['label_name'], 'white'), width=2, dash='dash')
                )
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            width=1000,
            height=600
        )
        
        return fig
    
    def plot_dataset_overview(self, stats_file: str, save_path: str = None):
        """
        Plot dataset overview from statistics file
        """
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Recording count by location
        if 'recording_stats' in stats and 'location_distribution' in stats['recording_stats']:
            loc_dist = stats['recording_stats']['location_distribution']
            axes[0, 0].bar(loc_dist.keys(), loc_dist.values())
            axes[0, 0].set_title('Recordings by Location')
            axes[0, 0].set_ylabel('Count')
        
        # Quality scores
        if 'recording_stats' in stats and 'quality_scores' in stats['recording_stats']:
            quality_scores = stats['recording_stats']['quality_scores']
            axes[0, 1].hist(quality_scores, bins=20, alpha=0.7)
            axes[0, 1].set_title('Quality Score Distribution')
            axes[0, 1].set_xlabel('Quality Score')
            axes[0, 1].set_ylabel('Frequency')
        
        # Segment type distribution
        if 'segment_stats' in stats and 'segment_types' in stats['segment_stats']:
            seg_types = stats['segment_stats']['segment_types']
            axes[1, 0].pie(seg_types.values(), labels=seg_types.keys(), autopct='%1.1f%%')
            axes[1, 0].set_title('Segment Type Distribution')
        
        # Duration statistics
        if 'recording_stats' in stats and 'average_duration' in stats['recording_stats']:
            avg_duration = stats['recording_stats']['average_duration']
            axes[1, 1].bar(['Average Duration'], [avg_duration])
            axes[1, 1].set_title('Average Recording Duration')
            axes[1, 1].set_ylabel('Duration (s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def create_visualization_report(data_dir: str, output_dir: str = "visualization_report"):
    """
    Create a comprehensive visualization report of the dataset
    """
    visualizer = PCGVisualizer(output_dir)
    
    data_path = Path(data_dir)
    
    # Load data if available
    feature_file = data_path / 'features' / 'recording_features.csv'
    segment_file = data_path / 'features' / 'segment_features.csv'
    stats_file = data_path / 'preprocessed' / 'dataset_statistics.json'
    
    print("Creating visualization report...")
    
    # Dataset overview
    if stats_file.exists():
        print("- Dataset overview")
        visualizer.plot_dataset_overview(str(stats_file), 'dataset_overview.png')
    
    # Feature analysis
    if feature_file.exists():
        print("- Feature analysis")
        feature_df = pd.read_csv(feature_file)
        
        # Select some key features for visualization
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        key_features = [col for col in numeric_cols if any(keyword in col.lower() 
                       for keyword in ['mfcc', 'spectral', 'rms', 'energy'])][:16]
        
        if key_features:
            visualizer.plot_feature_distribution(feature_df, key_features, 
                                               'feature_distributions.png')
            visualizer.plot_correlation_matrix(feature_df, key_features, 
                                             'feature_correlations.png')
        
        # Patient characteristics
        if any(col in feature_df.columns for col in ['outcome', 'age', 'sex', 'murmur']):
            visualizer.plot_outcome_distribution(feature_df, 'patient_characteristics.png')
    
    # Segment analysis
    if segment_file.exists():
        print("- Segment analysis")
        segment_df = pd.read_csv(segment_file)
        visualizer.plot_segment_analysis(segment_df, 'segment_analysis.png')
    
    print(f"Visualization report saved to: {output_dir}")

if __name__ == "__main__":
    # Example usage
    data_dir = "../data"
    create_visualization_report(data_dir)
