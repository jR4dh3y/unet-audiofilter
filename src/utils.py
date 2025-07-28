"""
Utility functions for speech enhancement project
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import librosa
import librosa.display
import os
from typing import List, Dict, Tuple, Optional
import pandas as pd

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_waveform(waveform: np.ndarray, sr: int = 16000, title: str = "Waveform", 
                 figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot audio waveform
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    time = np.arange(len(waveform)) / sr
    plt.plot(time, waveform)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(spectrogram: np.ndarray, sr: int = 16000, hop_length: int = 256,
                    title: str = "Spectrogram", figsize: Tuple[int, int] = (12, 6),
                    y_axis: str = 'hz', x_axis: str = 'time') -> None:
    """
    Plot magnitude spectrogram
    
    Args:
        spectrogram: Magnitude spectrogram
        sr: Sample rate
        hop_length: Hop length used in STFT
        title: Plot title
        figsize: Figure size
        y_axis: Y-axis scale ('hz', 'linear', 'log', 'mel')
        x_axis: X-axis scale ('time', 'frames')
    """
    plt.figure(figsize=figsize)
    
    # Convert to dB scale
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    # Plot spectrogram
    img = librosa.display.specshow(
        spectrogram_db,
        sr=sr,
        hop_length=hop_length,
        y_axis=y_axis,
        x_axis=x_axis,
        cmap='viridis'
    )
    
    plt.title(title)
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def plot_comparison_spectrograms(noisy_spec: np.ndarray, clean_spec: np.ndarray, 
                                enhanced_spec: np.ndarray, sr: int = 16000,
                                hop_length: int = 256, figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot comparison of noisy, clean, and enhanced spectrograms
    
    Args:
        noisy_spec: Noisy spectrogram
        clean_spec: Clean spectrogram
        enhanced_spec: Enhanced spectrogram
        sr: Sample rate
        hop_length: Hop length
        figsize: Figure size
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Convert to dB scale
    noisy_db = librosa.amplitude_to_db(noisy_spec, ref=np.max)
    clean_db = librosa.amplitude_to_db(clean_spec, ref=np.max)
    enhanced_db = librosa.amplitude_to_db(enhanced_spec, ref=np.max)
    
    # Plot noisy
    img1 = librosa.display.specshow(
        noisy_db, sr=sr, hop_length=hop_length, y_axis='hz', x_axis='time',
        ax=axes[0], cmap='viridis'
    )
    axes[0].set_title('Noisy Speech')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # Plot clean
    img2 = librosa.display.specshow(
        clean_db, sr=sr, hop_length=hop_length, y_axis='hz', x_axis='time',
        ax=axes[1], cmap='viridis'
    )
    axes[1].set_title('Clean Speech')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    # Plot enhanced
    img3 = librosa.display.specshow(
        enhanced_db, sr=sr, hop_length=hop_length, y_axis='hz', x_axis='time',
        ax=axes[2], cmap='viridis'
    )
    axes[2].set_title('Enhanced Speech')
    fig.colorbar(img3, ax=axes[2], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history: Dict[str, List[float]], 
                         figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot training history (loss curves)
    
    Args:
        history: Dictionary containing training history
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics if available
    metrics_to_plot = ['pesq', 'stoi', 'sdr']
    metric_found = False
    
    for metric in metrics_to_plot:
        if f'val_{metric}' in history:
            axes[1].plot(history[f'val_{metric}'], label=f'Val {metric.upper()}', linewidth=2)
            metric_found = True
    
    if metric_found:
        axes[1].set_title('Validation Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No metrics available', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Metrics (Not Available)')
    
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                           figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot comparison of metrics across different models/conditions
    
    Args:
        metrics_dict: Dictionary of metrics for different models
        figsize: Figure size
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics_dict).T
    
    # Select main metrics
    main_metrics = ['pesq_wb', 'stoi', 'sdr', 'si_sdr']
    available_metrics = [m for m in main_metrics if m in df.columns]
    
    if not available_metrics:
        print("No standard metrics found for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics[:4]):
        if i < len(axes):
            df[metric].plot(kind='bar', ax=axes[i], color='skyblue', alpha=0.7)
            axes[i].set_title(f'{metric.upper().replace("_", "-")} Comparison')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def save_audio(audio: np.ndarray, filepath: str, sr: int = 16000) -> None:
    """
    Save audio to file
    
    Args:
        audio: Audio array
        filepath: Output file path
        sr: Sample rate
    """
    import soundfile as sf
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Normalize audio to prevent clipping
    audio = np.clip(audio, -1.0, 1.0)
    
    # Save audio
    sf.write(filepath, audio, sr)
    print(f"Audio saved to: {filepath}")

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def create_output_dirs(base_dir: str, subdirs: List[str]) -> Dict[str, str]:
    """
    Create output directories
    
    Args:
        base_dir: Base directory path
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary mapping subdirectory names to full paths
    """
    paths = {}
    
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        os.makedirs(full_path, exist_ok=True)
        paths[subdir] = full_path
    
    return paths

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        NumPy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    return tensor.numpy()

def numpy_to_tensor(array: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor
    
    Args:
        array: NumPy array
        device: Target device
        
    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(array).float()
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor

def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU)
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def print_model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> None:
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
    """
    print("=" * 70)
    print(f"Model Summary")
    print("=" * 70)
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Try to get model size
    try:
        dummy_input = torch.randn(1, *input_shape)
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"Model size: {model_size / 1024 / 1024:.2f} MB")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Could not compute model statistics: {e}")
    
    print("=" * 70)

class ProgressTracker:
    """Simple progress tracker for training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking variables"""
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def update(self, train_loss: float, val_loss: float = None, 
               lr: float = None, epoch: int = 0):
        """Update tracking history"""
        self.history['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
            
            # Check for best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                return True  # New best model
            else:
                self.patience_counter += 1
        
        if lr is not None:
            self.history['learning_rate'].append(lr)
        
        return False  # Not a new best model
    
    def should_stop(self, patience: int) -> bool:
        """Check if training should stop based on patience"""
        return self.patience_counter >= patience
    
    def get_summary(self) -> str:
        """Get training summary"""
        summary = f"Training Summary:\n"
        summary += f"Best validation loss: {self.best_loss:.6f} (epoch {self.best_epoch})\n"
        summary += f"Final training loss: {self.history['train_loss'][-1]:.6f}\n"
        if self.history['val_loss']:
            summary += f"Final validation loss: {self.history['val_loss'][-1]:.6f}\n"
        
        return summary

if __name__ == "__main__":
    print("Utilities module loaded successfully!")
    
    # Test device detection
    device = get_device()
    print(f"Available device: {device}")
    
    # Test progress tracker
    tracker = ProgressTracker()
    print("Progress tracker initialized")
