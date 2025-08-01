#!/usr/bin/env python3
"""
Generate audio comparison visualization with waveforms and spectrograms
Creates a single image showing clean, noisy, and enhanced audio comparisons
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.audio_utils import load_audio

def create_audio_comparison(sample_id="p232_010", output_file="audio_comparison.png"):
    """
    Create a comprehensive audio comparison visualization
    
    Args:
        sample_id: Audio sample identifier (e.g., "p232_010")
        output_file: Output filename for the comparison image
    """
    
    # Define file paths
    base_dir = Path(__file__).parent.parent
    clean_path = base_dir / "dataset" / "clean_testset_wav" / f"{sample_id}.wav"
    noisy_path = base_dir / "dataset" / "noisy_testset_wav" / f"{sample_id}.wav"
    enhanced_path = base_dir / "results" / "comparison" / f"{sample_id}_enhanced.wav"
    
    # Fallback paths if primary paths don't exist
    if not enhanced_path.exists():
        enhanced_path = base_dir / "results" / "comparison" / f"test_enhanced_{sample_id}.wav"
    
    print(f"Loading audio files for {sample_id}...")
    print(f"Clean: {clean_path}")
    print(f"Noisy: {noisy_path}")
    print(f"Enhanced: {enhanced_path}")
    
    # Load audio files
    try:
        clean_audio, sr = load_audio(str(clean_path))
        noisy_audio, _ = load_audio(str(noisy_path))
        enhanced_audio, _ = load_audio(str(enhanced_path))
        
        print(f"Sample rate: {sr} Hz")
        print(f"Audio lengths - Clean: {len(clean_audio)}, Noisy: {len(noisy_audio)}, Enhanced: {len(enhanced_audio)}")
        
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return
    
    # Ensure all audio has the same length (trim to shortest)
    min_length = min(len(clean_audio), len(noisy_audio), len(enhanced_audio))
    clean_audio = clean_audio[:min_length]
    noisy_audio = noisy_audio[:min_length]
    enhanced_audio = enhanced_audio[:min_length]
    
    # Create time axis
    time = np.linspace(0, len(clean_audio) / sr, len(clean_audio))
    
    # Compute spectrograms
    n_fft = 1024
    hop_length = 256
    
    clean_stft = librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length)
    noisy_stft = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)
    enhanced_stft = librosa.stft(enhanced_audio, n_fft=n_fft, hop_length=hop_length)
    
    clean_db = librosa.amplitude_to_db(np.abs(clean_stft), ref=np.max)
    noisy_db = librosa.amplitude_to_db(np.abs(noisy_stft), ref=np.max)
    enhanced_db = librosa.amplitude_to_db(np.abs(enhanced_stft), ref=np.max)
    
    # Create the plot
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = ['#2E8B57', '#DC143C', '#4169E1']  # Sea Green, Crimson, Royal Blue
    titles = ['Clean Audio', 'Noisy Audio', 'Enhanced Audio']
    audio_data = [clean_audio, noisy_audio, enhanced_audio]
    spectrogram_data = [clean_db, noisy_db, enhanced_db]
    
    # Plot waveforms (top row)
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(time, audio_data[i], color=colors[i], linewidth=0.8, alpha=0.8)
        ax.set_title(f'{titles[i]} - Waveform', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(time))
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
    
    # Plot spectrograms (bottom row)
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        img = librosa.display.specshow(
            spectrogram_data[i], 
            sr=sr, 
            hop_length=hop_length,
            x_axis='time', 
            y_axis='hz',
            ax=ax,
            cmap='viridis'
        )
        ax.set_title(f'{titles[i]} - Spectrogram', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.tick_params(labelsize=10)
        
        # Styling
        ax.tick_params(labelsize=10)
    
    # Add main title
    fig.suptitle(
        f'Audio Enhancement Comparison - Sample {sample_id}', 
        fontsize=18, 
        fontweight='bold', 
        y=0.95
    )
    
    # Add subtitle with technical info
    duration = len(clean_audio) / sr
    fig.text(
        0.5, 0.91, 
        f'Duration: {duration:.2f}s | Sample Rate: {sr} Hz | FFT Size: {n_fft} | Hop Length: {hop_length}',
        ha='center', 
        fontsize=12, 
        style='italic',
        alpha=0.7
    )
    
    # Save the figure
    output_path = Path(__file__).parent / output_file
    plt.savefig(
        output_path, 
        dpi=300, 
        bbox_inches='tight', 
        facecolor='white',
        edgecolor='none'
    )
    
    print(f"✅ Comparison visualization saved to: {output_path}")
    plt.close()

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate audio comparison visualization')
    parser.add_argument(
        '--sample', 
        default='p232_010', 
        help='Audio sample ID (default: p232_010)'
    )
    parser.add_argument(
        '--output', 
        default='audio_comparison.png', 
        help='Output filename (default: audio_comparison.png)'
    )
    
    args = parser.parse_args()
    
    print("🎵 Audio Enhancement Comparison Generator")
    print("=" * 50)
    
    create_audio_comparison(args.sample, args.output)

if __name__ == "__main__":
    main()
