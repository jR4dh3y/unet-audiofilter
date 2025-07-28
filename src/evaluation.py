"""
Evaluation Metrics for Speech Enhancement
"""

import numpy as np
import librosa
import torch
from pesq import pesq
from pystoi import stoi
import mir_eval
from scipy import signal
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class SpeechEnhancementEvaluator:
    """Comprehensive evaluation for speech enhancement models"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def compute_pesq(self, reference: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Compute PESQ (Perceptual Evaluation of Speech Quality)
        Args:
            reference: Clean reference audio
            enhanced: Enhanced audio
        Returns:
            PESQ score (higher is better, range: -0.5 to 4.5)
        """
        try:
            # Ensure same length
            min_len = min(len(reference), len(enhanced))
            reference = reference[:min_len]
            enhanced = enhanced[:min_len]
            
            # PESQ expects 16kHz audio
            if self.sample_rate != 16000:
                reference = librosa.resample(reference, orig_sr=self.sample_rate, target_sr=16000)
                enhanced = librosa.resample(enhanced, orig_sr=self.sample_rate, target_sr=16000)
            
            # Normalize audio to prevent clipping
            reference = reference / (np.max(np.abs(reference)) + 1e-8)
            enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8)
            
            return pesq(16000, reference, enhanced, 'wb')  # Wide-band PESQ
        
        except Exception as e:
            print(f"PESQ computation failed: {e}")
            return -1.0
    
    def compute_stoi(self, reference: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Compute STOI (Short-Time Objective Intelligibility)
        Args:
            reference: Clean reference audio
            enhanced: Enhanced audio
        Returns:
            STOI score (higher is better, range: 0 to 1)
        """
        try:
            # Ensure same length
            min_len = min(len(reference), len(enhanced))
            reference = reference[:min_len]
            enhanced = enhanced[:min_len]
            
            # Normalize audio
            reference = reference / (np.max(np.abs(reference)) + 1e-8)
            enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8)
            
            return stoi(reference, enhanced, self.sample_rate, extended=False)
        
        except Exception as e:
            print(f"STOI computation failed: {e}")
            return 0.0
    
    def compute_sdr(self, reference: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """
        Compute SDR (Signal-to-Distortion Ratio) and related metrics
        Args:
            reference: Clean reference audio
            enhanced: Enhanced audio
        Returns:
            Dictionary with SDR, SIR, SAR values
        """
        try:
            # Ensure same length
            min_len = min(len(reference), len(enhanced))
            reference = reference[:min_len]
            enhanced = enhanced[:min_len]
            
            # Reshape for mir_eval (expects 2D arrays)
            reference = reference.reshape(1, -1)
            enhanced = enhanced.reshape(1, -1)
            
            sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
                reference, enhanced, compute_permutation=False
            )
            
            return {
                'sdr': float(sdr[0]),
                'sir': float(sir[0]),
                'sar': float(sar[0])
            }
        
        except Exception as e:
            print(f"SDR computation failed: {e}")
            return {'sdr': -np.inf, 'sir': -np.inf, 'sar': -np.inf}
    
    def compute_si_sdr(self, reference: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Compute Scale-Invariant SDR
        Args:
            reference: Clean reference audio
            enhanced: Enhanced audio
        Returns:
            SI-SDR score (higher is better)
        """
        try:
            # Ensure same length
            min_len = min(len(reference), len(enhanced))
            reference = reference[:min_len]
            enhanced = enhanced[:min_len]
            
            # Zero-mean signals
            reference = reference - np.mean(reference)
            enhanced = enhanced - np.mean(enhanced)
            
            # Compute scale factor
            alpha = np.dot(enhanced, reference) / (np.dot(reference, reference) + 1e-8)
            
            # Scale reference
            scaled_reference = alpha * reference
            
            # Compute noise
            noise = enhanced - scaled_reference
            
            # Compute SI-SDR
            si_sdr = 10 * np.log10(
                (np.dot(scaled_reference, scaled_reference) + 1e-8) / 
                (np.dot(noise, noise) + 1e-8)
            )
            
            return float(si_sdr)
        
        except Exception as e:
            print(f"SI-SDR computation failed: {e}")
            return -np.inf
    
    def compute_spectral_metrics(self, reference: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral domain metrics
        Args:
            reference: Clean reference audio
            enhanced: Enhanced audio
        Returns:
            Dictionary with spectral metrics
        """
        try:
            # Compute spectrograms
            ref_stft = librosa.stft(reference, n_fft=1024, hop_length=256)
            enh_stft = librosa.stft(enhanced, n_fft=1024, hop_length=256)
            
            ref_mag = np.abs(ref_stft)
            enh_mag = np.abs(enh_stft)
            
            # Ensure same shape
            min_shape = (min(ref_mag.shape[0], enh_mag.shape[0]), 
                        min(ref_mag.shape[1], enh_mag.shape[1]))
            ref_mag = ref_mag[:min_shape[0], :min_shape[1]]
            enh_mag = enh_mag[:min_shape[0], :min_shape[1]]
            
            # Log-magnitude spectral distance
            ref_log = np.log(ref_mag + 1e-8)
            enh_log = np.log(enh_mag + 1e-8)
            lsd = np.mean((ref_log - enh_log) ** 2)
            
            # Spectral convergence
            numerator = np.linalg.norm(ref_mag - enh_mag, 'fro')
            denominator = np.linalg.norm(ref_mag, 'fro')
            spectral_convergence = numerator / (denominator + 1e-8)
            
            # Magnitude error
            magnitude_error = np.mean(np.abs(ref_mag - enh_mag))
            
            return {
                'lsd': float(lsd),
                'spectral_convergence': float(spectral_convergence),
                'magnitude_error': float(magnitude_error)
            }
        
        except Exception as e:
            print(f"Spectral metrics computation failed: {e}")
            return {'lsd': np.inf, 'spectral_convergence': np.inf, 'magnitude_error': np.inf}
    
    def evaluate_pair(self, reference: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """
        Compute all evaluation metrics for a single audio pair
        Args:
            reference: Clean reference audio
            enhanced: Enhanced audio
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Objective metrics
        metrics['pesq'] = self.compute_pesq(reference, enhanced)
        metrics['stoi'] = self.compute_stoi(reference, enhanced)
        metrics['si_sdr'] = self.compute_si_sdr(reference, enhanced)
        
        # SDR metrics
        sdr_metrics = self.compute_sdr(reference, enhanced)
        metrics.update(sdr_metrics)
        
        # Spectral metrics
        spectral_metrics = self.compute_spectral_metrics(reference, enhanced)
        metrics.update(spectral_metrics)
        
        return metrics
    
    def evaluate_dataset(self, reference_files: List[str], enhanced_files: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate entire dataset
        Args:
            reference_files: List of clean audio file paths
            enhanced_files: List of enhanced audio file paths
        Returns:
            Dictionary with lists of metrics for each file
        """
        all_metrics = {
            'pesq': [], 'stoi': [], 'si_sdr': [], 'sdr': [], 'sir': [], 'sar': [],
            'lsd': [], 'spectral_convergence': [], 'magnitude_error': []
        }
        
        for ref_file, enh_file in zip(reference_files, enhanced_files):
            try:
                # Load audio files
                reference, _ = librosa.load(ref_file, sr=self.sample_rate)
                enhanced, _ = librosa.load(enh_file, sr=self.sample_rate)
                
                # Compute metrics
                metrics = self.evaluate_pair(reference, enhanced)
                
                # Store metrics
                for key, value in metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(value)
                
                print(f"Processed: {ref_file}")
                
            except Exception as e:
                print(f"Error processing {ref_file}: {e}")
                # Add NaN values for failed processing
                for key in all_metrics:
                    all_metrics[key].append(np.nan)
        
        return all_metrics
    
    def compute_summary_statistics(self, metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for metrics
        Args:
            metrics: Dictionary with lists of metrics
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        for metric_name, values in metrics.items():
            # Remove NaN values
            valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
            
            if valid_values:
                summary[metric_name] = {
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'median': float(np.median(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'count': len(valid_values)
                }
            else:
                summary[metric_name] = {
                    'mean': np.nan, 'std': np.nan, 'median': np.nan,
                    'min': np.nan, 'max': np.nan, 'count': 0
                }
        
        return summary
    
    def plot_metrics_distribution(self, metrics: Dict[str, List[float]], 
                                save_path: Optional[str] = None) -> None:
        """
        Plot distribution of evaluation metrics
        Args:
            metrics: Dictionary with lists of metrics
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        metric_names = ['pesq', 'stoi', 'si_sdr', 'sdr', 'sir', 'sar', 
                       'lsd', 'spectral_convergence', 'magnitude_error']
        
        for i, metric_name in enumerate(metric_names):
            if metric_name in metrics and i < len(axes):
                values = [v for v in metrics[metric_name] if not np.isnan(v) and not np.isinf(v)]
                
                if values:
                    axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{metric_name.upper()} Distribution')
                    axes[i].set_xlabel(metric_name.upper())
                    axes[i].set_ylabel('Count')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add statistics text
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    axes[i].axvline(mean_val, color='red', linestyle='--', 
                                  label=f'Mean: {mean_val:.3f}')
                    axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Helper functions
def load_audio_pairs(clean_files: List[str], enhanced_files: List[str], 
                    sample_rate: int = 16000) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load pairs of clean and enhanced audio files
    Args:
        clean_files: List of clean audio file paths
        enhanced_files: List of enhanced audio file paths
        sample_rate: Target sample rate
    Returns:
        List of (clean, enhanced) audio pairs
    """
    pairs = []
    
    for clean_file, enhanced_file in zip(clean_files, enhanced_files):
        try:
            clean_audio, _ = librosa.load(clean_file, sr=sample_rate)
            enhanced_audio, _ = librosa.load(enhanced_file, sr=sample_rate)
            pairs.append((clean_audio, enhanced_audio))
        except Exception as e:
            print(f"Error loading pair {clean_file}, {enhanced_file}: {e}")
            continue
    
    return pairs

if __name__ == "__main__":
    # Test the evaluator
    evaluator = SpeechEnhancementEvaluator(sample_rate=16000)
    
    # Generate test signals
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean signal (sine wave)
    clean_signal = np.sin(2 * np.pi * 440 * t)
    
    # Add noise to create "enhanced" signal
    noise = 0.1 * np.random.randn(len(clean_signal))
    enhanced_signal = clean_signal + noise
    
    # Evaluate
    metrics = evaluator.evaluate_pair(clean_signal, enhanced_signal)
    
    print("Test Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("✅ Evaluation module test passed!")
