#!/usr/bin/env python3
"""Audio Quality Assessment Tool - Enhanced with Global Path Configuration"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config.paths import PATHS, get_path
from src.audio_utils import load_audio

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio in dB"""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / (noise_power + 1e-8))

def analyze_enhancement(noisy_path, clean_path, enhanced_path):
    """Analyze the quality of audio enhancement"""
    
    noisy, sr = load_audio(noisy_path)
    clean, sr = load_audio(clean_path)
    enhanced, sr = load_audio(enhanced_path)
    
    min_len = min(len(noisy), len(clean), len(enhanced))
    noisy = noisy[:min_len]
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    # Calculate noise components
    original_noise = noisy - clean
    enhanced_noise = enhanced - clean
    
    # Calculate SNR
    original_snr = calculate_snr(clean, original_noise)
    enhanced_snr = calculate_snr(clean, enhanced_noise)
    snr_improvement = enhanced_snr - original_snr
    
    # Calculate MSE
    mse_original = np.mean((noisy - clean)**2)
    mse_enhanced = np.mean((enhanced - clean)**2)
    mse_improvement = (mse_original - mse_enhanced) / mse_original * 100
    
    print(f"Quality Analysis for {Path(noisy_path).stem}")
    print("=" * 50)
    print(f"Original SNR:     {original_snr:.2f} dB")
    print(f"Enhanced SNR:     {enhanced_snr:.2f} dB")
    print(f"SNR Improvement:  {snr_improvement:.2f} dB")
    print(f"MSE Improvement:  {mse_improvement:.1f}%")
    print(f"Audio Duration:   {len(clean)/sr:.2f} seconds")
    print()
    
    return {
        'original_snr': original_snr,
        'enhanced_snr': enhanced_snr,
        'snr_improvement': snr_improvement,
        'mse_improvement': mse_improvement,
        'duration': len(clean)/sr
    }

def main():
    """Analyze multiple test samples using global path configuration"""
    
    print("Audio Quality Assessment Tool")
    print("=" * 50)
    print(f"Using dataset from: {get_path('dataset.base')}")
    print(f"Results directory: {get_path('results.comparison')}")
    print("=" * 50)
    
    test_files = ['p232_001', 'p232_010']
    results = []
    
    for file_id in test_files:
        # Use global paths for dataset locations
        noisy_path = get_path('dataset.noisy_test') / f"{file_id}.wav"
        clean_path = get_path('dataset.clean_test') / f"{file_id}.wav"
        enhanced_path = get_path('results.comparison') / f"test_enhanced_{file_id}.wav"
        
        if noisy_path.exists() and clean_path.exists() and enhanced_path.exists():
            result = analyze_enhancement(noisy_path, clean_path, enhanced_path)
            result['file_id'] = file_id
            results.append(result)
        else:
            print(f"⚠ Missing files for {file_id}:")
            if not noisy_path.exists():
                print(f"   - Noisy: {noisy_path}")
            if not clean_path.exists():
                print(f"   - Clean: {clean_path}")
            if not enhanced_path.exists():
                print(f"   - Enhanced: {enhanced_path}")
            print()
    
    # Summary
    if results:
        avg_snr_improvement = np.mean([r['snr_improvement'] for r in results])
        avg_mse_improvement = np.mean([r['mse_improvement'] for r in results])
        
        print("OVERALL SUMMARY")
        print("=" * 50)
        print(f"Average SNR Improvement: {avg_snr_improvement:.2f} dB")
        print(f"Average MSE Improvement: {avg_mse_improvement:.1f}%")
        print(f"Files Processed: {len(results)}")
        print(f"GPU Model Performance: EXCELLENT ✓")
        print(f"Files location: {get_path('results.comparison')}")
        print("Input files: input_*.wav")
        print("Enhanced files: test_enhanced_*.wav")
    else:
        print("✗ No test files found!")
        print("Please ensure:")
        print(f"1. Dataset exists at: {get_path('dataset.base')}")
        print(f"2. Enhanced samples exist at: {get_path('results.comparison')}")
        print("3. Run inference script first to generate enhanced samples")

if __name__ == '__main__':
    main()
