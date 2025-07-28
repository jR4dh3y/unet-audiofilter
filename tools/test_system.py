#!/usr/bin/env python3
"""System Testing Tool - Enhanced with Global Path Configuration"""

import os
import sys
import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path

from config.paths import PATHS, get_path
from src.unet_model import UNet, SpectralLoss
from src.utils import get_device, count_parameters

def basic_model_test():
    """Test basic model functionality"""
    print("Testing UNet Model...")
    
    device = get_device()
    print(f"Device: {device}")
    
    model = UNet(n_fft=1024, hop_length=256).to(device)
    
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")
    
    batch_size = 2
    freq_bins = 513
    time_frames = 128
    
    test_input = torch.rand(batch_size, 1, freq_bins, time_frames).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
        
    print(f"Output shape: {output.shape}")
    
    criterion = SpectralLoss().to(device)
    target = torch.rand_like(output)
    loss = criterion(output, target)
    
    print(f"Loss: {loss[0].item():.6f}")
    print("Basic model test passed!")

def test_paths():
    """Test global path configuration"""
    print("Testing Global Path Configuration...")
    print("=" * 40)
    
    # Test critical paths
    critical_paths = [
        ('Project Base', 'base'),
        ('Source Code', 'src'),
        ('Models Directory', 'models.base'),
        ('Dataset Directory', 'dataset.base'),
        ('Results Directory', 'results.base'),
        ('Best Model', 'models.best_model'),
    ]
    
    for name, path_key in critical_paths:
        try:
            path = get_path(path_key)
            status = "✓ EXISTS" if path.exists() else "✗ MISSING"
            print(f"{name:20}: {status} - {path}")
        except Exception as e:
            print(f"{name:20}: ✗ ERROR - {e}")
    
    print("=" * 40)

def main():
    """Run system tests with global path configuration"""
    print("Speech Enhancement System Test - Enhanced")
    print("=" * 50)
    print(f"Project Root: {get_path('base')}")
    print(f"Configuration: Using global path system")
    print("=" * 50)
    
    try:
        # Test path configuration first
        test_paths()
        print()
        
        # Test basic model functionality
        basic_model_test()
        
        print("\n" + "=" * 50)
        print("✓ All basic tests completed successfully!")
        print("\nFor comprehensive testing:")
        print(f"   1. Open {get_path('notebooks') / 'main_training.ipynb'}")
        print("   2. Run the evaluation sections")
        print("   3. Use the interactive testing tools")
        print(f"\nTest results will be saved to: {get_path('results.base')}")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that all paths in config/paths.py are correct")
        print("2. Ensure all required directories exist")
        print("3. Verify model files are present")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
