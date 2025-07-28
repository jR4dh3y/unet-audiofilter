#!/usr/bin/env python3
"""System Testing - Simplified Version"""

import os
import sys
import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path

sys.path.append('/home/radhey/code/ai-clrvoice')

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
    
    print(f"Loss: {loss.item():.6f}")
    print("Basic model test passed!")

def main():
    """Run basic system tests"""
    print("Speech Enhancement System Test - Simplified")
    print("=" * 50)
    print("This is a simplified test suite.")
    print("   Full testing capabilities are available in:")
    print("   notebooks/main_training.ipynb")
    print("   - Model training and validation")
    print("   - Audio quality metrics")
    print("   - Before/after comparisons")
    print("   - Interactive testing tools")
    print("=" * 50)
    
    try:
        basic_model_test()
        print("\nBasic tests completed successfully!")
        print("\nFor comprehensive testing:")
        print("   1. Open notebooks/main_training.ipynb")
        print("   2. Run the evaluation sections")
        print("   3. Use the interactive testing tools")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
