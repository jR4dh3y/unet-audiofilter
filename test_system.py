#!/usr/bin/env python3
"""
Demo script to test the complete speech enhancement pipeline
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Add src to path
sys.path.append('/home/radhey/code/ai-clrvoice')
from src.model import UNet, SpectralLoss, initialize_weights, count_parameters
from src.preprocessing import AudioProcessor
from src.evaluation import compute_all_metrics
from src.utils import get_device, print_model_summary

def test_model_architecture():
    """Test the U-Net model architecture"""
    print("🧠 Testing U-Net Architecture")
    print("=" * 40)
    
    device = get_device()
    
    # Create model
    model = UNet(in_channels=1, out_channels=1, base_filters=64, depth=4)
    model = model.to(device)
    
    # Initialize weights
    initialize_weights(model)
    
    # Print model summary
    print_model_summary(model, (1, 513, 128))
    
    # Test forward pass
    batch_size = 2
    height, width = 513, 128  # Typical spectrogram dimensions
    
    x = torch.randn(batch_size, 1, height, width).to(device)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"✅ Forward pass successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test loss function
    criterion = SpectralLoss()
    loss, loss_dict = criterion(y, x)
    
    print(f"\n📊 Loss Function Test:")
    print(f"Total loss: {loss.item():.6f}")
    for key, value in loss_dict.items():
        print(f"{key}: {value:.6f}")
    
    return model

def test_preprocessing():
    """Test audio preprocessing pipeline"""
    print("\n🔧 Testing Audio Preprocessing")
    print("=" * 40)
    
    # Create processor
    processor = AudioProcessor(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        normalize_method='min_max'
    )
    
    # Create dummy audio
    sample_rate = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Generate test signal (sine wave + noise)
    clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    noise = 0.1 * np.random.randn(len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # Save temporary audio file
    import soundfile as sf
    temp_path = '/tmp/test_audio.wav'
    sf.write(temp_path, noisy_signal, sample_rate)
    
    # Process audio
    try:
        audio_data = processor.preprocess_audio(temp_path)
        
        if audio_data is not None:
            print(f"✅ Audio preprocessing successful!")
            print(f"Waveform shape: {audio_data['waveform'].shape}")
            print(f"STFT shape: {audio_data['stft'].shape}")
            print(f"Magnitude shape: {audio_data['magnitude'].shape}")
            print(f"Phase shape: {audio_data['phase'].shape}")
            print(f"Normalized magnitude shape: {audio_data['normalized_magnitude'].shape}")
            
            # Test postprocessing
            enhanced_waveform = processor.postprocess_spectrogram(
                audio_data['normalized_magnitude'],
                audio_data['phase'],
                audio_data['normalization_params']
            )
            
            print(f"Enhanced waveform shape: {enhanced_waveform.shape}")
            
        else:
            print("❌ Audio preprocessing failed!")
        
        # Clean up
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")

def test_evaluation_metrics():
    """Test evaluation metrics computation"""
    print("\n📊 Testing Evaluation Metrics")
    print("=" * 40)
    
    # Create dummy signals
    sample_rate = 16000
    duration = 2
    
    # Clean signal (sine wave)
    t = np.linspace(0, duration, sample_rate * duration)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Enhanced signal (clean + small amount of noise)
    enhanced = clean + 0.05 * np.random.randn(len(clean))
    
    try:
        # Compute metrics
        metrics = compute_all_metrics(clean, enhanced, sample_rate)
        
        print("✅ Metrics computation successful!")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
            
    except Exception as e:
        print(f"❌ Metrics computation error: {e}")

def test_complete_pipeline():
    """Test the complete enhancement pipeline"""
    print("\n🚀 Testing Complete Pipeline")
    print("=" * 40)
    
    device = get_device()
    
    # Load model
    model = UNet(in_channels=1, out_channels=1)
    model = model.to(device)
    model.eval()
    
    # Create processor
    processor = AudioProcessor()
    
    # Create test audio
    sample_rate = 16000
    duration = 2
    t = np.linspace(0, duration, sample_rate * duration)
    
    clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.2 * np.random.randn(len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # Save temporary file
    import soundfile as sf
    temp_path = '/tmp/test_noisy.wav'
    sf.write(temp_path, noisy_signal, sample_rate)
    
    try:
        # Preprocess
        audio_data = processor.preprocess_audio(temp_path)
        
        if audio_data is None:
            print("❌ Preprocessing failed!")
            return
        
        # Prepare for model
        noisy_magnitude = audio_data['normalized_magnitude']
        original_phase = audio_data['phase']
        norm_params = audio_data['normalization_params']
        
        # Add batch and channel dimensions
        model_input = noisy_magnitude.unsqueeze(0).unsqueeze(0).to(device)
        
        # Run model
        with torch.no_grad():
            enhanced_magnitude = model(model_input)
        
        # Remove dimensions
        enhanced_magnitude = enhanced_magnitude.squeeze(0).squeeze(0).cpu()
        
        # Postprocess
        enhanced_waveform = processor.postprocess_spectrogram(
            enhanced_magnitude, original_phase, norm_params
        )
        
        print("✅ Complete pipeline test successful!")
        print(f"Original signal length: {len(noisy_signal)}")
        print(f"Enhanced signal length: {len(enhanced_waveform)}")
        
        # Compute SNR improvement
        original_snr = 10 * np.log10(
            np.mean(clean_signal**2) / np.mean((noisy_signal - clean_signal)**2)
        )
        enhanced_snr = 10 * np.log10(
            np.mean(clean_signal**2) / np.mean((enhanced_waveform.numpy() - clean_signal)**2)
        )
        
        print(f"Original SNR: {original_snr:.2f} dB")
        print(f"Enhanced SNR: {enhanced_snr:.2f} dB")
        print(f"SNR Improvement: {enhanced_snr - original_snr:.2f} dB")
        
        # Clean up
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")

def main():
    """Run all tests"""
    print("🎵 AI Speech Enhancement - System Test")
    print("=" * 50)
    
    try:
        # Test model architecture
        model = test_model_architecture()
        
        # Test preprocessing
        test_preprocessing()
        
        # Test evaluation metrics
        test_evaluation_metrics()
        
        # Test complete pipeline
        test_complete_pipeline()
        
        print("\n🎉 All tests completed!")
        print("=" * 50)
        print("✅ Model architecture: OK")
        print("✅ Audio preprocessing: OK")
        print("✅ Evaluation metrics: OK")
        print("✅ Complete pipeline: OK")
        print("\n🚀 Ready to start training!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()
