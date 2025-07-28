#!/usr/bin/env python3
"""Speech Enhancement Inference Script"""

import os
import sys
import torch
import numpy as np
import librosa
import argparse
from pathlib import Path

# Import global path configuration
from config.paths import PATHS, get_path
from src.unet_model import UNet
from src.audio_utils import load_audio, save_audio

class GPUSpeechEnhancer:
    """High-quality speech enhancer using trained GPU model"""

    def __init__(self, model_path=None, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Use global path configuration for model if not specified
        if model_path is None:
            model_path = get_path('models.best_model')
        
        print(f"Loading model from: {model_path}")

        # Initialize model with same architecture as training
        self.model = UNet(
            in_channels=1,
            out_channels=1,
            base_filters=32,
            depth=3,
            dropout=0.1
        ).to(self.device)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Audio processing parameters
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.chunk_duration = 4.0  # Process in 4-second chunks
        self.overlap_ratio = 0.25

        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

    def enhance_audio(self, input_path, output_path):
        """Enhance audio file with trained model"""
        print(f"Processing: {input_path}")
        
        # Load audio using ffmpeg
        audio, sr = load_audio(input_path, self.sample_rate)
        if sr != self.sample_rate:
            # Resample if needed
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        print(f"Audio duration: {len(audio)/self.sample_rate:.2f} seconds")
        
        # Process in chunks for memory efficiency
        enhanced_audio = self._process_chunks(audio)

        # Save enhanced audio using ffmpeg
        save_audio(enhanced_audio, output_path, self.sample_rate)
        print(f"Enhanced audio saved: {output_path}")

        return enhanced_audio

    def _process_chunks(self, audio):
        """Process audio in overlapping chunks"""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(chunk_samples * self.overlap_ratio)
        hop_samples = chunk_samples - overlap_samples

        if len(audio) <= chunk_samples:
            # Process entire audio if short
            return self._enhance_chunk(audio)

        # Process overlapping chunks
        enhanced_chunks = []
        for start_idx in range(0, len(audio) - overlap_samples, hop_samples):
            end_idx = min(start_idx + chunk_samples, len(audio))
            chunk = audio[start_idx:end_idx]

            # Pad chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

            enhanced_chunk = self._enhance_chunk(chunk)
            enhanced_chunks.append((enhanced_chunk, start_idx, len(audio[start_idx:end_idx])))

        # Blend overlapping chunks
        return self._blend_chunks(enhanced_chunks, len(audio), overlap_samples)

    def _enhance_chunk(self, audio_chunk):
        """Enhance a single audio chunk"""
        # Convert to spectrogram
        stft = librosa.stft(
            audio_chunk,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Normalize magnitude
        magnitude_normalized = magnitude / (np.max(magnitude) + 1e-8)

        # Model inference
        with torch.no_grad():
            magnitude_tensor = torch.FloatTensor(magnitude_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            enhanced_magnitude = self.model(magnitude_tensor)
            enhanced_magnitude = enhanced_magnitude.cpu().squeeze().numpy()

        # Denormalize
        enhanced_magnitude = enhanced_magnitude * (np.max(magnitude) + 1e-8)

        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(
            enhanced_stft,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )

        return enhanced_audio

    def _blend_chunks(self, enhanced_chunks, total_length, overlap_samples):
        """Blend overlapping chunks with crossfading"""
        enhanced_audio = np.zeros(total_length)
        weight_sum = np.zeros(total_length)

        for enhanced_chunk, start_idx, original_length in enhanced_chunks:
            end_idx = min(start_idx + len(enhanced_chunk), total_length)
            chunk_length = end_idx - start_idx
            
            # Apply crossfading weights
            weights = np.ones(chunk_length)
            if overlap_samples > 0:
                # Fade in
                fade_in_samples = min(overlap_samples, chunk_length // 2)
                weights[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
                
                # Fade out
                fade_out_samples = min(overlap_samples, chunk_length // 2)
                weights[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
            
            enhanced_audio[start_idx:end_idx] += enhanced_chunk[:chunk_length] * weights
            weight_sum[start_idx:end_idx] += weights

        # Normalize
        weight_sum[weight_sum == 0] = 1
        enhanced_audio /= weight_sum

        return enhanced_audio

def main():
    """Main inference function"""

    parser = argparse.ArgumentParser(description='High-Quality Speech Enhancement')
    parser.add_argument('input', help='Input noisy audio file')
    parser.add_argument('output', help='Output enhanced audio file')
    parser.add_argument('--model', default=None,
                       help='Path to trained model (default: use global config)')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda)')

    args = parser.parse_args()

    # Initialize enhancer
    enhancer = GPUSpeechEnhancer(args.model, args.device)

    # Process audio
    enhanced_audio = enhancer.enhance_audio(args.input, args.output)

    print("✓ Enhancement complete!")

if __name__ == '__main__':
    main()
