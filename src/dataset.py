"""
Dataset classes for speech enhancement training
"""

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
import random
from typing import Tuple, List, Optional

class SpeechEnhancementDataset(Dataset):
    """Dataset for loading paired noisy-clean speech data"""
    
    def __init__(self, 
                 clean_dir: str,
                 noisy_dir: str, 
                 file_list: List[str],
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: Optional[int] = None,
                 segment_length: Optional[int] = None,
                 normalize: str = 'min_max'):
        """
        Initialize dataset
        
        Args:
            clean_dir: Directory containing clean audio files
            noisy_dir: Directory containing noisy audio files
            file_list: List of file IDs (without .wav extension)
            sample_rate: Audio sample rate
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            segment_length: Length of audio segments (samples), None for full files
            normalize: Normalization method ('min_max', 'z_score', or None)
        """
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.file_list = file_list
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.segment_length = segment_length
        self.normalize = normalize
        
        # Filter valid files
        self.valid_files = self._filter_valid_files()
        print(f"Dataset initialized with {len(self.valid_files)} valid files")
    
    def _filter_valid_files(self) -> List[str]:
        """Filter files that exist in both clean and noisy directories"""
        valid_files = []
        
        for file_id in self.file_list:
            clean_path = os.path.join(self.clean_dir, f"{file_id}.wav")
            noisy_path = os.path.join(self.noisy_dir, f"{file_id}.wav")
            
            if os.path.exists(clean_path) and os.path.exists(noisy_path):
                valid_files.append(file_id)
        
        return valid_files
    
    def _load_audio(self, file_path: str) -> torch.Tensor:
        """Load audio file and resample if needed"""
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            return waveform.squeeze(0)
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(self.sample_rate)  # Return silence on error
    
    def _compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute magnitude spectrogram from waveform"""
        # Create window
        window = torch.hann_window(self.win_length)
        
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode='reflect'
        )
        
        # Get magnitude
        magnitude = torch.abs(stft)
        
        return magnitude
    
    def _normalize_spectrogram(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Normalize spectrogram"""
        if self.normalize == 'min_max':
            min_val = torch.min(spectrogram)
            max_val = torch.max(spectrogram)
            normalized = (spectrogram - min_val) / (max_val - min_val + 1e-8)
            params = {'min_val': min_val, 'max_val': max_val}
            
        elif self.normalize == 'z_score':
            mean_val = torch.mean(spectrogram)
            std_val = torch.std(spectrogram)
            normalized = (spectrogram - mean_val) / (std_val + 1e-8)
            params = {'mean_val': mean_val, 'std_val': std_val}
            
        else:
            normalized = spectrogram
            params = {}
        
        return normalized, params
    
    def _get_random_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract random segment from waveform"""
        if self.segment_length is None or len(waveform) <= self.segment_length:
            return waveform
        
        # Pad if too short
        if len(waveform) < self.segment_length:
            padding = self.segment_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Extract random segment
        start_idx = random.randint(0, len(waveform) - self.segment_length)
        segment = waveform[start_idx:start_idx + self.segment_length]
        
        return segment
    
    def __len__(self) -> int:
        return len(self.valid_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample"""
        file_id = self.valid_files[idx]
        
        # Load audio files
        clean_path = os.path.join(self.clean_dir, f"{file_id}.wav")
        noisy_path = os.path.join(self.noisy_dir, f"{file_id}.wav")
        
        clean_audio = self._load_audio(clean_path)
        noisy_audio = self._load_audio(noisy_path)
        
        # Ensure same length
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        
        # Extract segments if specified
        if self.segment_length is not None:
            # Use same random seed for both to get corresponding segments
            random_state = random.getstate()
            clean_segment = self._get_random_segment(clean_audio)
            random.setstate(random_state)
            noisy_segment = self._get_random_segment(noisy_audio)
        else:
            clean_segment = clean_audio
            noisy_segment = noisy_audio
        
        # Compute spectrograms
        clean_spec = self._compute_stft(clean_segment)
        noisy_spec = self._compute_stft(noisy_segment)
        
        # Normalize
        clean_spec_norm, _ = self._normalize_spectrogram(clean_spec)
        noisy_spec_norm, _ = self._normalize_spectrogram(noisy_spec)
        
        # Add channel dimension
        clean_spec_norm = clean_spec_norm.unsqueeze(0)  # (1, freq, time)
        noisy_spec_norm = noisy_spec_norm.unsqueeze(0)  # (1, freq, time)
        
        return noisy_spec_norm, clean_spec_norm

class ValidationDataset(SpeechEnhancementDataset):
    """Dataset for validation - always uses full files without segmentation"""
    
    def __init__(self, *args, **kwargs):
        # Remove segment_length for validation
        kwargs['segment_length'] = None
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get validation sample with file ID"""
        noisy_spec, clean_spec = super().__getitem__(idx)
        file_id = self.valid_files[idx]
        
        return noisy_spec, clean_spec, file_id

def create_data_loaders(config: dict, 
                       train_files: List[str], 
                       val_files: List[str]) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        config: Configuration dictionary
        train_files: List of training file IDs
        val_files: List of validation file IDs
        
    Returns:
        Training and validation data loaders
    """
    # Dataset parameters
    dataset_config = config['dataset']
    training_config = config['training']
    paths_config = config['paths']
    
    # Create datasets
    train_dataset = SpeechEnhancementDataset(
        clean_dir=paths_config['clean_train'],
        noisy_dir=paths_config['noisy_train'],
        file_list=train_files,
        sample_rate=dataset_config['sample_rate'],
        n_fft=dataset_config['n_fft'],
        hop_length=dataset_config['hop_length'],
        win_length=dataset_config['win_length'],
        segment_length=dataset_config.get('segment_length', None),
        normalize='min_max'
    )
    
    val_dataset = ValidationDataset(
        clean_dir=paths_config['clean_train'],  # Use train dir for validation split
        noisy_dir=paths_config['noisy_train'],
        file_list=val_files,
        sample_rate=dataset_config['sample_rate'],
        n_fft=dataset_config['n_fft'],
        hop_length=dataset_config['hop_length'],
        win_length=dataset_config['win_length'],
        normalize='min_max'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Process one file at a time for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """Custom collate function to handle variable-length spectrograms"""
    noisy_specs, clean_specs = zip(*batch)
    
    # Find maximum dimensions
    max_freq = max(spec.shape[1] for spec in noisy_specs)
    max_time = max(spec.shape[2] for spec in noisy_specs)
    
    # Pad all spectrograms to the same size
    padded_noisy = []
    padded_clean = []
    
    for noisy_spec, clean_spec in zip(noisy_specs, clean_specs):
        # Pad frequency dimension
        freq_pad = max_freq - noisy_spec.shape[1]
        time_pad = max_time - noisy_spec.shape[2]
        
        if freq_pad > 0 or time_pad > 0:
            noisy_padded = torch.nn.functional.pad(noisy_spec, (0, time_pad, 0, freq_pad))
            clean_padded = torch.nn.functional.pad(clean_spec, (0, time_pad, 0, freq_pad))
        else:
            noisy_padded = noisy_spec
            clean_padded = clean_spec
        
        padded_noisy.append(noisy_padded)
        padded_clean.append(clean_padded)
    
    # Stack into batches
    noisy_batch = torch.stack(padded_noisy)
    clean_batch = torch.stack(padded_clean)
    
    return noisy_batch, clean_batch

if __name__ == "__main__":
    # Test the dataset
    import yaml
    
    # Load config
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load file lists
    with open('../dataset/train.scp', 'r') as f:
        train_files = [line.strip() for line in f.readlines()]
    
    # Create small test dataset
    test_files = train_files[:10]
    
    dataset = SpeechEnhancementDataset(
        clean_dir=config['paths']['clean_train'],
        noisy_dir=config['paths']['noisy_train'],
        file_list=test_files,
        segment_length=32000  # 2 seconds at 16kHz
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        noisy_spec, clean_spec = dataset[0]
        print(f"Noisy spectrogram shape: {noisy_spec.shape}")
        print(f"Clean spectrogram shape: {clean_spec.shape}")
        print("✅ Dataset test passed!")
    else:
        print("❌ No valid files found in dataset")
