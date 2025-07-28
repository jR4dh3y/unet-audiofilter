
import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

class STFTProcessor:
    """STFT processing for speech enhancement"""

    def __init__(self, config):
        self.n_fft = config['dataset']['n_fft']
        self.hop_length = config['dataset']['hop_length']
        self.win_length = config['dataset']['win_length']
        self.window = config['dataset']['window']
        self.center = config['dataset']['center']
        self.pad_mode = config['dataset']['pad_mode']

    def stft(self, audio):
        """Compute STFT of audio signal"""
        stft_complex = librosa.stft(
            audio, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode
        )
        return stft_complex

    def istft(self, stft_complex):
        """Inverse STFT to reconstruct audio"""
        audio = librosa.istft(
            stft_complex,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center
        )
        return audio

    def magnitude_phase(self, stft_complex):
        """Extract magnitude and phase"""
        magnitude = np.abs(stft_complex)
        phase = np.angle(stft_complex)
        return magnitude, phase

    def reconstruct_complex(self, magnitude, phase):
        """Reconstruct complex STFT from magnitude and phase"""
        return magnitude * np.exp(1j * phase)


class SpeechEnhancementDataset(Dataset):
    """Dataset class for speech enhancement"""

    def __init__(self, file_list, clean_dir, noisy_dir, config, transform=None):
        self.file_list = file_list
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.config = config
        self.transform = transform
        self.stft_processor = STFTProcessor(config)
        self.sample_rate = config['dataset']['sample_rate']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]

        # Load audio files
        clean_path = os.path.join(self.clean_dir, f"{file_id}.wav")
        noisy_path = os.path.join(self.noisy_dir, f"{file_id}.wav")

        clean_audio, _ = librosa.load(clean_path, sr=self.sample_rate)
        noisy_audio, _ = librosa.load(noisy_path, sr=self.sample_rate)

        # Ensure same length
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        # Compute STFT
        clean_stft = self.stft_processor.stft(clean_audio)
        noisy_stft = self.stft_processor.stft(noisy_audio)

        # Get magnitude spectrograms
        clean_mag, clean_phase = self.stft_processor.magnitude_phase(clean_stft)
        noisy_mag, noisy_phase = self.stft_processor.magnitude_phase(noisy_stft)

        # Normalize (log scale)
        clean_mag_db = librosa.amplitude_to_db(clean_mag + 1e-8)
        noisy_mag_db = librosa.amplitude_to_db(noisy_mag + 1e-8)

        # Convert to tensors
        noisy_tensor = torch.FloatTensor(noisy_mag_db).unsqueeze(0)
        clean_tensor = torch.FloatTensor(clean_mag_db).unsqueeze(0)

        # Apply transforms if specified
        if self.transform:
            noisy_tensor = self.transform(noisy_tensor)
            clean_tensor = self.transform(clean_tensor)

        return {
            'noisy': noisy_tensor,
            'clean': clean_tensor,
            'noisy_phase': torch.FloatTensor(noisy_phase),
            'file_id': file_id
        }
