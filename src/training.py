#!/usr/bin/env python3
"""
GPU Training Module for Speech Enhancement
Extracted and refactored from gpu.ipynb
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from tqdm import tqdm
import time
import json
import random

# Python 3.13+ compatibility
from .compat import setup_aifc_compatibility
setup_aifc_compatibility()
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('/home/radhey/code/ai-clrvoice')
from src.model import UNet


class ChunkedSpeechDataset(Dataset):
    """Memory-efficient dataset with on-the-fly chunked loading"""
    
    def __init__(self, clean_dir, noisy_dir, file_list, 
                 chunk_duration=4.0, sample_rate=16000, 
                 n_fft=1024, hop_length=256, overlap=0.25):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.overlap = overlap
        
        # Filter valid files
        self.valid_files = []
        for file_id in file_list:
            clean_path = os.path.join(clean_dir, f"{file_id}.wav")
            noisy_path = os.path.join(noisy_dir, f"{file_id}.wav")
            if os.path.exists(clean_path) and os.path.exists(noisy_path):
                self.valid_files.append(file_id)
        
        print(f"Dataset initialized with {len(self.valid_files)} files")
        
        # Pre-calculate chunks per file for indexing
        self.chunks_info = []
        for file_id in self.valid_files:
            clean_path = os.path.join(clean_dir, f"{file_id}.wav")
            try:
                # Quick duration check without loading full audio
                info = sf.info(clean_path)
                duration = info.frames / info.samplerate
                
                # Calculate number of chunks
                if duration >= chunk_duration:
                    hop_duration = chunk_duration * (1 - overlap)
                    num_chunks = max(1, int((duration - chunk_duration) / hop_duration) + 1)
                    
                    for chunk_idx in range(num_chunks):
                        self.chunks_info.append((file_id, chunk_idx, num_chunks))
                else:
                    # Use full file if shorter than chunk duration
                    self.chunks_info.append((file_id, 0, 1))
            except Exception as e:
                print(f"Skipping {file_id}: {e}")
                continue
        
        print(f"Total chunks: {len(self.chunks_info)}")
    
    def __len__(self):
        return len(self.chunks_info)
    
    def __getitem__(self, idx):
        file_id, chunk_idx, total_chunks = self.chunks_info[idx]
        
        # Load audio files
        clean_path = os.path.join(self.clean_dir, f"{file_id}.wav")
        noisy_path = os.path.join(self.noisy_dir, f"{file_id}.wav")
        
        try:
            clean_audio, _ = librosa.load(clean_path, sr=self.sample_rate)
            noisy_audio, _ = librosa.load(noisy_path, sr=self.sample_rate)
            
            # Extract chunk
            if total_chunks > 1:
                hop_samples = int(self.chunk_samples * (1 - self.overlap))
                start_idx = chunk_idx * hop_samples
                end_idx = start_idx + self.chunk_samples
                
                clean_chunk = clean_audio[start_idx:end_idx]
                noisy_chunk = noisy_audio[start_idx:end_idx]
                
                # Pad if necessary
                if len(clean_chunk) < self.chunk_samples:
                    pad_length = self.chunk_samples - len(clean_chunk)
                    clean_chunk = np.pad(clean_chunk, (0, pad_length), mode='constant')
                    noisy_chunk = np.pad(noisy_chunk, (0, pad_length), mode='constant')
            else:
                clean_chunk = clean_audio
                noisy_chunk = noisy_audio
                
                # Pad to chunk size if needed
                if len(clean_chunk) < self.chunk_samples:
                    pad_length = self.chunk_samples - len(clean_chunk)
                    clean_chunk = np.pad(clean_chunk, (0, pad_length), mode='constant')
                    noisy_chunk = np.pad(noisy_chunk, (0, pad_length), mode='constant')
                elif len(clean_chunk) > self.chunk_samples:
                    clean_chunk = clean_chunk[:self.chunk_samples]
                    noisy_chunk = noisy_chunk[:self.chunk_samples]
            
            # Convert to spectrograms
            noisy_spec = self._audio_to_spec(noisy_chunk)
            clean_spec = self._audio_to_spec(clean_chunk)
            
            return torch.FloatTensor(noisy_spec), torch.FloatTensor(clean_spec)
            
        except Exception as e:
            print(f"Error loading {file_id}: {e}")
            # Return zeros as fallback
            spec_shape = (self.n_fft // 2 + 1, (self.chunk_samples // self.hop_length) + 1)
            return torch.zeros(spec_shape), torch.zeros(spec_shape)
    
    def _audio_to_spec(self, audio):
        """Convert audio to magnitude spectrogram"""
        stft = librosa.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.n_fft,
            window='hann',
            center=True
        )
        magnitude = np.abs(stft)
        
        # Normalize to [0, 1]
        magnitude = magnitude / (np.max(magnitude) + 1e-8)
        
        return magnitude


class GPUTrainer:
    """GPU Training class for speech enhancement model"""
    
    def __init__(self, config=None):
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Clear GPU cache and enable optimizations
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        # Training configuration
        self.config = config or {
            'batch_size': 8 if self.device.type == 'cuda' else 4,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'patience': 10,
            'checkpoint_freq': 5,
            'model_dir': '/home/radhey/code/ai-clrvoice/models',
            'results_dir': '/home/radhey/code/ai-clrvoice/results'
        }
        
        # Create directories
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['results_dir'], exist_ok=True)
    
    def load_file_list(self, scp_path):
        """Load file list from .scp file"""
        with open(scp_path, 'r') as f:
            files = [line.strip().replace('.wav', '') for line in f.readlines()]
        return files
    
    def create_datasets(self):
        """Create training and validation datasets"""
        # Load file lists
        train_files = self.load_file_list('/home/radhey/code/ai-clrvoice/dataset/train.scp')
        
        # Split training files for validation (90% train, 10% val)
        random.shuffle(train_files)
        split_idx = int(0.9 * len(train_files))
        train_files_final = train_files[:split_idx]
        val_files = train_files[split_idx:]
        
        print(f"Training files: {len(train_files_final)}")
        print(f"Validation files: {len(val_files)}")
        
        # Create datasets
        train_dataset = ChunkedSpeechDataset(
            clean_dir='/home/radhey/code/ai-clrvoice/dataset/clean_trainset_28spk_wav',
            noisy_dir='/home/radhey/code/ai-clrvoice/dataset/noisy_trainset_28spk_wav',
            file_list=train_files_final,
            chunk_duration=4.0,
            overlap=0.25
        )
        
        val_dataset = ChunkedSpeechDataset(
            clean_dir='/home/radhey/code/ai-clrvoice/dataset/clean_trainset_28spk_wav',
            noisy_dir='/home/radhey/code/ai-clrvoice/dataset/noisy_trainset_28spk_wav',
            file_list=val_files,
            chunk_duration=4.0,
            overlap=0.25
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=4,
            pin_memory=self.device.type == 'cuda',
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=2,
            pin_memory=self.device.type == 'cuda',
            drop_last=False
        )
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader, val_dataset
    
    def create_model(self):
        """Create and initialize model"""
        model = UNet(
            in_channels=1,
            out_channels=1,
            base_filters=32,  # Memory efficient
            depth=3,
            dropout=0.1
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (noisy, clean) in enumerate(pbar):
            # Move to device
            noisy = noisy.unsqueeze(1).to(self.device)
            clean = clean.unsqueeze(1).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            enhanced = model(noisy)
            loss = criterion(enhanced, clean)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            # Clear GPU cache periodically
            if self.device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for noisy, clean in pbar:
                noisy = noisy.unsqueeze(1).to(self.device)
                clean = clean.unsqueeze(1).to(self.device)
                
                enhanced = model(noisy)
                loss = criterion(enhanced, clean)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def train(self, model, train_loader, val_loader):
        """Main training loop"""
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'], 
                             weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training state
        train_losses = []
        val_losses = []
        learning_rates = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start_time = time.time()
            print(f"\n=== Epoch {epoch + 1}/{self.config['num_epochs']} ===")
            
            # Training and validation
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self.validate_epoch(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            learning_rates.append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.8f}")
            print(f"Epoch Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'learning_rates': learning_rates
                }
                
                torch.save(checkpoint, os.path.join(self.config['model_dir'], 'best_gpu_model.pth'))
                print(f"✓ New best model saved! (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{self.config['patience']}")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['checkpoint_freq'] == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'learning_rates': learning_rates
                }
                torch.save(checkpoint, os.path.join(self.config['model_dir'], f'checkpoint_epoch_{epoch + 1}.pth'))
                print(f"Checkpoint saved at epoch {epoch + 1}")
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        print(f"\n=== Training Complete ===")
        print(f"Total training time: {total_time / 3600:.1f} hours")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'best_val_loss': best_val_loss,
            'total_time': total_time
        }
    
    def save_training_plots(self, history):
        """Save training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(history['train_losses'], label='Training Loss', color='blue', linewidth=2)
        axes[0, 0].plot(history['val_losses'], label='Validation Loss', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Learning rate schedule
        axes[0, 1].plot(history['learning_rates'], color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Loss improvement
        if len(history['train_losses']) > 1:
            train_improvement = [(history['train_losses'][0] - loss) / history['train_losses'][0] * 100 
                               for loss in history['train_losses']]
            val_improvement = [(history['val_losses'][0] - loss) / history['val_losses'][0] * 100 
                             for loss in history['val_losses']]
            
            axes[1, 0].plot(train_improvement, label='Training Improvement', color='blue', linewidth=2)
            axes[1, 0].plot(val_improvement, label='Validation Improvement', color='red', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Improvement (%)')
            axes[1, 0].set_title('Loss Improvement Over Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting monitor
        if len(history['train_losses']) == len(history['val_losses']):
            loss_diff = [v - t for v, t in zip(history['val_losses'], history['train_losses'])]
            axes[1, 1].plot(loss_diff, color='orange', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Val Loss - Train Loss')
            axes[1, 1].set_title('Overfitting Monitor')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config['results_dir'], 'gpu_training_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training plots saved to: {plot_path}")


def main():
    """Main training function"""
    # Initialize trainer
    trainer = GPUTrainer()
    
    # Create datasets
    train_loader, val_loader, val_dataset = trainer.create_datasets()
    
    # Create model
    model = trainer.create_model()
    
    # Train model
    history = trainer.train(model, train_loader, val_loader)
    
    # Save training plots
    trainer.save_training_plots(history)
    
    # Save training history
    history_path = os.path.join(trainer.config['results_dir'], 'gpu_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to: {history_path}")
    print(f"Best model saved to: {os.path.join(trainer.config['model_dir'], 'best_gpu_model.pth')}")
    print("✓ Training complete!")


if __name__ == "__main__":
    main()
