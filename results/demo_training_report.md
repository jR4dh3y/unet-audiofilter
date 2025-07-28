
# Demo Training Report - Speech Enhancement U-Net

## Model Configuration (Reduced for Demo)
- Architecture: U-Net with 3 layers (reduced from 4)  
- Base filters: 32 (reduced from 64)
- Dropout: 0.2
- Total parameters: 1,927,841
- Device: cpu (switched from GPU due to memory constraints)

## Training Configuration
- Batch size: 1 (reduced for memory)
- Learning rate: 0.001
- Epochs completed: 1 (demo run)
- Training samples processed: 5 batches
- Validation samples processed: 3 batches

## Demo Results
- Training loss: 1691.65
- Validation loss: 1755.85
- Model shows initial learning capability
- Note: Performance not optimal due to limited training

## Memory Constraints Encountered
- Original GPU memory: 3.7 GB
- Model too large for available GPU memory
- Switched to CPU training for demonstration
- Reduced model size and batch size for feasibility

## Recommendations for Full Training
1. Use machine with larger GPU memory (8GB+ recommended)
2. Implement gradient checkpointing for memory efficiency
3. Use mixed precision training (float16)
4. Consider data streaming for large datasets
5. Train for full epochs (50-100) for convergence

## Dataset Information  
- Training subset: 500 files (reduced from full dataset)
- Validation subset: 50 files
- Spectrogram dimensions: ~257 x 150 (varies by file)
- Sample rate: 16kHz
- STFT parameters: n_fft=512, hop_length=256

## Technical Notes
- Custom collate function handles variable-length spectrograms
- SpectralLoss combines MSE, L1, and spectral convergence
- Model architecture based on encoder-decoder U-Net
- Data augmentation includes frequency and time masking
