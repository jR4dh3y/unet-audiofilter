# AI Clear Voice - Clean Modular Architecture

## Overview

This is a refactored version of the AI Clear Voice speech enhancement project, extracted from the original `gpu.ipynb` notebook into a clean, modular Python codebase. The system uses a U-Net architecture for GPU-accelerated speech enhancement with ffmpeg-based audio I/O for maximum compatibility.

## Key Features

- **Python 3.13+ Compatible**: Uses ffmpeg for audio I/O instead of deprecated aifc module
- **Modular Architecture**: Clean separation of training, inference, and utilities
- **GPU Optimized**: CUDA support with chunked processing for memory efficiency
- **High Quality Enhancement**: U-Net model with 1.9M parameters
- **Multiple Audio Formats**: Supports any format that ffmpeg can handle

## Project Structure

```
ai-clrvoice/
├── src/                          # Core modules
│   ├── model.py                  # U-Net architecture
│   ├── training.py               # GPU training pipeline
│   ├── audio_io.py               # ffmpeg-based audio I/O
│   ├── dataset.py                # Dataset handling
│   ├── evaluation.py             # Quality metrics
│   ├── preprocessing.py          # Audio preprocessing
│   └── utils.py                  # Utilities
├── models/                       # Trained models
│   └── best_gpu_model.pth        # Best trained model
├── dataset/                      # Training/test data
├── results/                      # Output results
├── inference_gpu.py              # Standalone inference
├── train_gpu.py                  # Simple training script
├── test_quality.py               # Quality assessment
└── requirements.txt              # Dependencies
```

## System Requirements

### Prerequisites
- Python 3.13+
- CUDA-capable GPU (recommended)
- ffmpeg (for audio I/O)

### Install ffmpeg
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Audio Enhancement (Inference)
```bash
# Enhance a single audio file
python inference_gpu.py input_noisy.wav output_enhanced.wav

# Specify model and device
python inference_gpu.py input.wav output.wav --model models/best_gpu_model.pth --device cuda
```

### 2. Model Training
```bash
# Train new model with default settings
python train_gpu.py

# The training will use the dataset/ folder structure
```

### 3. Quality Assessment
```bash
# Evaluate enhancement quality
python test_quality.py
```

## Audio I/O Architecture

### ffmpeg Integration
The system uses ffmpeg for all audio operations, providing:
- **Universal Format Support**: WAV, MP3, FLAC, M4A, etc.
- **Python 3.13+ Compatibility**: No dependency on deprecated aifc module
- **Robust Processing**: Handles various sample rates and bit depths
- **Fallback Support**: scipy.io.wavfile as backup for WAV files

### Usage Example
```python
from src.audio_io import load_audio, save_audio

# Load any audio format
audio, sr = load_audio('input.mp3', sample_rate=16000)

# Save in any format ffmpeg supports
save_audio(audio, 'output.wav', sample_rate=16000)
```

## Model Architecture

### U-Net Details
- **Input/Output**: Single channel spectrograms
- **Parameters**: 1,927,841 total
- **Architecture**: 3-level depth, 32 base filters
- **Regularization**: 0.1 dropout rate
- **Training Loss**: ~0.000071 validation loss

### Processing Pipeline
1. **Audio Loading**: ffmpeg → numpy array
2. **STFT**: Convert to magnitude/phase spectrograms
3. **Enhancement**: U-Net processes magnitude
4. **Reconstruction**: ISTFT with enhanced magnitude + original phase
5. **Chunked Processing**: 4-second chunks with 25% overlap for long audio

## Training Pipeline

### ChunkedSpeechDataset
- **Memory Efficient**: Loads audio on-demand
- **Flexible**: Supports various dataset structures
- **Robust**: Handles different audio formats via ffmpeg

### GPUTrainer Features
- **Progress Tracking**: tqdm progress bars
- **Visualization**: Training curves and spectrograms
- **Checkpointing**: Automatic model saving
- **Validation**: Real-time quality metrics

### Training Configuration
```python
trainer = GPUTrainer(
    train_dir='dataset/noisy_trainset_28spk_wav',
    clean_dir='dataset/clean_trainset_28spk_wav',
    batch_size=16,
    learning_rate=0.001,
    num_epochs=20
)
```

## Performance Metrics

### Quality Measures
- **PESQ**: Perceptual evaluation of speech quality
- **STOI**: Short-time objective intelligibility
- **SNR**: Signal-to-noise ratio improvement
- **MSE**: Mean squared error reduction

### Typical Results
- **SNR Improvement**: 3-6 dB average
- **MSE Reduction**: 40-70% typical
- **Processing Speed**: Real-time on modern GPUs
- **Memory Usage**: <2GB VRAM for inference

## Compatibility Notes

### Python 3.13+ Changes
- **aifc module removed**: Replaced with ffmpeg + scipy fallbacks
- **Updated dependencies**: librosa >=0.10.0, scipy for WAV fallback
- **Error handling**: Graceful fallbacks for audio format issues

### Migration from soundfile
- **Old**: `sf.read()` / `sf.write()`
- **New**: `load_audio()` / `save_audio()` from `src.audio_io`
- **Benefits**: More formats, better error handling, no aifc dependency

## Troubleshooting

### Common Issues

1. **ffmpeg not found**
   ```bash
   # Install ffmpeg system-wide
   sudo apt install ffmpeg  # Ubuntu/Debian
   brew install ffmpeg      # macOS
   ```

2. **CUDA out of memory**
   ```python
   # Reduce batch size in training
   trainer = GPUTrainer(batch_size=8)  # Instead of 16
   ```

3. **Audio format errors**
   ```python
   # The system will automatically fallback to scipy for WAV files
   # Ensure ffmpeg supports your input format
   ```

### Performance Optimization
- **Use CUDA**: 10-50x faster than CPU
- **Batch processing**: Process multiple files together
- **Chunked inference**: For very long audio files
- **Memory monitoring**: Use `nvidia-smi` to monitor GPU usage

## Development

### Adding New Features
1. **Audio Processing**: Extend `src/audio_io.py`
2. **Model Architecture**: Modify `src/model.py`
3. **Training Logic**: Update `src/training.py`
4. **Evaluation Metrics**: Enhance `src/evaluation.py`

### Testing
```bash
# Run quality tests
python test_quality.py

# Test system integration
python test_system.py

# Manual inference test
python inference_gpu.py dataset/noisy_testset_wav/p232_001.wav test_output.wav
```

## License & Credits

This refactored version maintains compatibility with the original project while providing a clean, maintainable codebase suitable for production use and further development.

---

**Note**: This clean architecture eliminates the aifc dependency issues in Python 3.13+ while maintaining full functionality and improving code organization.
