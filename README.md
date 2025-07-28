# AI Speech Enhancement - Clean Voice

A GPU-accelerated deep learning system for speech enhancement using U-Net architecture.

## 🎯 Project Overview

This project implements a complete pipeline for removing background noise from speech recordings using a U-Net model trained on the VoiceBank+DEMAND dataset.

## 📁 Project Structure

```
ai-clrvoice/
├── src/
│   ├── __init__.py
│   └── model.py              # U-Net model implementation
├── notebooks/
│   ├── gpu.ipynb             # Working GPU training notebook
│   └── 07_gpu_training.ipynb # Complete GPU training pipeline
├── streamlit_app/
│   ├── app.py               # Web interface for audio enhancement
│   └── requirements.txt     # Streamlit dependencies
├── models/
│   └── best_gpu_model.pth   # Trained model checkpoint
├── dataset/                 # VoiceBank+DEMAND dataset
├── results/                 # Training results and enhanced audio
├── inference_gpu.py         # Standalone inference script
├── test_quality.py         # Quality assessment tools
└── requirements.txt        # Project dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Use the Web Interface
```bash
cd streamlit_app
streamlit run app.py
```

### 3. Command Line Inference
```bash
python inference_gpu.py input_noisy.wav output_enhanced.wav
```

## 🧠 Model Architecture

- **U-Net Architecture**: Encoder-decoder with skip connections
- **Parameters**: 1,927,841 trainable parameters
- **Input**: Magnitude spectrograms (513 x 128)
- **Output**: Enhanced magnitude spectrograms
- **Training**: GPU-accelerated with chunked data loading

### Model Configuration:
- Base filters: 32
- Depth: 3 layers
- Dropout: 0.1
- Sample rate: 16kHz
- STFT: n_fft=1024, hop_length=256

## 📊 Performance

- **Validation Loss**: 0.000071
- **Average Improvement**: 56.3%
- **SNR Improvement**: +5.84 dB on test samples
- **Processing**: Real-time capable on GPU

## 🛠️ Training

To retrain the model, use the `gpu.ipynb` notebook:

1. Open `notebooks/gpu.ipynb`
2. Run all cells in sequence
3. Model will be saved to `models/best_gpu_model.pth`

### Training Features:
- Chunked data loading for memory efficiency
- GPU optimization with CUDA
- Early stopping and checkpointing
- Robust error handling and recovery

## 📈 Usage Examples

### Web Interface
1. Upload a noisy audio file
2. Click "Enhance Audio"
3. Listen to results and download enhanced audio
4. View waveform and spectrogram comparisons

### Command Line
```bash
# Enhance a single file
python inference_gpu.py noisy_speech.wav clean_speech.wav

# Quality assessment
python test_quality.py
```

## 🎵 Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- M4A

## 🔧 Technical Details

### Data Processing:
- 4-second chunks with 25% overlap
- STFT-based magnitude processing
- Min-max normalization
- Crossfading for chunk blending

### GPU Optimization:
- CUDA memory management
- Batch processing
- Gradient clipping
- Mixed precision support

## 📝 Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)
- 4GB+ GPU memory for training
- 16GB+ RAM for dataset loading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- VoiceBank+DEMAND dataset for training data
- U-Net architecture for speech enhancement
- PyTorch team for the deep learning framework
- Streamlit for the web interface framework
