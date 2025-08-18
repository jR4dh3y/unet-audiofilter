# AI Speech Enhancement - Clean Voice

A GPU-accelerated deep learning system for speech enhancement using U-Net architecture.
## Project Overview

This project implements a complete pipeline for removing background noise from speech recordings using a U-Net model trained on the VoiceBank+DEMAND dataset.

## Project Structure

```
ai-clrvoice/
├── config/                       # Global configuration
│   ├── __init__.py
│   └── paths.py                  # Global path configuration (auto-detects root)
├── src/                          # Core components
│   ├── __init__.py               
│   ├── unet_model.py             # U-Net architecture & loss functions
│   ├── audio_utils.py            # Audio I/O utilities
│   └── utils.py                  # Helper functions 
├── scripts/                      # Main execution scripts
│   ├── __init__.py
│   ├── inference.py              # Audio enhancement script
│   ├── setup_environment.py      # Environment setup helper
│   └── train.py                  # Training guide (points to notebook)
├── tools/                        # Testing and analysis tools
│   ├── __init__.py
│   ├── test_system.py            # System functionality tests
│   └── test_quality.py           # Audio quality assessment
├── apps/                         # User-facing applications
│   ├── __init__.py
│   ├── requirements.txt          
│   └── streamlit_app.py          # Web-based audio enhancement interface
├── notebooks/                    
│   └── main_training.ipynb       # Complete training pipeline
├── dataset/                      # Training and test data
├── models/                       # Saved model checkpoints   
├── results/                      # Training results and analysis
├── presentation/                 
├── setup.py                      # Post-clone setup script
├── requirements.txt              
├── config.yaml                   # Project configuration
├── run_app.sh                    
├── run_inference.sh              
└── run_tests.sh                  
```

## Quick Setup Guide

The path system now auto-detects the project root. No manual editing of `BASE_DIR` is required.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Pin Project Root via Environment Variable
Useful if running from outside the repo (e.g. notebooks, Docker, CI):
```bash
export UNET_AUDIOFILTER_ROOT="$(pwd)"   # or AI_CLEANVOICE_ROOT / PROJECT_ROOT
```

Add to your shell profile to persist.

### 3. Validate Paths
```bash
python -c "from config.paths import quick_setup; quick_setup()"
```

### 4. Run a System Test
```bash
python tools/test_system.py
```

### 5. Inference & App
```bash
./run_inference.sh input.wav output.wav
./run_app.sh
```

### 6. (Optional) Generate .env
```bash
python scripts/setup_environment.py --create-env
```

The `.env` will include `UNET_AUDIOFILTER_ROOT` for reproducible launches.

## Key Features:
- **Automatic Root Detection**: No path edits after cloning
- **Environment Override**: Set `UNET_AUDIOFILTER_ROOT` (or `AI_CLEANVOICE_ROOT` / `PROJECT_ROOT`) to relocate at runtime
- **Single Source of Truth**: `config/paths.py` builds all derived directories

## Model Architecture

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

### Performance

- **Validation Loss**: 0.000071
- **Average Improvement**: 56.3%
- **SNR Improvement**: +5.84 dB on test samples
- **Processing**: Real-time capable on GPU

## Training

The main training pipeline is located in `notebooks/main_training.ipynb`. This notebook contains:

- Complete GPU training setup
- Real-time progress monitoring
- Model evaluation and testing
- Audio visualization tools
- Interactive parameter tuning

To start training:
1. Open `notebooks/main_training.ipynb`
2. Run all cells in sequence
3. Model will be saved to `models/best_gpu_model.pth`

### Training Features:
- Chunked data loading for memory efficiency
- GPU optimization with CUDA
- Early stopping and checkpointing
- Robust error handling and recovery

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)
- 4GB+ GPU memory for training
- 16GB+ RAM for dataset loading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- VoiceBank+DEMAND dataset for training data
- U-Net architecture for speech enhancement
- PyTorch team for the deep learning framework
- Streamlit for the web interface framework