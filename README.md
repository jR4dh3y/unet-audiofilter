# AI Speech Enhancement - Clean Voice

A GPU-accelerated deep learning system for speech enhancement using U-Net architecture.
## Project Overview

This project implements a complete pipeline for removing background noise from speech recordings using a U-Net model trained on the VoiceBank+DEMAND dataset.

## Project Structure

```
ai-clrvoice/
├── config/                       # Global configuration
│   ├── __init__.py
│   └── paths.py                  # Global path configuration - UPDATE AFTER CLONING
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

After cloning this repository, follow these steps to configure the project for your system:

### Option 1: Automatic Setup (Recommended)

```bash
# Run the setup script
python setup.py

# Or with validation
python setup.py --validate
```

### Option 2: Manual Configuration

1. **Update Base Path**
   - Edit `config/paths.py`
   - Change `BASE_DIR = Path("/home/radhey/code/ai-clrvoice")` to your path
   - Or uncomment the auto-detection line

2. **Alternative: Use Auto-Detection**
   - In `config/paths.py`, comment out the manual path
   - Uncomment: `BASE_DIR = Path(__file__).parent.parent.absolute()`

### Option 3: Environment Variables

Create a `.env` file in the project root:

```bash
# Generate .env file
python scripts/setup_environment.py --create-env
```

### Verification

Test your setup:

```bash
# Validate configuration
python scripts/setup_environment.py --validate

# Quick system test
python tools/test_system.py
```

### Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Run inference: `./run_inference.sh input.wav output.wav`
3. Start web app: `./run_app.sh`
4. Run tests: `./run_tests.sh`

## Key Features:
- **Single Configuration File**: `config/paths.py` contains all path definitions
- **Easy Setup**: Run `python setup.py` after cloning to configure paths
- **Auto-Detection**: Automatically detects project structure

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