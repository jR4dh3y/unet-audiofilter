# AI Speech Enhancement - Simplified Structure

## 📁 Repository Structure

```
ai-clrvoice/
├── 📓 notebooks/
│   └── main_training.ipynb          # 🎯 MAIN WORKING FILE - Complete training pipeline
├── 🧠 src/                          # Core components
│   ├── __init__.py                  # Package initialization
│   ├── unet_model.py               # U-Net architecture & loss functions
│   ├── audio_utils.py              # Audio I/O utilities (Python 3.13+ compatible)
│   └── utils.py                    # Helper functions & utilities
├── 🚀 streamlit_app/
│   └── app.py                      # Streamlit web interface
├── 📊 dataset/                     # Training and test data
├── 🏆 models/                      # Saved model checkpoints
├── 📈 results/                     # Training results and plots
├── inference.py                    # Audio enhancement script
├── train.py                       # Training guide (points to notebook)
├── test_system.py                 # Basic system tests
└── test_quality.py               # Audio quality assessment
```

## 🎯 Main Working File

**`notebooks/main_training.ipynb`** is your primary development environment containing:
- ✅ Complete GPU training pipeline  
- ✅ Model architecture and training
- ✅ Real-time progress monitoring
- ✅ Audio quality evaluation
- ✅ Before/after comparisons
- ✅ Interactive testing tools
- ✅ Visualization and analysis

## 🧠 Core Components

### Essential Files (KEEP):
- **`src/unet_model.py`** - U-Net model architecture, required by notebook
- **`src/audio_utils.py`** - Audio loading/saving, Python 3.13+ compatible
- **`src/utils.py`** - Utility functions used throughout the project
- **`notebooks/main_training.ipynb`** - Complete working pipeline

### Simplified Scripts:
- **`inference.py`** - Standalone audio enhancement
- **`train.py`** - Points users to the notebook
- **`test_system.py`** - Basic functionality tests

## 🚀 Getting Started

1. **For Training**: Open `notebooks/main_training.ipynb`
2. **For Inference**: Run `python inference.py --input audio.wav --output enhanced.wav`  
3. **For Testing**: Run `python test_system.py`

## 📝 What Was Removed

Redundant files that duplicated notebook functionality:
- ❌ `src/training.py` - Training logic (moved to notebook)
- ❌ `src/dataset.py` - Dataset classes (defined in notebook)  
- ❌ `src/preprocessing.py` - Audio processing (integrated in notebook)
- ❌ `src/evaluation.py` - Metrics computation (available in notebook)
- ❌ `src/compat.py` - Compatibility layer (integrated)

## 🎮 Usage

### Training (Recommended)
```bash
# Open the main training notebook
jupyter notebook notebooks/main_training.ipynb
# or use VS Code with Jupyter extension
```

### Quick Testing
```bash
python test_system.py
```

### Audio Enhancement
```bash
python inference.py --input noisy_audio.wav --output enhanced_audio.wav
```

This simplified structure focuses on the **working notebook** while keeping essential **importable components** that the notebook depends on.
