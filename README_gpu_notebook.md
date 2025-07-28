# GPU Training Notebook Documentation

This document explains the structure and logic of the `gpu.ipynb` notebook for high-quality speech enhancement using a U-Net model trained on GPU with chunked data loading.

---

## 1. Environment and Imports
- **Libraries:** PyTorch, NumPy, Matplotlib, Librosa, SoundFile, tqdm, YAML, JSON, Pathlib, Random, Warnings.
- **Device Setup:** Automatically selects CUDA GPU if available, prints GPU info, and enables CUDA optimizations for speed and memory efficiency.
- **Source Path:** Adds the `src` directory to Python path to import the custom U-Net model.

---

## 2. ChunkedSpeechDataset Class
- **Purpose:** Efficiently loads large speech datasets in small, overlapping chunks to avoid memory issues.
- **Initialization:**
  - Takes directories for clean/noisy audio, file list, chunk duration, sample rate, FFT params, and overlap.
  - Validates file existence and precomputes chunk indices for all files.
- **__getitem__:**
  - Loads audio, extracts the correct chunk, pads if needed, and converts to normalized spectrograms.
  - Returns noisy and clean spectrogram tensors for training.
- **_audio_to_spec:**
  - Converts audio to magnitude spectrogram using STFT and normalizes to [0, 1].

---

## 3. Data Loading and Batching
- **Testing Data Loading:**
  - Loads a sample to verify shapes and value ranges.
- **DataLoader Setup:**
  - Uses PyTorch DataLoader for both training and validation sets.
  - Batch size is set based on GPU availability (8 for GPU, 4 for CPU).
  - Uses multiple workers and pin_memory for fast GPU data transfer.

---

## 4. Model Initialization
- **U-Net Model:**
  - 1 input/output channel, 32 base filters, depth 3, dropout 0.1 (memory efficient for GPU).
- **Loss and Optimizer:**
  - MSE loss, Adam optimizer, ReduceLROnPlateau scheduler for adaptive learning rate.
- **Training Config:**
  - 50 epochs, early stopping with patience, checkpoint directory setup.

---

## 5. Training and Validation Functions
- **train_epoch:**
  - Trains model for one epoch, computes loss, applies gradient clipping, and clears GPU cache periodically.
- **validate_epoch:**
  - Evaluates model on validation set, computes loss, and clears GPU cache.

---

## 6. Training Loop
- **Main Loop:**
  - Runs for up to 50 epochs, tracks losses and learning rates, saves best model and periodic checkpoints.
  - Implements early stopping and prints progress.
- **Post-Training:**
  - Prints total training time, best validation loss, and final learning rate.

---

## 7. Visualization and Analysis
- **Plots:**
  - Training/validation loss, learning rate schedule, overfitting monitor, and improvement curves.
  - Saves plots and training history to results folder.
- **Testing:**
  - Loads best model, runs on validation samples, computes MSE and improvement, prints summary statistics.

---

## 8. Robust Training (Optional)
- **train_with_checkpoints:**
  - Advanced function for robust training with frequent checkpointing and recovery from interruptions or errors.

---

## 9. Model Loading and Inference
- **Checkpoint Loading:**
  - Loads best model checkpoint, restores training history if available.
  - Prints model parameters and validation loss.

---

## 10. Inference Script Generation
- **inference_gpu.py:**
  - Generates a standalone script for high-quality inference using the trained model.
  - Supports chunked processing, crossfading, and command-line interface.
  - Script is made executable and tested on a sample file.

---

## 11. Quality Assessment
- **SNR and MSE:**
  - Compares noisy, clean, and enhanced audio for SNR and MSE improvement.
  - Prints and saves results for easy evaluation.

---

## 12. Summary
- **End-to-End Pipeline:**
  - The notebook provides a complete, memory-efficient, and GPU-optimized pipeline for speech enhancement.
  - Includes robust training, checkpointing, visualization, and production-ready inference.

---

## Usage
- **Training:** Run the notebook cells in order to train and validate the model.
- **Inference:** Use the generated `inference_gpu.py` script:
  ```bash
  python inference_gpu.py input_noisy.wav output_enhanced.wav
  ```
- **Results:**
  - Training curves, model checkpoints, and enhanced audio are saved in the `results/` and `models/` folders.

---

## Notes
- Adjust batch size and chunk duration for your GPU memory.
- The notebook is robust to interruptions and can resume from checkpoints.
- For best results, use a CUDA-capable GPU.

---

## Appendix: U-Net Model Architecture and AI Concepts

### U-Net Model Overview
U-Net is a convolutional neural network (CNN) architecture originally designed for biomedical image segmentation, but it is highly effective for audio spectrogram enhancement due to its ability to capture both local and global features. In this notebook, U-Net is adapted for single-channel (mono) speech spectrogram enhancement.

#### Key Features:
- **Encoder-Decoder Structure:** The model consists of a contracting path (encoder) and an expansive path (decoder).
- **Skip Connections:** Feature maps from the encoder are concatenated with corresponding decoder layers, preserving fine-grained details and enabling better reconstruction.
- **Fully Convolutional:** No dense layers; the model can process variable-length inputs.
- **Dropout and BatchNorm:** Used for regularization and training stability.

### Layer-by-Layer Breakdown

#### 1. Encoder (Downsampling Path)
- **Convolutional Blocks:** Each block consists of two convolutional layers (Conv2d) with ReLU activation and optional batch normalization.
- **Downsampling:** MaxPooling2d layers reduce the spatial dimension (frequency/time) after each block, allowing the network to learn hierarchical features.
- **Base Filters:** The number of filters starts at 32 and doubles at each depth level, capturing increasingly abstract features.

#### 2. Bottleneck
- The deepest part of the network, with the highest number of filters, captures the most abstract representation of the input.

#### 3. Decoder (Upsampling Path)
- **Transposed Convolutions (ConvTranspose2d):** Used to upsample the feature maps, increasing their spatial resolution.
- **Concatenation (Skip Connections):** The upsampled features are concatenated with the corresponding encoder features, allowing the model to recover fine details lost during downsampling.
- **Convolutional Blocks:** Similar to the encoder, each block refines the upsampled features.

#### 4. Output Layer
- **Final Convolution:** A 1x1 convolution reduces the number of channels to 1 (for mono spectrogram output).
- **Activation:** Typically linear for regression tasks like spectrogram enhancement.

### AI Concepts Used
- **Convolutional Neural Networks (CNNs):** Extract local patterns in spectrograms, such as harmonics and noise.
- **Skip Connections:** Help prevent vanishing gradients and allow the network to use both low-level and high-level features for reconstruction.
- **Batch Normalization:** Stabilizes and accelerates training by normalizing activations.
- **Dropout:** Reduces overfitting by randomly dropping units during training.
- **Mean Squared Error (MSE) Loss:** Measures the difference between predicted and target spectrograms.
- **Adam Optimizer:** Adaptive learning rate optimization for efficient training.
- **ReduceLROnPlateau Scheduler:** Dynamically reduces learning rate when validation loss plateaus.
- **Early Stopping:** Stops training when validation loss does not improve, preventing overfitting.

### Why U-Net for Speech Enhancement?
- **Preserves Temporal and Spectral Details:** Skip connections allow the model to reconstruct clean speech with minimal loss of detail.
- **Handles Variable Input Lengths:** Fully convolutional design is robust to different audio durations.
- **Efficient and Effective:** Achieves high-quality enhancement with relatively few parameters (1.9M in this setup).

### Model Hyperparameters in This Notebook
- **Input/Output Channels:** 1 (magnitude spectrogram)
- **Base Filters:** 32
- **Depth:** 3 (3 downsampling and 3 upsampling blocks)
- **Dropout:** 0.1
- **BatchNorm:** Used in each block for stability

---

For further details, see the `src/model.py` file for the exact U-Net implementation used in this project.
