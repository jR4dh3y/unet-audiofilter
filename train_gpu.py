#!/usr/bin/env python3
"""
Simple training script using the refactored GPU training module
"""

import sys
import os

# Add src to path
sys.path.append('/home/radhey/code/ai-clrvoice')

# Python 3.13+ compatibility
from src.compat import setup_aifc_compatibility
setup_aifc_compatibility()

from src.training import GPUTrainer

def main():
    """Run GPU training"""
    print("🚀 Starting GPU Training for Speech Enhancement")
    print("=" * 50)
    
    # Custom configuration (optional)
    config = {
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 10,
        'checkpoint_freq': 5,
        'model_dir': '/home/radhey/code/ai-clrvoice/models',
        'results_dir': '/home/radhey/code/ai-clrvoice/results'
    }
    
    # Initialize and run training
    trainer = GPUTrainer(config)
    
    # Create datasets
    print("\n📂 Creating datasets...")
    train_loader, val_loader, val_dataset = trainer.create_datasets()
    
    # Create model
    print("\n🧠 Initializing model...")
    model = trainer.create_model()
    
    # Train model
    print("\n🎯 Starting training...")
    history = trainer.train(model, train_loader, val_loader)
    
    # Save results
    print("\n💾 Saving results...")
    trainer.save_training_plots(history)
    
    print("\n🎉 Training completed successfully!")
    print(f"✓ Best validation loss: {history['best_val_loss']:.6f}")
    print(f"✓ Model saved to: models/best_gpu_model.pth")
    print(f"✓ Use inference_gpu.py for enhancement")

if __name__ == "__main__":
    main()
