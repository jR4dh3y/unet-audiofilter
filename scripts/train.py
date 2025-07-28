#!/usr/bin/env python3
"""Training script - redirects to notebook with global path configuration"""

import sys
import os
from config.paths import PATHS, get_path

def main():
    """Training has been moved to the Jupyter notebook"""
    print("Speech Enhancement Training")
    print("=" * 50)
    print("Training functionality has been moved to the Jupyter notebook")
    print(f"   Please use: {get_path('notebooks') / 'main_training.ipynb'}")
    print("   This provides a complete training pipeline with:")
    print("   - GPU training setup")
    print("   - Real-time progress monitoring") 
    print("   - Model evaluation and testing")
    print("   - Audio visualization tools")
    print("   - Interactive parameter tuning")
    print("=" * 50)
    print("To start training:")
    print("1. Open VS Code or Jupyter")
    print(f"2. Open {get_path('notebooks') / 'main_training.ipynb'}")
    print("3. Run the cells step by step")
    print(f"\nModels will be saved to: {get_path('models.base')}")
    print(f"Results will be saved to: {get_path('results.base')}")
    
if __name__ == "__main__":
    main()
