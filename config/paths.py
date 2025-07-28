#!/usr/bin/env python3
"""
Global Path Configuration for AI Speech Enhancement System

This module provides centralized path management for the entire project.
After cloning the repository, users only need to update BASE_DIR to match
their local setup.

Usage:
    from config.paths import PATHS
    model_path = PATHS['models']['best_model']
    dataset_path = PATHS['dataset']['noisy_test']
"""

import os
from pathlib import Path

# ============================================================================
# MAIN CONFIGURATION - UPDATE THIS AFTER CLONING
# ============================================================================

# Base directory - UPDATE THIS PATH after cloning to your system
# Option 1: Manual path (update this line after cloning)
BASE_DIR = Path("/home/radhey/code/ai-clrvoice")

# Option 2: Auto-detect base directory (recommended for most users)
# Uncomment the line below and comment out the line above to use auto-detection
# BASE_DIR = Path(__file__).parent.parent.absolute()

# ============================================================================
# DERIVED PATHS - DO NOT MODIFY UNLESS CHANGING PROJECT STRUCTURE
# ============================================================================

PATHS = {
    # Core directories
    'base': BASE_DIR,
    'src': BASE_DIR / 'src',
    'config': BASE_DIR / 'config',
    
    # Script directories
    'scripts': BASE_DIR / 'scripts',
    'tools': BASE_DIR / 'tools', 
    'apps': BASE_DIR / 'apps',
    
    # Data directories
    'dataset': {
        'base': BASE_DIR / 'dataset',
        'clean_test': BASE_DIR / 'dataset' / 'clean_testset_wav',
        'clean_train': BASE_DIR / 'dataset' / 'clean_trainset_28spk_wav',
        'noisy_test': BASE_DIR / 'dataset' / 'noisy_testset_wav',
        'noisy_train': BASE_DIR / 'dataset' / 'noisy_trainset_28spk_wav',
        'test_scp': BASE_DIR / 'dataset' / 'test.scp',
        'train_scp': BASE_DIR / 'dataset' / 'train.scp',
    },
    
    # Model directories  
    'models': {
        'base': BASE_DIR / 'models',
        'best_model': BASE_DIR / 'models' / 'best_gpu_model.pth',
        'checkpoint_10': BASE_DIR / 'models' / 'checkpoint_epoch_10.pth',
        'checkpoint_5': BASE_DIR / 'models' / 'checkpoint_epoch_5.pth',
    },
    
    # Results directories
    'results': {
        'base': BASE_DIR / 'results',
        'comparison': BASE_DIR / 'results' / 'comparison',
        'enhanced_samples': BASE_DIR / 'results' / 'enhanced_samples',
        'gpu_training_history': BASE_DIR / 'results' / 'gpu_training_history.json',
        'gpu_model_summary': BASE_DIR / 'results' / 'gpu_model_summary.json',
    },
    
    # Documentation and presentation
    'notebooks': BASE_DIR / 'notebooks',
    'presentation': BASE_DIR / 'presentation',
    
    # Configuration files
    'config_yaml': BASE_DIR / 'config.yaml',
    'requirements': BASE_DIR / 'requirements.txt',
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_path(key_path: str) -> Path:
    """
    Get a path using dot notation
    
    Args:
        key_path: Path key using dot notation (e.g., 'models.best_model')
        
    Returns:
        Path object
        
    Example:
        >>> model_path = get_path('models.best_model')
        >>> dataset_path = get_path('dataset.noisy_test')
    """
    keys = key_path.split('.')
    current = PATHS
    
    for key in keys:
        current = current[key]
    
    return Path(current)

def ensure_dir_exists(path_key: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        path_key: Path key for directory
        
    Returns:
        Path object
    """
    path = get_path(path_key) if isinstance(path_key, str) else Path(path_key)
    path.mkdir(parents=True, exist_ok=True)
    return path

def validate_paths() -> dict:
    """
    Validate that all critical paths exist
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': [],
        'missing': [],
        'errors': []
    }
    
    # Critical paths that must exist
    critical_paths = [
        'base',
        'src', 
        'dataset.base',
        'models.base',
        'results.base'
    ]
    
    for path_key in critical_paths:
        try:
            path = get_path(path_key)
            if path.exists():
                results['valid'].append(f"{path_key}: {path}")
            else:
                results['missing'].append(f"{path_key}: {path}")
        except Exception as e:
            results['errors'].append(f"{path_key}: {e}")
    
    return results

def setup_project_structure():
    """
    Create necessary directories for the project
    """
    directories_to_create = [
        'scripts',
        'tools', 
        'apps',
        'results.comparison',
        'results.enhanced_samples',
        'config'
    ]
    
    for dir_key in directories_to_create:
        try:
            ensure_dir_exists(dir_key)
            print(f"✓ Created/verified: {get_path(dir_key)}")
        except Exception as e:
            print(f"✗ Failed to create {dir_key}: {e}")

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def add_to_python_path():
    """Add project directories to Python path"""
    import sys
    
    paths_to_add = [
        str(PATHS['base']),
        str(PATHS['src']),
        str(PATHS['config'])
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

# Auto-setup when imported
add_to_python_path()

# ============================================================================
# QUICK SETUP FOR NEW INSTALLATIONS
# ============================================================================

def quick_setup(new_base_dir: str = None):
    """
    Quick setup function for new installations
    
    Args:
        new_base_dir: New base directory path (optional)
    """
    if new_base_dir:
        global BASE_DIR, PATHS
        BASE_DIR = Path(new_base_dir)
        
        # Rebuild PATHS with new base directory
        # Note: In production, you'd want to update the file itself
        print(f"Base directory set to: {BASE_DIR}")
        print("Please update BASE_DIR in config/paths.py for permanent changes")
    
    # Validate current setup
    validation = validate_paths()
    
    print("=== Path Validation ===")
    for valid_path in validation['valid']:
        print(f"✓ {valid_path}")
    
    for missing_path in validation['missing']:
        print(f"✗ Missing: {missing_path}")
    
    for error in validation['errors']:
        print(f"⚠ Error: {error}")
    
    # Create missing directories
    setup_project_structure()
    
    print(f"\n=== Setup Complete ===")
    print(f"Project root: {BASE_DIR}")
    print("Ready to use!")

if __name__ == "__main__":
    # Run quick setup when executed directly
    quick_setup()
