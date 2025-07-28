#!/usr/bin/env python3
"""
Environment Setup Script for AI Speech Enhancement System

This script helps users quickly configure the project after cloning.
It can auto-detect the project location or accept a custom path.

Usage:
    python scripts/setup_environment.py                    # Auto-detect
    python scripts/setup_environment.py /path/to/project   # Custom path
    python scripts/setup_environment.py --validate         # Just validate
"""

import sys
import os
import argparse
from pathlib import Path

def find_project_root():
    """Auto-detect project root directory"""
    current = Path(__file__).parent.parent.absolute()
    
    # Look for key project files to confirm this is the right directory
    key_files = ['config/paths.py', 'src/unet_model.py', 'README.md']
    
    if all((current / file).exists() for file in key_files):
        return current
    
    return None

def update_path_config(project_root: Path):
    """Update the BASE_DIR in config/paths.py"""
    
    config_file = project_root / 'config' / 'paths.py'
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    try:
        # Read current config
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Find and replace the BASE_DIR line
        lines = content.split('\n')
        updated_lines = []
        found_base_dir = False
        
        for line in lines:
            if line.strip().startswith('BASE_DIR = Path(') and not line.strip().startswith('#'):
                # Replace with new path
                updated_lines.append(f'BASE_DIR = Path("{project_root}")')
                found_base_dir = True
                print(f"✓ Updated BASE_DIR to: {project_root}")
            else:
                updated_lines.append(line)
        
        if not found_base_dir:
            print("⚠ Could not find BASE_DIR line to update")
            return False
        
        # Write updated config
        with open(config_file, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def create_env_file(project_root: Path):
    """Create a .env file for environment variables"""
    
    env_file = project_root / '.env'
    
    env_content = f"""# AI Speech Enhancement Environment Configuration
# This file is created automatically by setup_environment.py

# Project root directory
AI_SPEECH_ENHANCEMENT_ROOT={project_root}

# Dataset paths
DATASET_ROOT={project_root}/dataset
CLEAN_TEST_PATH={project_root}/dataset/clean_testset_wav
CLEAN_TRAIN_PATH={project_root}/dataset/clean_trainset_28spk_wav
NOISY_TEST_PATH={project_root}/dataset/noisy_testset_wav
NOISY_TRAIN_PATH={project_root}/dataset/noisy_trainset_28spk_wav

# Model paths
MODELS_DIR={project_root}/models
BEST_MODEL_PATH={project_root}/models/best_gpu_model.pth

# Results paths
RESULTS_DIR={project_root}/results
COMPARISON_DIR={project_root}/results/comparison
ENHANCED_SAMPLES_DIR={project_root}/results/enhanced_samples

# Python path additions
PYTHONPATH={project_root}:{project_root}/src:{project_root}/config
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✓ Created environment file: {env_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def validate_setup(project_root: Path):
    """Validate the project setup"""
    
    print(f"\n=== Validating Project Setup ===")
    print(f"Project root: {project_root}")
    
    # Check critical directories
    critical_dirs = [
        'src',
        'config', 
        'scripts',
        'tools',
        'dataset',
        'models',
        'results'
    ]
    
    all_good = True
    
    for dir_name in critical_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"❌ Missing: {dir_name}/")
            all_good = False
    
    # Check critical files
    critical_files = [
        'config/paths.py',
        'src/unet_model.py',
        'src/audio_utils.py',
        'src/utils.py',
        'README.md',
        'requirements.txt'
    ]
    
    for file_name in critical_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"❌ Missing: {file_name}")
            all_good = False
    
    # Test import
    try:
        sys.path.insert(0, str(project_root))
        from config.paths import PATHS, get_path
        print("✓ Global path configuration import successful")
        
        # Test a few key paths
        model_path = get_path('models.best_model')
        dataset_path = get_path('dataset.clean_test')
        print(f"✓ Path resolution working: {model_path.name}")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        all_good = False
    
    if all_good:
        print(f"\n🎉 Setup validation successful!")
        print(f"Ready to use AI Speech Enhancement system!")
    else:
        print(f"\n⚠ Some issues found. Please check the errors above.")
    
    return all_good

def main():
    parser = argparse.ArgumentParser(description='Setup AI Speech Enhancement environment')
    parser.add_argument('project_path', nargs='?', help='Path to project root directory')
    parser.add_argument('--validate', action='store_true', help='Only validate current setup')
    parser.add_argument('--create-env', action='store_true', help='Create .env file')
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_path:
        project_root = Path(args.project_path).absolute()
    else:
        project_root = find_project_root()
        if not project_root:
            print("❌ Could not auto-detect project root.")
            print("Please run from within the project directory or specify the path:")
            print("python scripts/setup_environment.py /path/to/project")
            return 1
    
    print(f"Using project root: {project_root}")
    
    if not project_root.exists():
        print(f"❌ Project directory does not exist: {project_root}")
        return 1
    
    # Validate only mode
    if args.validate:
        success = validate_setup(project_root)
        return 0 if success else 1
    
    # Update configuration
    print(f"\n=== Updating Configuration ===")
    success = update_path_config(project_root)
    
    if not success:
        print("❌ Failed to update configuration")
        return 1
    
    # Create .env file if requested
    if args.create_env:
        create_env_file(project_root)
    
    # Validate the setup
    validate_setup(project_root)
    
    print(f"\n=== Next Steps ===")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test the setup: python scripts/setup_environment.py --validate")
    print("3. Run inference: ./run_inference.sh input.wav output.wav")
    print("4. Start web app: ./run_app.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
