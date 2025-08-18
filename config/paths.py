#!/usr/bin/env python3
"""Centralized project path handling (refactored).

Changes vs old version:
* Removed user-specific absolute BASE_DIR.
* Adds environment variable / auto-detection based root resolution.
* Provides `reinit_paths` to rebuild PATHS dynamically.
* Makes `quick_setup` actually use the provided path.

Public API retained: BASE_DIR, PATHS, get_path, ensure_dir_exists, validate_paths,
setup_project_structure, quick_setup.

Environment variable overrides (first match used):
    UNET_AUDIOFILTER_ROOT | AI_CLEANVOICE_ROOT | PROJECT_ROOT

Auto-detect markers (any present in a parent => root):
    .git, config.yaml, requirements.txt, README.md, src/

Usage:
    from config.paths import PATHS, get_path
    model_path = get_path('models.best_model')
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable, Dict, Any

__all__ = [
    'BASE_DIR', 'PATHS', 'get_path', 'ensure_dir_exists', 'validate_paths',
    'setup_project_structure', 'quick_setup', 'reinit_paths'
]


def _env_root_candidates() -> Iterable[str]:
    for var in ("UNET_AUDIOFILTER_ROOT", "AI_CLEANVOICE_ROOT", "PROJECT_ROOT"):
        val = os.getenv(var)
        if val:
            yield val


def _detect_base_dir() -> Path:
    # 1. Environment variable overrides
    for candidate in _env_root_candidates():
        p = Path(candidate).expanduser().resolve()
        if p.exists():
            return p

    # 2. Walk parents from this file upwards searching for marker files/dirs
    here = Path(__file__).resolve()
    markers = {"config.yaml", "requirements.txt", "README.md", ".git", "src"}
    for parent in [here.parent] + list(here.parents):
        if any((parent / m).exists() for m in markers):
            return parent

    # 3. Fallback to current working directory
    return Path.cwd()


def _build_paths(base: Path) -> Dict[str, Any]:
    return {
        'base': base,
        'src': base / 'src',
        'config': base / 'config',
        'scripts': base / 'scripts',
        'tools': base / 'tools',
        'apps': base / 'apps',
        'dataset': {
            'base': base / 'dataset',
            'clean_test': base / 'dataset' / 'clean_testset_wav',
            'clean_train': base / 'dataset' / 'clean_trainset_28spk_wav',
            'noisy_test': base / 'dataset' / 'noisy_testset_wav',
            'noisy_train': base / 'dataset' / 'noisy_trainset_28spk_wav',
            'test_scp': base / 'dataset' / 'test.scp',
            'train_scp': base / 'dataset' / 'train.scp',
        },
        'models': {
            'base': base / 'models',
            'best_model': base / 'models' / 'best_gpu_model.pth',
            'checkpoint_10': base / 'models' / 'checkpoint_epoch_10.pth',
            'checkpoint_5': base / 'models' / 'checkpoint_epoch_5.pth',
        },
        'results': {
            'base': base / 'results',
            'comparison': base / 'results' / 'comparison',
            'enhanced_samples': base / 'results' / 'enhanced_samples',
            'gpu_training_history': base / 'results' / 'gpu_training_history.json',
            'gpu_model_summary': base / 'results' / 'gpu_model_summary.json',
        },
        'notebooks': base / 'notebooks',
        'presentation': base / 'presentation',
        'config_yaml': base / 'config.yaml',
        'requirements': base / 'requirements.txt',
    }


# Initialize globals
BASE_DIR: Path = _detect_base_dir()
PATHS: Dict[str, Any] = _build_paths(BASE_DIR)

# ============================================================================
# MAIN CONFIGURATION - UPDATE THIS AFTER CLONING
# ============================================================================

def reinit_paths(new_base: Path) -> None:
    """Rebuild PATHS using a new base directory and refresh sys.path.

    Args:
        new_base: Path to new root.
    """
    global BASE_DIR, PATHS
    BASE_DIR = new_base.expanduser().resolve()
    PATHS = _build_paths(BASE_DIR)
    # Refresh python path ordering after changing base
    try:
        add_to_python_path(force=True)  # type: ignore[name-defined]
    except NameError:
        # add_to_python_path defined later; silent pass (will run at end of file)
        pass

# ============================================================================
# DERIVED PATHS - DO NOT MODIFY UNLESS CHANGING PROJECT STRUCTURE
# ============================================================================

    # (Moved python path refresh inside reinit_paths)

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

def add_to_python_path(force: bool = False):
    """Add project directories to Python path if not already present.

    Args:
        force: If True, ensure paths are moved to front even if present.
    """
    import sys

    paths_to_add = [str(PATHS['base']), str(PATHS['src']), str(PATHS['config'])]
    for p in paths_to_add:
        if p in sys.path:
            if force:
                sys.path.remove(p)
                sys.path.insert(0, p)
        else:
            sys.path.insert(0, p)

# Auto-setup when imported
add_to_python_path()

# ============================================================================
# QUICK SETUP FOR NEW INSTALLATIONS
# ============================================================================

def quick_setup(new_base_dir: str | Path | None = None):
    """Validate and (optionally) re-root the project paths.

    Args:
        new_base_dir: Optional new base directory. If provided, PATHS are rebuilt.
    """
    if new_base_dir:
        reinit_paths(Path(new_base_dir))
        print(f"→ Base directory overridden: {BASE_DIR}")
    else:
        print(f"→ Using detected base directory: {BASE_DIR}")

    validation = validate_paths()

    print("=== Path Validation ===")
    for valid_path in validation['valid']:
        print(f"✓ {valid_path}")
    for missing_path in validation['missing']:
        print(f"✗ Missing: {missing_path}")
    for error in validation['errors']:
        print(f"⚠ Error: {error}")

    # Create missing directories (non-destructive)
    setup_project_structure()

    print("\n=== Setup Complete ===")
    print(f"Project root: {BASE_DIR}")
    if new_base_dir:
        print("(Override is session-scoped; set an environment variable for persistence)")
    print("Ready to use!")

if __name__ == "__main__":
        quick_setup()
