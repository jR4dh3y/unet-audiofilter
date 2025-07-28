"""
Configuration package for AI Speech Enhancement System

This package contains all configuration files and path management utilities.
"""

from .paths import PATHS, get_path, ensure_dir_exists, validate_paths, setup_project_structure, quick_setup

__all__ = [
    'PATHS',
    'get_path', 
    'ensure_dir_exists',
    'validate_paths',
    'setup_project_structure',
    'quick_setup'
]
