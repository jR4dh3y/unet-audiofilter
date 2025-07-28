"""
Python 3.13+ Compatibility Module
Handles deprecated modules that were removed in Python 3.13
"""

import sys
import warnings

def setup_aifc_compatibility():
    """Setup aifc compatibility for Python 3.13+"""
    try:
        import aifc
        return True
    except ImportError:
        # Create a basic aifc module replacement
        import types
        
        # Create a minimal aifc module that provides basic functionality
        aifc_module = types.ModuleType('aifc')
        
        # Add essential attributes that might be accessed
        aifc_module.__file__ = '<compatibility>'
        aifc_module.__doc__ = 'AIFC compatibility module for Python 3.13+'
        
        # Define common constants and exceptions that might be used
        class Error(Exception):
            pass
        
        aifc_module.Error = Error
        
        # Basic open function that raises a proper error
        def open(file, mode='rb'):
            raise Error("AIFC format not supported in Python 3.13+. Use WAV format instead.")
        
        aifc_module.open = open
        
        # Install the compatibility module
        sys.modules['aifc'] = aifc_module
        
        # Show warning only once
        if not hasattr(setup_aifc_compatibility, '_warned'):
            warnings.warn(
                "aifc module not available in Python 3.13+. "
                "AIFF-C audio files are not supported. "
                "Please use WAV format for audio files.",
                UserWarning,
                stacklevel=2
            )
            setup_aifc_compatibility._warned = True
        
        return False

# Setup compatibility on import
setup_aifc_compatibility()
