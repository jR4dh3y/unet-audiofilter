#!/usr/bin/env python3
"""
Setup Script for AI Speech Enhancement System

This is a simplified setup script that redirects to the comprehensive
setup environment script in the scripts/ directory.

Quick Setup:
    python setup.py                    # Auto-detect and setup
    python setup.py --validate         # Validate current setup
    python setup.py /path/to/project   # Setup with custom path

For advanced options, use:
    python scripts/setup_environment.py --help
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Main setup function - delegates to comprehensive setup script"""
    
    print("AI Speech Enhancement System - Setup")
    print("=" * 50)
    
    # Get current directory
    current_dir = Path(__file__).parent.absolute()
    setup_script = current_dir / 'scripts' / 'setup_environment.py'
    
    if not setup_script.exists():
        print(f"❌ Setup script not found: {setup_script}")
        print("Please ensure you're running this from the project root directory.")
        return 1
    
    # Pass all arguments to the comprehensive setup script
    args = [sys.executable, str(setup_script)] + sys.argv[1:]
    
    try:
        # Run the comprehensive setup script
        result = subprocess.run(args, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running setup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
