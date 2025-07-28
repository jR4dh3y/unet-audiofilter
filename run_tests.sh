#!/bin/bash
# AI Speech Enhancement - Test Suite
# Usage: ./run_tests.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Running system tests..."
python tools/test_system.py

echo ""
echo "Running quality assessment..."
python tools/test_quality.py
