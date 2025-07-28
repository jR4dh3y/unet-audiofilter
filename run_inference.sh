#!/bin/bash
# AI Speech Enhancement - Inference Script
# Usage: ./run_inference.sh input.wav output.wav

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
python scripts/inference.py "$@"
