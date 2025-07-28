#!/bin/bash
# AI Speech Enhancement - Streamlit App
# Usage: ./run_app.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Add the project root to PYTHONPATH so imports work correctly
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

streamlit run apps/streamlit_app.py
