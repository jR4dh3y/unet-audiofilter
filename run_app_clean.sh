#!/bin/bash

# AI Speech Enhancement - Clean Streamlit App Launcher
echo "🎙️ Starting AI Speech Enhancement Web App..."

# Check if in correct directory
if [ ! -f "inference_gpu.py" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check if model exists
if [ ! -f "models/best_gpu_model.pth" ]; then
    echo "❌ Trained model not found at models/best_gpu_model.pth"
    echo "💡 Please train the model first using notebooks/gpu.ipynb"
    exit 1
fi

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -r streamlit_app/requirements.txt
else
    source venv/bin/activate
fi

# Start Streamlit app
echo "🚀 Launching Streamlit app..."
cd streamlit_app
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

echo "✅ App started! Open http://localhost:8501 in your browser"
