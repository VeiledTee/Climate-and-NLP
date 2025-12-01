#!/usr/bin/env bash

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv || { echo "Failed to create venv"; exit 1; }
    echo "Virtual environment created"
fi

# Activate venv
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }
echo "Virtual environment activated"

# Upgrade pip first
python -m pip install --upgrade pip || { echo "pip upgrade failed"; exit 1; }

# Install requirements
pip install -r requirements.txt || { echo "Requirements installation failed"; exit 1; }
echo "Dependencies installed"

# Verify Python path
which python
echo "Python path confirmed"

echo "Pulling Ollama models..."
ollama pull qwen3:0.6b \
&& ollama pull qwen3:1.7b \
&& ollama pull qwen3:4b \
&& ollama pull qwen3:8b \
&& ollama pull qwen3:14b \
&& ollama pull qwen3:32b
echo "Ollama models ready"

# 1) Run main experiments
python main.py || { echo "main.py failed"; exit 1; }

# 2) Run the analysis
python analysis.py || { echo "analysis.py failed"; exit 1; }

# 3) Run the visualization
python visualize.py || { echo "visualize.py failed"; exit 1; }
