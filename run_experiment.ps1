# run_experiment.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot


# Check if venv exists, create if not
if (-not (Test-Path -Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    Write-Host "Virtual environment created"
}

# Activate venv
$activateScript = ".\venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Failed to activate venv"
    exit 1
}
& $activateScript
Write-Host "Virtual environment activated"

# Upgrade pip
python -m pip install --upgrade pip
Write-Host "pip upgraded"

# Install requirements
pip install -r requirements.txt
Write-Host "Dependencies installed"

# Verify Python path
(Get-Command python).Source
Write-Host "Python path confirmed"

Write-Host "Pulling Ollama models..."
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
ollama pull qwen3:4b
ollama pull qwen3:8b
ollama pull qwen3:14b
ollama pull qwen3:32b
Write-Host "Ollama models ready"


# 1) Run main experiments
python main.py
Write-Host "main.py completed"

# 2) Run the analysis
python analysis.py
Write-Host "analysis.py completed"

# 3) Run the visualization
python visualize.py
Write-Host "visualize.py completed"
