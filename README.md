# Climate-and-NLP: Climate-Focused Natural Language Processing

## ⚙️ Installation and Setup
### Prerequisites

  * Python (3.10 recommended)
  * Ensure "nq_mini_1000.csv" file is in the "data" directory.

### Step 1: Clone the Repository

```bash
git clone https://github.com/VeiledTee/Climate-and-NLP.git
cd Climate-and-NLP
```

### Step 2: Install Dependencies

All required Python packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```


## ▶️ Usage and Running Experiments

The core functionality of the project can be executed via the `main.py` script or dedicated experiment runners. Note that [Ollama](https://ollama.com/) must be installed for this to work.

### Running a Full Experiment

To start a complete experiment, you can use the provided shell scripts:

**On Linux/macOS:**

```bash
./run_experiment.sh
```

**On Windows (PowerShell):**

```powershell
.\run_experiment.ps1
```

These scripts are configured to call `main.py`, `analysis.py`, and `visualize.py` in sequence.

### Analyzing Results

Once an experiment is complete, use the analysis and visualization tools:

```bash
python analysis.py
python visualize.py
```

Outputs (metrics, plots, predictions) will be saved in the respective `results/` and `plots/` directories.
