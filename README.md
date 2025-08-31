### Overview

This repository evaluates simple naive forecasting approaches on a ticketing-machines time-series dataset and produces comparison charts.

- Methods: Seasonal Naive, Hour of Week (historical average), Holt–Winters/ETS
- Configuration: edit `config.toml` (data path, horizons, toggles)
- Outputs: metrics saved to `results/` as JSON and CSV; charts saved to `charts/`

### Installation

Prerequisites: Python 3.10+ is recommended (3.11+ preferred). Create and activate a virtual environment, then install dependencies.

macOS/Linux:

```bash
cd /path/to/ticketing_machines
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
cd C:\path\to\ticketing_machines
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Execution

- Run the main script (uses `config.toml` and writes to `results/`):

```bash
python main.py
```

- Generate dumbbell and panel charts comparing TPGNN vs naive baselines (reads from `results/` and writes PNGs into `charts/`):

```bash
python charts/dumbbell_graphs.py
```

- Example with explicit arguments and per-metric PNGs:

```bash
python charts/dumbbell_graphs.py \
  --tpgnn_csv results/tpgnn_baseline.csv \
  --naive_json results/naive_approaches_results.json \
  --out_dir charts \
  --horizons 3 6 9 12 \
  --per_metric_pngs
```

Notes:
- Adjust `config.toml` to change dataset path, horizons, or which methods run.
- Default chart inputs expect `results/naive_approaches_results.json` (from `main.py`) and `results/tpgnn_baseline.csv`.

