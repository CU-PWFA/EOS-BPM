# EOS BPM Simulation

A reusable Python implementation of the electro-optic sampling (EOS) BPM simulation in `EOS_BPM_Sim_3D_geom_2.ipynb`.

The current model simulates a two-bunch THz field interacting with a GaP electro-optic crystal and a four-piece rectangular crystal geometry. The top and bottom crystals use the rotated response and can be given an additional time delay; left and right use the unrotated response. Contributions from overlapping crystal regions are accumulated.

## What is included

- `src/eos_bpm_sim/core.py`: reusable physics, geometry, and plotting functions.
- `examples/basic_simulation.py`: minimal script showing the public API.
- `examples/EOS_BPM_simulation.ipynb`: clean notebook for interactive use.
- `tests/`: lightweight tests for importability, geometry, and a small simulation smoke test.
- `environment.yml`: Conda environment for VS Code/Jupyter.
- `requirements.txt`: pip-based installation alternative.
- `pyproject.toml`: package metadata and development dependencies.

## Installation in VS Code

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate eos-bpm
```

Then in VS Code:

1. Open the repository folder.
2. Open the Command Palette and choose **Python: Select Interpreter**.
3. Select the `eos-bpm` environment.
4. Open `examples/EOS_BPM_simulation.ipynb` and select the same kernel.

### Option B: Python virtual environment

Use Python 3.11 for the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

For a development installation:

```bash
python -m pip install -e ".[dev]"
```

## Basic usage

```python
from eos_bpm_sim import simulate_eos_2b_finite

result = simulate_eos_2b_finite(
    r01x=2e-3,
    r01y=0.0,
    r02x=0.0,
    r02y=0.0,
    time_delay=0.0,
    Q1=1.6e-9,
    Q2=0.0,
    pixelsx=512,
    pixelsy=512,
    plot=True,
)

intensity = result.intensity
gamma = result.gamma
time_map = result.time_map
```

For compatibility with the original notebook, the result can still be unpacked as:

```python
intensity, gamma, time_map = result
```

## Units

The package uses SI units for lengths and times unless documented otherwise. In particular, `Ri` and `Ro` follow the original notebook and are specified in **mm**. Positions such as `r01x`, `r01y`, `r02x`, and `r02y` are in **m**.

## Scientific validation

This repository is a software refactor of the supplied notebook; it is not a claim that the underlying physical model has been independently validated. The original equations are intentionally preserved as closely as practical. In particular, the code has not silently changed the FFT/radial-profile convention used by the notebook.

A few notebook-level items were deliberately excluded from the core package because they are exploratory or machine-specific: hard-coded `/Users/...` output paths, repeated scratch implementations, saved cell outputs, scan notebooks, and BPM fitting experiments.

## Notes on changes from the notebook

The refactor makes software-level changes needed for reuse:

- removed the invalid `from functools import bpartial` import;
- removed duplicate imports and duplicate function definitions;
- exposed `ax_angle` as an actual simulation parameter instead of hard-coding it to 10 degrees;
- removed unused camera-pixel/magnification arguments from the canonical simulation API;
- made plotting optional (`plot=False` by default);
- returned a structured `SimulationResult` while preserving legacy tuple unpacking;
- avoided mutating the frequency-response input during interpolation;
- added basic input validation.

## Running tests

```bash
pytest
```

The smoke test uses a very small grid so that it completes quickly. Full-resolution runs such as `2856 x 2856` are substantially more memory- and CPU-intensive.

## Citation / provenance

The equations and material parameters in this package originate from the supplied EOS BPM simulation notebook and its cited GaP model references. Add the appropriate project citation and license before publishing the repository publicly.
