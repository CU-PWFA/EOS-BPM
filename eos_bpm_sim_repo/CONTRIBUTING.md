# Contributing

Create the development environment from the repository root, then run the test suite before committing changes.

```bash
conda env create -f environment.yml
conda activate eos-bpm
python -m pip install -e ".[dev]"
pytest
```

Keep the physics implementation in `src/eos_bpm_sim/`, and put exploratory or publication-specific work in `examples/` or `notebooks/` rather than adding it to the package API automatically.
