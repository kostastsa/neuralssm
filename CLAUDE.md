# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**neuralssm** is a research framework for likelihood-free inference on State Space Models (SSMs). It compares four inference algorithms — SMC-ABC, BPF-MCMC, SNL, and T-SNL — across multiple simulators (LGSSM, LVSSM, SVSSM, SIRSSM). Computations use JAX with Flax neural networks.

## Commands

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Lint
flake8 src/neuralssm tests

# Type check
mypy .

# Run experiment (main entry point)
python src/neuralssm/main.py run <descriptor>
python src/neuralssm/main.py trials <descriptor> --num-trials 10

# Analyze results
python src/neuralssm/main.py errors <descriptors...>
python src/neuralssm/main.py ensemble <descriptors...>
python src/neuralssm/main.py view <descriptor>
```

## Architecture

### Experiment Lifecycle

```
create_exps.py → experiment_descriptor.py → experiment_runner.py → disk
                                                                      ↓
main.py (errors/ensemble/view) ← experiment_viewer.py ← results loaded from disk
```

1. **Descriptor** (`experiment_descriptor.py`): Text-format configs parsed via regex into typed descriptor objects. The descriptor encodes the full experiment configuration and deterministically generates the storage directory path. Example: `lgssm_d4_snl_scan_num_sims_50_1000`.

2. **Runner** (`experiment_runner.py`, class `ExperimentRunner`): Given a descriptor, samples/loads ground truth params, simulates observations, runs the inference algorithm, saves results + metrics + logs to `~/experiments/<simulator>/<config>/<trial>/`.

3. **Results** (`experiment_results.py`): Thin containers (`ABC_Results`, `MCMC_Results`, `SNL_Results`, `TSNL_Results`) that carry results arrays plus per-algorithm plotting metadata (name, marker, color).

4. **Viewer** (`experiment_viewer.py`, class `ExperimentViewer`): Loads saved results and generates posterior histograms, ESS plots, and distance plots into `<trial>/figures/`.

5. **Main** (`main.py`): The primary CLI (~65KB). 15+ subcommands orchestrate the full workflow including bootstrap confidence intervals, MMD analysis, model selection, and error aggregation across trials.

6. **Create Exps** (`create_exps.py`): Programmatic factory for generating descriptor combinations (e.g., scanning over hyperparameters). Hyperparameters live here in code, not in external config files.

### Inference Algorithms (`inference/`)
- `abc/`: SMC-ABC (Sequential Monte Carlo Approximate Bayesian Computation)
- `mcmc/`: BPF-MCMC (Bootstrap Particle Filter MCMC)
- `snl/`: SNL and T-SNL (Sequential Neural Likelihood, with/without temporal lag)

### Neural Density Model (`maf/`)
Masked Autoregressive Flow used by SNL/T-SNL for density estimation. Key file: `density_models.py`.

### Simulators (`simulators/`)
Each simulator (`lgssm.py`, `lvssm.py`, `svssm.py`, `sirssm.py`) implements a consistent interface used by `misc.get_simulator()`.

### Parameters (`parameters.py`)
JAX-PyTree-registered parameter containers. `ParameterProperties` tracks trainability and TFP bijector constraints. Parameters live in both constrained (model) and unconstrained (optimizer) forms.

## Key Conventions

**Experiment storage**: Results are saved under `~/experiments/` (hardcoded root in `misc.get_root()`). Directory paths encode the full config — changing any descriptor field changes the path.

**Descriptor uniqueness**: `AlreadyExistingExperiment` is raised if a result directory already exists. `NonExistentExperiment` is raised on missing reads. Both are defined in `misc.py`.

**RNG**: JAX PRNG keys are passed explicitly. The `--seed` argument accepts an integer (fixed), `'r'` (random), or `'s'` (shared across trials).

**Memory cleanup**: After long runs, `jax.clear_caches()` and `gc.collect()` are called explicitly. This is intentional — keep it.

**Plotting style**: `scienceplots` + matplotlib. Per-algorithm convention: ABC=blue/circle, MCMC=red/cross, SNL=tomato/square, T-SNL=green/diamond.

**Relative imports**: Source files in `src/neuralssm/` use bare relative imports (e.g., `import experiment_descriptor as ed`). Run via `pip install -e .` or from within `src/neuralssm/`.

## Hardcoded Paths

`misc.get_root()` and some paths in `main.py` reference `/Users/kostastsampourakis/`. When generalizing for other environments, these need to be made configurable.
