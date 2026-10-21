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

## Servers: reports and live monitor

Both are reachable from the laptop/phone over Tailscale (the mini is `100.83.107.60`, MagicDNS `mac-mini`). Manage them with `src/neuralssm/serve.sh` (run from `src/neuralssm/`; it is zsh):

```bash
./serve.sh start      # also the default; idempotent
./serve.sh stop       # kills the tmux server `srv` and any monitor it spawned
./serve.sh restart
./serve.sh status     # tmux sessions + whether :8000 / :8001 are listening
```

- `start` creates a tmux server `tmux -L srv` with two sessions: `reports` (static server for `src/neuralssm/reports/` on **:8000**, always on) and `watch` (`monitor.py --watch`, which starts the live monitor on **:8001** whenever a `main.py run|trials` process exists). Nothing restarts after a reboot — run `./serve.sh` again. No launchd agents are installed (the user declined them).
- If port 8000 is already taken by another process, `start` says so and skips the reports server; find and stop the old one before retrying. Don't kill processes you didn't start without asking.
- The monitor (`monitor.py`) shows each running `trials` job as a group of experiments (file order, readable label, hover for the full descriptor) with progress bars, mean trial time and ETA, the running processes, and the tmux panes. A finished job stays on the page for 15 min (`--grace-minutes`). Finished experiments have a button that runs `main.py errors` (only if trials lack saved errors) and `main.py report`, then links to the report; reports are written to `src/neuralssm/reports/` (scratch/logs in `reports/.work/`, gitignored). "Stop monitor" on the page closes it until a new job starts. It only watches `trials` jobs, not single `main.py run`.
- Standalone use: `python monitor.py --port 8001`; one-off report: `python main.py report <sample_gt> <start> <end> <descriptor files> [--out file.html]`.

### Launching experiments so the monitor can track them

Run `main.py trials True s <start> <end> <descriptor file>` from `src/neuralssm/` inside tmux (one tmux server per algorithm, `tmux -L abc|mcmc|snl|tsnl`; SNL/T-SNL via `run_nocrash.zsh`). Gotchas:
- The shell must have the `jaxenv` conda env active (`conda activate jaxenv`); a fresh tmux server may open in `(base)` without `jax`, and the system `python3` crashes on an architecture mismatch.
- Trial numbers start at **1** (`trial # must be a positive integer`).
- A descriptor file holds several experiments that run **sequentially, in file order**, each over the whole trial range. Existing trial directories are skipped, never overwritten.
- The monitor reads progress from each trial directory (`out.log`, `trial_time.pkl`), so don't rename or move result directories while a job runs.

## Architecture

### Experiment Lifecycle

```
create_exps.py → experiment_descriptor.py → experiment_runner.py → disk
                                                                      ↓
main.py (errors/ensemble/view) ← experiment_viewer.py ← results loaded from disk
```

1. **Descriptor** (`experiment_descriptor.py`): Text-format configs parsed via regex into typed descriptor objects. The descriptor encodes the full experiment configuration and deterministically generates the storage directory path. Example: `lgssm_d4_snl_scan_num_sims_50_1000`.

2. **Runner** (`experiment_runner.py`, class `ExperimentRunner`): Given a descriptor, samples/loads ground truth params, simulates observations, runs the inference algorithm, saves results + metrics + logs to `src/neuralssm/data/experiments/<simulator>/<config>/<trial>/`.

3. **Results** (`experiment_results.py`): Thin containers (`ABC_Results`, `MCMC_Results`, `SNL_Results`, `TSNL_Results`) that carry results arrays plus per-algorithm plotting metadata (name, marker, color).

4. **Viewer** (`experiment_viewer.py`, class `ExperimentViewer`): Loads saved results and generates posterior histograms, ESS plots, and distance plots into `<trial>/figures/`.

5. **Main** (`main.py`): The primary CLI (~65KB). 15+ subcommands orchestrate the full workflow including bootstrap confidence intervals, MMD analysis, model selection, and error aggregation across trials.

6. **Create Exps** (`create_exps.py`): Programmatic factory for generating descriptor combinations (e.g., scanning over hyperparameters). Hyperparameters live here in code, not in external config files.

### Inference Algorithms (`inference/`)
- `abc/`: SMC-ABC (Sequential Monte Carlo Approximate Bayesian Computation)
- `mcmc/`: BPF-MCMC (Bootstrap Particle Filter MCMC)
- `snl/`: SNL and T-SNL (Sequential Neural Likelihood, with/without temporal lag)

### Lag selection for T-SNL (`lag: auto` and `lag: ada`)

The `lag` field of a `tsnl` descriptor has three kinds of value. You choose the kind in the descriptor, when you create the experiment.

| `lag:` | Meaning | Status |
|---|---|---|
| integer | Fixed manual lag. This is the original behavior. The storage path does not change. | done |
| `auto` | The code selects the lag once from the round-0 simulations and keeps it fixed (Option A). The T-SNL network is a plain MAF at that lag. | done |
| `ada` | The code selects the lag again after every round (Option B). The network is a lag-dropout network of fixed shape. The lag is a mask. | done, D3 (A against B) deferred |

Both kinds select the lag with one lag-dropout network (decision D2). The code for one network per lag is removed. The descriptor of `auto` and `ada` has one more field, `lag_tol`. It is the TV tolerance η (default 0.1, so K* = 2η²). It is part of the storage path, for example `..._lag_auto_tol_0.1_subsample_...` or `..._lag_ada_tol_0.1_subsample_...`. Example, as `create_exps.py` writes it: `inf.create_desc(model_args, n_samples, n_rounds, 'ada', subsample, train_on, sampler, mcmc_steps, lag_tol=0.1)`.

**Decision log.** All decisions of this work are in `shared/results/I1_decision_log.md`. When the user decides something about I1, add a row to the log in the same turn. List each default that you choose without asking in the row E9.

**Code.** `inference/lag_selection.py` has the NumPy blocks. `inference/lag_scoring.py` has the JAX training and scoring. `SequentialNeuralLikelihood.learn_likelihood` calls both.

**The selection.**
1. Stage 0 runs a Gaussian proxy (PACF and innovation-covariance decrements) on the round-0 emissions. It gives L0 and the first grid 0..L_grid, with L_grid = min(max(2·L0, 2), L_cap). `L_cap` = T/4 is the largest lag that the method can examine.
2. Stage 1 trains one MAF of lag L_grid on 85 % of the round-0 trajectories. It scores the network on the other 15 % over the window t > L_cap. The input is `[theta, m_{L_grid..1}, masked y_{t-L_grid..t-1}, y_t]`. Here m is a lag indicator, and the observations older than the lag are zero. Each training row has a random lag. The training takes `LD_EPOCHS_FACTOR` (4) times the epochs of the descriptor. The network scores every lag 0..L_grid. L* = min(L_tol, L_cv). The fit uses the first run of at least 3 significant decrements. The one-SE rule is paired.
3. Grid growth (`lag_selection.grow_grid`): if the grid limits the result, L_grid doubles (up to L_cap) and the code trains the network again at the new shape. The flag `grid_capped` means that L_grid = L_cap and the grid still limits the result.
4. The flag `resolution_failure` is true if stage 0 sees at least 3 significant lags and stage 1 sees fewer. Then L* is a lower bound. The code writes a WARNING line in `out.log` and the flag in the JSON and in the figure. The flag does not change L*.
5. The diagnostic for damaged scores (`lag_selection.damage_diagnostic`, decision E13): a held-out trajectory is damaged if some lag scores more than 1 nat per observation worse than its own lag 0, or if a score is not finite. The code writes `damaged_fraction` in the JSON. It writes a WARNING in `out.log` if more than 5 % of the trajectories are damaged. It does not change the lag. The test cannot see damage that is the same at every lag. A trimmed mean as a remedy was tested and removed (decision E6).

**`auto`.** The runner passes a model factory `n_cond -> MAF`. The code builds the T-SNL network at L* and trains it on all round-0 trajectories. The code discards the candidate network.

**`ada`.**
- L_max = min(max(2·L*_0, 2), L_cap) is the lag of the network for the whole run.
- In every round the network trains with a random lag for each row (`LagDropoutMask`). It keeps its parameters from round to round.
- Before round r ≥ 1 trains, the network of round r−1 scores the new simulations of round r. They are held out by construction. `lag_scoring.reselect_lag` gives L*_r on the fixed grid 0..L_max.
- Rule against a failed round (decision E5): if the selection has `fit_status` `none`, or a score is not finite, the code keeps the lag of the previous round and writes a WARNING.
- The MCMC of round r and the final posterior use L*_r through `LagMaskedModel`. It shows the network at one lag as a T-SNL model of that lag.
- If the grid 0..L_max limits the result, the code flags the round (`capped`).
- Options of the learner in `lag_opts`, none of them in the descriptor: `L_max` (the cap), `ld_epochs_factor_warm` (the factor for the rounds after round 0), `hold` (the rule against a failed round, default true). The default values are in the log, rows E5, E7 and E8.

**Output.** Each trial writes `lag_selection.json`, `lag_selection.png` and `lag_selection_nll.npz` (the held-out scores of every selection, so that you can test a rule again without training). For `ada` the JSON has `L_max`, `L_star_r` (the lag of every round), `L_star_raw_r` (the value of the routine), `held_rounds`, `L_final`, `capped`, `capped_rounds`, `resolution_failure_rounds` and `damaged_rounds`. Every selection also has the keys `damaged_fraction`, `n_damaged`, `damaged_nats` and `damaged_warning`. The key `rounds` has the time and the number of XLA compilations of each phase of each round (also for `auto`). The saved `model` of an `ada` trial is the lag-dropout network.

**Access.** Code that needs the integer lag of a finished trial must call `experiment_descriptor.resolve_lag(inf, trial_dir)`. For an integer lag it returns `inf.lag`. Code that loads `model` must pass it through `experiment_descriptor.resolve_model(inf, trial_dir, model)`. It returns the model itself, except for `ada`.

**Cost.** The lag-dropout network is one training of 4 times the epochs. This is about 0.2 to 0.4 times the time of the old scheme with one network per lag. `ada` trains for 4 times the epochs in every round. In the first sanity run it took 1.6 times the time of `auto`.

**Tests.** Integer lags give the same results as before the I1 work (`tests/test_nde_integer_lag.py`). Run `python -m unittest tests.test_lag_selection tests.test_lgssm_exact tests.test_lag_scoring tests.test_descriptor_lag tests.test_nde_integer_lag`. The end-to-end runs are opt-in (`NEURALSSM_SLOW=1`, `tests.test_auto_lag_end_to_end`). The D2 scripts are `i1_d2_compare.py` and `i1_d2_summary.py`. The results of the removed per-lag scheme stay in `data/shared_results/D2`. The notes `I1_progress_*.md` in the shared channel give the evidence. The per-lag scheme and the trimmed mean are removed from the code. Their results stay in the notes and in `data/shared_results`.

### Neural Density Model (`maf/`)
Masked Autoregressive Flow used by SNL/T-SNL for density estimation. Key file: `density_models.py`.

### Simulators (`simulators/`)
Each simulator (`lgssm.py`, `lvssm.py`, `svssm.py`, `sirssm.py`) implements a consistent interface used by `misc.get_simulator()`.

### Parameters (`parameters.py`)
JAX-PyTree-registered parameter containers. `ParameterProperties` tracks trainability and TFP bijector constraints. Parameters live in both constrained (model) and unconstrained (optimizer) forms.

## Key Conventions

**Experiment storage**: Results are saved under `src/neuralssm/data/experiments/` (root is `misc.get_root()` = `src/neuralssm/data`); HTML reports from the `report` subcommand go to `src/neuralssm/reports/`. Directory paths encode the full config — changing any descriptor field changes the path.

**Descriptor uniqueness**: `AlreadyExistingExperiment` is raised if a result directory already exists. `NonExistentExperiment` is raised on missing reads. Both are defined in `misc.py`.

**RNG**: JAX PRNG keys are passed explicitly. The `--seed` argument accepts an integer (fixed), `'r'` (random), or `'s'` (shared across trials).

**Memory cleanup**: After long runs, `jax.clear_caches()` and `gc.collect()` are called explicitly. This is intentional — keep it.

**Batch-norm log-determinant (fixed 2026-10-20)**: `maf/layers.py::BatchNormLayer` used to sum the log-determinant over the conditioning columns too, which it does not transform. The trained network and the posterior over θ were unaffected, but the reported NLL (`loss_fn`, `loglik_fn`, `marg_loglik` and the saved `mll` files) carried an offset that depends on `dcond`, hence on the lag, and on the number of training steps. Saved `mll` files from before the fix keep that offset. Do not compare them with new runs or with `loglik_fn` recomputed from a loaded model (which now uses the fixed layer).

**Plotting style**: `scienceplots` + matplotlib. Per-algorithm convention: ABC=blue/circle, MCMC=red/cross, SNL=tomato/square, T-SNL=green/diamond.

**Relative imports**: Source files in `src/neuralssm/` use bare relative imports (e.g., `import experiment_descriptor as ed`). Run via `pip install -e .` or from within `src/neuralssm/`.

## Hardcoded Paths

`misc.get_root()` and some paths in `main.py` reference `/Users/kostastsampourakis/`. When generalizing for other environments, these need to be made configurable.

## Shared channel with the paper revision

`/Users/kostastsampourakis/Desktop/STCO review/shared/` is read/write for this session and the paper-revision session (see its `README.md`). At the start of a task, check `requests/` for asks from the revision (task IDs A1–A5 etc.). `results/` holds only `.md` files that explain the results, named `<taskID>_<name>.md`, giving the config, seed, commit hash, caveats and the path of the data. Keep the data itself (CSV, JSON, PNG) in this repo under `src/neuralssm/data/shared_results/` as `<taskID>_<name>.<ext>`, never in `results/`. Don't edit the revision's files or overwrite earlier results; version them (`_v2`).

## Simplified Technical English (ASD-STE100, basic rules)

Write technical documentation and instructions (procedures, READMEs, warnings, how-to text) in STE. Chat replies and code comments are not covered unless I ask. Spec detail is in the `ste` skill: load it only to check one rule, look up a word, or review a text in full.

- **Words**: use approved dictionary words, each with one approved meaning and part of speech. Technical nouns and technical verbs of the subject field are allowed. Do not use a technical noun as a verb, or a technical verb as a noun. Use one term for one item. American spelling. Examples: start (not begin), use (not utilize), make (not manufacture), keep (not maintain).
- **Noun strings**: maximum three words. Write a longer technical noun in full.
- **Verbs**: only infinitive, imperative, simple present, simple past, simple future, and past participle as an adjective. No perfect or progressive tenses, no auxiliary constructions. "-ing" only in technical nouns. Use the active voice (passive only in descriptions when the agent is unknown). Use a verb for an action, not a noun. No phrasal verbs.
- **Sentences**: short and clear. No contractions. Do not omit words or articles. Use "the", "a", "an", "this", "these" before nouns. Use a vertical list for long or complex items. Connect related sentences with connecting words.
- **Procedures**: maximum 20 words per sentence. One instruction per sentence (two only if the actions occur at the same time). Imperative form. Put a condition first, then a comma, then the command. Notes give information, not instructions.
- **Descriptions**: maximum 25 words per sentence. One topic per paragraph. Maximum six sentences per paragraph. Give information gradually, with key words and phrases for structure.
- **Safety**: WARNING for risk of injury or death, CAUTION for risk of damage (WARNING if both). Start with a clear command or condition, then explain the risk or result.
- **Punctuation**: no semicolons. Hyphens only to join directly related words. Parentheses only for references, step or item identifiers, abbreviations, singular/plural, explanations, alternatives. A number, a number with its unit, an abbreviation, an identifier, quoted text, a label, or a proper noun counts as one word. Text in parentheses counts as one word.
