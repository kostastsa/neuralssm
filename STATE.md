# STATE — 2026-10-21 (session Mac-mini-neuralssm-tasks)

## Doing
Request I1 (lag selection for T-SNL, `lag: auto` and `lag: ada`). Lag dropout is now the only scheme. The flag, Option B, the hold rule and the damage diagnostic are in. The D3 validation is deferred by the user.

## Done
- **D2 closed:** lag dropout with 4× epochs (20-seed d4 test passed). All per-lag code removed: `inference/lag_scoring.py`, `inference/lag_selection.py` (`grow_grid`), the `i1_*.py` scripts, and `lag_variant` in `experiment_descriptor.py`. Paths are now `lag_auto_tol_<eta>` and `lag_ada_tol_<eta>`.
- **`resolution_failure` flag** (`lag_selection.resolution_failure`): stage 0 sees ≥ 3 significant lags and stage 1 fewer. Warning only.
- **Option B (`lag: ada`)**: `nde.py` (loop), `lag_scoring.py` (`LagDropoutMask`, `LagMaskedModel`, `reselect_lag`), `experiment_runner.py`, `experiment_descriptor.resolve_model`, `main.py`, `monitor.py`. Cap L_max = min(max(2·L*_0, 2), L_cap).
- **Rule E5 (hold):** a failed round keeps the previous lag. **Diagnostic E13:** damaged scores (1 nat, 5 %). Each trial saves `lag_selection_nll.npz`. Options in `lag_opts`: `L_max`, `ld_epochs_factor_warm`, `hold`.
- **Trimmed mean:** tested (no trim level passed the rule written before the run) and removed from the code.
- **Tests:** 108 lag tests pass with `NEURALSSM_SLOW=1`. Integer lags give the same results as before (`tests/test_nde_integer_lag.py`).
- **Notes** in `/Users/kostastsampourakis/Desktop/STCO review/shared/results/`: `I1_decision_log.md` (all decisions, keep it current), `I1_progress_d2.md`, `I1_progress_ada-flag.md`, `I1_progress_robust-score.md`, `I1_progress_damage-diagnostic.md`. Data: `src/neuralssm/data/shared_results/` (gitignored, so not in git).
- `CLAUDE.md` lag section rewritten. The `maf/layers.py` batch-norm fix (2026-10-20, earlier session) is committed with this work.

## Next
1. When the user decides to report A against B in the paper: D3 validation (A, B with and without hold, oracle lag; 3 test beds; 20 seeds). It also settles E7 (cap) and E8 (factor 1 against 4).
2. Plan decisions D1, D4, D5, D6 are not tested.
3. R=100 (A5 slow regime): all schemes give L*=0. The cumulative-gain test and a remedy for the overfit with more lags are not started.
4. Revision requests A1 to A5 (`.../STCO review/shared/requests/`): pick the order and use Option A or B once D3 is decided. Launch experiments via tmux and the monitor (see `CLAUDE.md`). Outputs go to `shared/results/` as `<taskID>_<name>.md`.

## Open
- Option B has one sanity run (d4, one seed): lags 2, 3, 1, 2, then the routine gave 0 and E5 held 2; time 1.6× Option A. The effect of E5 on the final posterior is not measured.
- E7 (cap) and E8 (training factor after round 0): open, defaults kept.
- E14 (why the MCMC gives a wide parameter set in some rounds): deferred by the user.
- Which revision request has priority?
- The per-lag results (`D2/`, step 3 and 4) cannot be reproduced from the repository.

Files changed (all committed at wrap): `CLAUDE.md`, `STATE.md`, `src/neuralssm/{experiment_descriptor,experiment_runner,main,monitor,lgssm_exact,i1_d2_compare,i1_d2_summary,i1_grid_probe,i1_step3_lgssm,i1_step3_plots}.py`, `src/neuralssm/inference/{nde,lag_scoring,lag_selection}.py`, `src/neuralssm/maf/layers.py`, `src/neuralssm/exps/test/`, `tests/test_{auto_lag_end_to_end,descriptor_lag,lag_scoring,lag_selection,lgssm_exact,nde_integer_lag}.py`; deleted `output.pdf`, `output 2.pdf`, `output.svg`. Nothing is half-finished.
