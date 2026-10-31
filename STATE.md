# STATE — 2026-10-31 (experiment session, MINI-STCO-EXPS)

The tracker in `/Users/kostastsampourakis/Desktop/STCO review/shared/` holds the full picture: `TODO.md` (section **Now**), `DECISIONS.md` (section **Open decisions**) and `REVIEW_MAP.md`. This note gives IDs only.

## Doing
Nothing runs. The A1 grid is complete (480 trials). The 4 tmux workers on `tmux -L tsnl` are idle.

## Done
- P-12 (a): T-SNL log density fixed in `src/neuralssm/util/train.py` and `src/neuralssm/inference/simulation_inference.py`. Test: `tests/test_tsnl_logdensity.py`. 110 lag tests pass. Old A1 trials: `src/neuralssm/data/experiments_tempered_p12/`.
- LGSSM descriptor fields `dyn_weight` and `emission_cov` (`experiment_descriptor.py`, `misc.py`, `simulators/lgssm.py`, `simulators/lgssm_params.py`). Default paths do not change.
- A1 v1 (P-10): descriptors `src/neuralssm/exps/lgssm/a1/`, scripts `src/neuralssm/scratch/a1/`, data `data/shared_results/A1_*`. Note: `STCO review/shared/results/A1_results_v1.md`. Bug note: `A1_progress_tempered-likelihood.md`.

## Next
1. F6: run the T-SNL experiments of Figs. 3–6 again, after the user confirms the scope (candidates in `TODO.md`, F6).
2. A5: the sparse observations. A1: the exact truncation error table, and F1 (BIC check).
3. H1 (push) is done by this wrap.

## Open
- F6 scope (user).
- Memory: 4 workers still caused 44 kills (exit code 137). Use 3 workers, or watch the memory, for F6.
- The inference key of `seed s` comes from the clock and is not recorded.
- I1-D1, I1-D4 to I1-D6, I1-E14 stay open (see `DECISIONS.md`).
