# scratch

This folder holds the scripts for side questions of the revision tasks. Each subfolder is one task. The experiments themselves run through the runner and `main.py trials`. The scripts here only make descriptor files or analyze the trials.

## a1 (tasks A1 and A5, decision P-10)

| File | Content |
|---|---|
| `a1/make_descs.py` | Writes the descriptor files `exps/lgssm/a1/a1_lag_<arm>.txt`. Arms: `lag: auto`, the fixed lags 1, 2, 5, 10, 20 and 50, and the oracle lag. Test beds: tb1 ($a=0.9$, $R=0.1$), tb2 ($a=0.99$, $R=1$), tb3 ($a=0.99$, $R=100$). |
| `a1/analyze.py` | Compares each T-SNL posterior with the exact Kalman posterior. Writes `data/shared_results/A1_*.csv` and `A1_posterior_vs_lag.png`. |
| `a1/logs/` | The output of the tmux workers. |

Launch (one tmux worker for each arm, from `src/neuralssm`):

```
bash run_nocrash.zsh False s 1 20 exps/lgssm/a1/a1_lag_<arm>.txt
```

Analysis: `python scratch/a1/analyze.py --trials 1 20`.
