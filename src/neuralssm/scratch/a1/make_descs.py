"""A1 / A5 (decision P-10): write the descriptor files for main.py trials.

Arms: lag auto (Option A, lag_tol 0.1) and the fixed lags 1, 2, 5, 10, 20, 50.
Test beds (scalar LGSSM, target d4, Q_gt = 0.1, T = 100):
  tb1  a = 0.9,  R = 0.1   (the model of the paper, default fields)
  tb2  a = 0.99, R = 1     (A5, fast forgetting of the filter)
  tb3  a = 0.99, R = 100   (A5, slow forgetting: the negative example of R2.4)
One file for each arm, so that one tmux worker runs one arm over the three test beds.
Run from src/neuralssm: python scratch/a1/make_descs.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import experiment_descriptor as ed  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'exps', 'lgssm', 'a1')
T = 100
TEST_BEDS = {'tb1': (None, None), 'tb2': (0.99, 1.0), 'tb3': (0.99, 100.0)}
ARMS = ['auto', 1, 2, 5, 10, 20, 50]
# Oracle arm: the exact tolerance lag at Q_gt = 0.1 (lgssm_exact.exact_tolerance_lag, T = 100, K* = 0.02).
ORACLE = {'tb1': 4, 'tb2': 11, 'tb3': 58}

# T-SNL settings of create_exps.py (the paper's T-SNL configuration).
MODEL_ARGS = (5, 32, 5, 'relu', False, True, False, False, 20, 1e-4)
N_SAMPLES, N_ROUNDS, SUBSAMPLE, TRAIN_ON, SAMPLER, MCMC_STEPS = 500, 5, 1.0, 'best', 'rwm', 1000


def main():
    os.makedirs(OUT, exist_ok=True)
    sim = ed.LGSSM_Descriptor()
    inf = ed.InferenceDescriptor().get_descriptor('tsnl')
    for arm in ARMS:
        text = ''
        for a, r in TEST_BEDS.values():
            sim_desc = sim.create_desc(1, 1, T, 1.0, 'd4', dyn_weight=a, emission_cov=r)
            inf_desc = inf.create_desc(MODEL_ARGS, N_SAMPLES, N_ROUNDS, arm, SUBSAMPLE, TRAIN_ON, SAMPLER,
                                       MCMC_STEPS, lag_tol=0.1 if arm == 'auto' else None)
            text += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc) + '\n'
        path = os.path.join(OUT, f'a1_lag_{arm}.txt')
        with open(path, 'w') as f:
            f.write(text)
        descs = ed.parse(text)
        print(path, [d.get_dir() for d in descs][0], len(descs))
    text = ''
    for tb, (a, r) in TEST_BEDS.items():
        sim_desc = sim.create_desc(1, 1, T, 1.0, 'd4', dyn_weight=a, emission_cov=r)
        inf_desc = inf.create_desc(MODEL_ARGS, N_SAMPLES, N_ROUNDS, ORACLE[tb], SUBSAMPLE, TRAIN_ON, SAMPLER, MCMC_STEPS)
        text += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc) + '\n'
    path = os.path.join(OUT, 'a1_lag_oracle.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(path, len(ed.parse(text)))


if __name__ == '__main__':
    main()
