import pickle
import sys
import os

def load(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

trial_dir = sys.argv[1]

(_, true_cps), _ = load(os.path.join(trial_dir, 'gt'))
print(f'true_cps: {true_cps}')

for name in ['kde_error', 'min_error', 'bias', 'ang_bias', 'sdev']:
    path = os.path.join(trial_dir, name)
    try:
        val = load(path)
        print(f'{name}: {val}')
    except FileNotFoundError:
        print(f'{name}: not found')
