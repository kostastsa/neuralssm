import pickle
import sys
import os

def load(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

def print_trial(trial_dir):
    print(f'  Directory : {trial_dir}')

    try:
        (_, true_cps), _ = load(os.path.join(trial_dir, 'gt'))
        print(f'  true_cps  : {true_cps}')
    except FileNotFoundError:
        print(f'  true_cps  : not found')

    try:
        lp = load(os.path.join(trial_dir, 'lp'))
        print(f'  log prior : {lp:.4f}  (-log p = {-lp:.4f})')
    except FileNotFoundError:
        print(f'  log prior : not found')

    time_txt = os.path.join(trial_dir, 'time.txt')
    time_pkl = os.path.join(trial_dir, 'time_all_rounds.pkl')
    if os.path.exists(time_txt):
        total = float(open(time_txt).read().strip())
        print(f'  time      : {total:.2f}s (total)')
    elif os.path.exists(time_pkl):
        rounds = load(os.path.join(trial_dir, 'time_all_rounds'))
        total = sum(rounds)
        per_round = ', '.join(f'{t:.2f}s' for t in rounds)
        print(f'  time      : {total:.2f}s total ({per_round})')
    else:
        print(f'  time      : not found')

    try:
        num_sims = load(os.path.join(trial_dir, 'num_sims'))
        print(f'  num_sims  : {num_sims}')
    except FileNotFoundError:
        print(f'  num_sims  : not found')

    print(f'  {"Metric":<12} {"Value"}')
    print(f'  {"-"*30}')
    for name in ['kde_error', 'min_error', 'bias', 'ang_bias', 'sdev']:
        try:
            val = load(os.path.join(trial_dir, name))
            print(f'  {name:<12} {val}')
        except FileNotFoundError:
            print(f'  {name:<12} not found')

dirs = sys.argv[1:]

for i, trial_dir in enumerate(dirs):
    print(f'[{i+1}/{len(dirs)}] {"="*50}')
    print_trial(trial_dir)
    print()
