r"""
I1, decision D2: lag-dropout network on the LGSSM of the T-SNL experiments (target
variable ``d4``, round 0 drawn from the prior).

Decision D2 (2026-10-21) chose the lag-dropout network. The comparison with one
network per lag used a ``perlag`` arm of this script. The code of that arm was
removed with the per-lag scheme, and its results stay in ``data/shared_results/D2``
(``__perlag`` files). ``i1_d2_summary.py`` still reads them.

One call runs ONE arm on ONE seed, so arms and seeds can run in parallel and a
sweep can resume. The data, the trajectory split and the keys depend on the
seed only, so the arms of one seed see the same held-out trajectories. The
file name is ``I1_D2_wide_<mode>_N<N>_T<T>_Lcap<L>_seed<s>__<arm>.npz`` with
``<arm>`` equal to ``ld<factor>`` (lag-dropout network trained for ``factor``
times the epochs of one candidate network).

The grid is fixed at ``0..L_cap`` (as in step 3), so every arm scores the same
lags and the decrements can be compared with the exact ones. The production
routine (stage 0 and grid growth) is tested in ``i1_grid_probe.py``.

Usage (from ``src/neuralssm``, ``jaxenv`` active)::

    python i1_d2_compare.py --seed 0 --arm ld5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.random as jr  # noqa: E402

import i1_step3_lgssm as s3  # noqa: E402
import misc  # noqa: E402
import simulators.lgssm as lgssm  # noqa: E402
from inference import lag_scoring as sc  # noqa: E402


def out_path(mode, n_sims, T, L_cap, seed, arm, tag=None):
    name = f'I1_D2_wide_{mode}_N{n_sims}_T{T}_Lcap{L_cap}_seed{seed}__{arm}'
    return os.path.join(misc.get_root(), 'shared_results', 'D2', (tag or '') + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fixed', 'prior'], default='prior')
    ap.add_argument('--n_sims', type=int, default=1000)
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--L_cap', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--holdout', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arm', type=str, required=True, help="'ld<factor>', for example ld4")
    ap.add_argument('--tag', type=str, default=None, help='prefix of the file name (tuning runs)')
    args = ap.parse_args()

    hp = dict(sc.PRODUCTION_HP)
    if args.epochs is not None:
        hp['epochs'] = args.epochs
    path = out_path(args.mode, args.n_sims, args.T, args.L_cap, args.seed, args.arm, args.tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path + '.npz'):
        print('exists, skipped:', path + '.npz')
        return

    setup = lgssm.setup(1, 1, 0, ['d4'])
    key = jr.PRNGKey(args.seed)
    key, kdata, kscore = jr.split(key, 3)
    cond, emissions = s3.simulate_round0(kdata, setup, args.mode, args.n_sims, args.T)
    N = emissions.shape[0]
    split = sc.split_trajectories(jr.fold_in(kscore, 1_000_000), N, args.holdout)
    log = lambda s: print(s, end='', flush=True)  # noqa: E731

    t0 = time.perf_counter()
    if not (args.arm.startswith('ld') and args.arm[2:]):
        raise SystemExit(f'unknown arm {args.arm}: use ld<factor>')
    res = sc.score_dropout(kscore, cond, emissions, args.L_cap, args.L_cap, hp, args.holdout, 'all',
                           epochs_factor=float(args.arm[2:]), logger=log, split=split)
    ho = res['heldout_idx']
    np.savez(path + '.npz', nll=res['nll'], heldout_idx=ho, z_heldout=np.asarray(cond)[ho, 0],
             train_loss=np.atleast_1d(res['train_loss']))
    with open(path + '.json', 'w') as f:
        json.dump({'args': vars(args), 'hp': hp,
                   'seconds_train': float(np.sum(res['seconds_train'])),
                   'seconds_score': float(np.sum(res['seconds_score'])),
                   'seconds_total': time.perf_counter() - t0,
                   'jax': jax.__version__, 'device': str(jax.devices()[0])}, f, indent=1)
    print(f"\n{args.arm} seed {args.seed}: train {np.sum(res['seconds_train']):.0f}s, saved {path}.npz")


if __name__ == '__main__':
    main()
