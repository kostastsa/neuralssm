r"""
I1: run the round-0 lag selection (``select_lag_round0``, with grid growth) on
simulated scalar LGSSM data with chosen parameters, and compare with the exact
tolerance lag.

There is no parameter conditioning (the conditioning column is zero), so the
networks learn the density of one stationary process. This isolates the grid
growth from the effect of the prior over theta.

Usage (from ``src/neuralssm``, ``jaxenv`` active)::

    python i1_grid_probe.py --a 0.99 --R 100 --T 200 --n_sims 300 --seed 0
    python i1_grid_probe.py --a 0.99 --R 100 --T 200 --n_sims 300 --seed 0 --ld_factor 10
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

import lgssm_exact as ex  # noqa: E402
import misc  # noqa: E402
from inference import lag_scoring as sc  # noqa: E402
from inference import lag_selection as ls  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, default=0.99)
    ap.add_argument('--h', type=float, default=1.0)
    ap.add_argument('--Q', type=float, default=0.1)
    ap.add_argument('--R', type=float, default=1.0)
    ap.add_argument('--T', type=int, default=200)
    ap.add_argument('--n_sims', type=int, default=300)
    ap.add_argument('--eta', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ld_factor', type=float, default=None,
                    help='epochs of the lag-dropout network as a multiple of hp epochs')
    args = ap.parse_args()

    hp = dict(sc.PRODUCTION_HP)
    if args.epochs is not None:
        hp['epochs'] = args.epochs
    y = ex.simulate(args.seed, args.a, args.h, args.Q, args.R, args.T, args.n_sims).astype('float32')
    cond = jnp.zeros((args.n_sims, 1))
    K = ls.kl_tolerance(args.T, eta=args.eta)
    exact_lag = ex.exact_tolerance_lag(args.a, args.h, args.Q, args.R, args.T, K)
    e = ex.exact_decrements(args.a, args.h, args.Q, args.R, 5)
    print(f'exact tolerance lag {exact_lag}, rho_KF^2 = {e["rho_KF"] ** 2:.4f}, L_cap = {args.T // 4}', flush=True)

    t0 = time.perf_counter()
    sel = sc.select_lag_round0(jr.PRNGKey(args.seed), cond, jnp.asarray(y), args.T, hp, eta=args.eta,
                               ld_epochs_factor=args.ld_factor,
                               logger=lambda s: print(s, end='', flush=True))
    sel['exact_tolerance_lag'] = exact_lag
    sel['exact_rho_KF_sq'] = e['rho_KF'] ** 2
    sel['model'] = {'a': args.a, 'h': args.h, 'Q': args.Q, 'R': args.R}
    suffix = f'_dropout{sel["ld_epochs_factor"]:g}'
    out = os.path.join(misc.get_root(), 'shared_results',
                       f'I1_step5_probe_a{args.a}_R{args.R}_T{args.T}_N{args.n_sims}_seed{args.seed}{suffix}.json')
    with open(out, 'w') as f:
        json.dump(sel, f, indent=1)
    print(f"\nL*={sel['L_star']} (exact tolerance lag {exact_lag}), final grid 0..{sel['L_grid']}, "
          f"L_cap={sel['L_cap']}, grid_capped={sel['grid_capped']}, total {time.perf_counter() - t0:.0f}s\nsaved {out}")


if __name__ == '__main__':
    main()
