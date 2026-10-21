r"""
I1, step 3: held-out decrements from trained networks against the exact
decrements, on the LGSSM of the T-SNL experiments (target variable ``d4``).

Model (``simulators/lgssm.py``, ``lgssm_params.py``): scalar state and
emission, :math:`a=0.9`, :math:`h=1`, :math:`R=0.1`, :math:`x_0\sim N(0,0.1)`,
target parameter :math:`Q` with the unconstrained parametrisation
:math:`z=\tfrac12\log Q` and prior :math:`z\sim N(0,1)`. The ground truth of
the experiments is :math:`Q=0.1`.

Two settings (``--mode``):

``fixed``
    All trajectories are simulated at :math:`Q=0.1`. The exact decrements
    :math:`D(L)` and :math:`H(L)` are those of one stationary ARMA(1,1)
    process (``lgssm_exact``). Isolates the estimation error of the networks.
``prior``
    Round 0 as in T-SNL: :math:`\theta^{(n)}` drawn from the prior, one
    trajectory each. The exact reference is the average of
    :math:`D(L;\theta^{(n)})` over the held-out draws.

Usage (from ``src/neuralssm``, ``jaxenv`` active)::

    python i1_step3_lgssm.py --mode fixed --n_sims 1000 --L_cap 20 --seed 0

``--L_cap`` is the largest lag and sets the scoring window ``t > L_cap``.
``--L_grid`` (default ``L_cap``) is the lag of the lag-dropout network, which scores every lag
``0..L_grid``. The step-3 results of 2026-10-20 in ``data/shared_results`` were made with one
network per lag (code removed on 2026-10-21, decision D2).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

import lgssm_exact as ex  # noqa: E402
import misc  # noqa: E402
import simulators.lgssm as lgssm  # noqa: E402
from inference import lag_scoring as sc  # noqa: E402
from inference import lag_selection as ls  # noqa: E402
from util.bijectors import PSDToRealBijector, RealToPSDBijector  # noqa: E402
from util.param import (get_unravel_fn, join_trees, params_from_tree,  # noqa: E402
                        sample_prior, to_train_array, tree_from_params)
from util.sample import sim_emissions  # noqa: E402
from util.train import get_sds  # noqa: E402

A, H, R, Q_TRUE = 0.9, 1.0, 0.1, 0.1


def q_from_z(z):
    """Constrained variance Q from the unconstrained parameter z."""
    return float(np.asarray(RealToPSDBijector()(jnp.asarray(z).reshape(1)))[0, 0])


def simulate_round0(key, setup, mode, n_sims, T):
    """Round-0 data in the production layout: (cond_params (N,1), emissions (N,T,1))."""
    ssm, props = setup['ssm'], setup['props']

    class _Learner:  # get_sds only needs these two attributes
        pass

    learner = _Learner()
    learner.props, learner.ssm = props, ssm

    if mode == 'prior':
        key, k1, k2 = jr.split(key, 3)
        params_sample = sample_prior(k1, props, n_sims)
        return get_sds(k2, None, learner, n_sims, params_sample, T)

    # fixed: ground-truth parameters of the experiments, as in ExperimentRunner
    key, k1, k2 = jr.split(key, 3)
    xparam = sample_prior(k1, props)[0]
    true_cps = lgssm.get_ground_truth(1, ['d4'])
    unravel = get_unravel_fn(xparam, props)(true_cps)
    new_tree = join_trees(unravel, tree_from_params(xparam), props)
    true_ps = params_from_tree(new_tree, setup['exp_info']['param_names'],
                               xparam._is_constrained_tree())
    true_ps.from_unconstrained(props)
    keys = jr.split(k2, n_sims)
    emissions = jax.vmap(lambda k: sim_emissions(k, true_ps, ssm, T))(keys)
    cond = jnp.tile(jnp.asarray(true_cps)[None], (n_sims, 1))
    return cond, emissions


def exact_reference(z_heldout, L_grid):
    """Average over held-out theta of the exact D(L) (L=0..L_grid-1) and H(L) (L=0..L_grid)."""
    Ds, Hs = [], []
    for z in z_heldout:
        e = ex.exact_decrements(A, H, q_from_z(z), R, L_grid)
        Ds.append(e['D'][:L_grid])
        Hs.append(e['H'][:L_grid + 1])
    Ds, Hs = np.array(Ds), np.array(Hs)
    # H(L) = 0.5 log(2 pi e V_L) is the entropy; the expected NLL equals it.
    return {'D': Ds.mean(axis=0), 'H': Hs.mean(axis=0), 'D_per_theta': Ds, 'H_per_theta': Hs}


def sanity_check_simulator(emissions, q, T_burn=30):
    """Compare sample autocovariances (t >= T_burn) with the exact stationary ones."""
    y = np.asarray(emissions)[:, T_burn:, 0]
    y = y - y.mean()
    emp = [np.mean(y[:, h:] * y[:, :y.shape[1] - h]) for h in range(4)]
    exa = ex.observation_autocov(A, H, q, R, 3)
    return np.array(emp), exa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fixed', 'prior'], default='fixed')
    ap.add_argument('--n_sims', type=int, default=1000)
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--L_cap', type=int, default=20)
    ap.add_argument('--L_grid', type=int, default=None, help='default: L_cap')
    ap.add_argument('--ld_factor', type=float, default=sc.LD_EPOCHS_FACTOR,
                    help='epochs of the lag-dropout network, as a multiple of --epochs')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--holdout', type=float, default=0.15)
    ap.add_argument('--train_windows', choices=['all', 'common'], default='all')
    ap.add_argument('--eta', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default=None)
    args = ap.parse_args()

    hp = dict(sc.PRODUCTION_HP)
    if args.epochs is not None:
        hp['epochs'] = args.epochs
    L_grid = args.L_cap if args.L_grid is None else args.L_grid
    tag = args.tag or f'{args.mode}_N{args.n_sims}_T{args.T}_Lcap{args.L_cap}_Lgrid{L_grid}_seed{args.seed}'
    out_dir = os.path.join(misc.get_root(), 'shared_results')
    os.makedirs(out_dir, exist_ok=True)

    setup = lgssm.setup(1, 1, 0, ['d4'])
    key = jr.PRNGKey(args.seed)
    key, kdata, kscore = jr.split(key, 3)

    t0 = time.perf_counter()
    cond, emissions = simulate_round0(kdata, setup, args.mode, args.n_sims, args.T)
    jax.block_until_ready(emissions)
    sec_sim = time.perf_counter() - t0
    print(f'simulated {emissions.shape} in {sec_sim:.1f}s ({args.mode})', flush=True)

    if args.mode == 'fixed':
        emp, exa = sanity_check_simulator(emissions, Q_TRUE)
        print('simulator autocov (t>=30) empirical', np.round(emp, 4), 'exact', np.round(exa, 4), flush=True)

    res = sc.score_dropout(kscore, cond, emissions, args.L_cap, L_grid, hp, args.holdout,
                           args.train_windows, epochs_factor=args.ld_factor,
                           logger=lambda s: print(s, end='', flush=True))
    ho = res['heldout_idx']
    z_ho = np.asarray(cond)[ho, 0]
    ref = exact_reference(z_ho, L_grid)

    K = ls.kl_tolerance(args.T, eta=args.eta)
    sel = ls.select_lag(res['nll'], args.T, K)
    dec = ls.decrements_from_nll(res['nll'])
    summary = {
        'tag': tag, 'args': vars(args), 'hp': hp, 'a': A, 'h': H, 'R': R,
        'seconds': {'simulation': sec_sim, 'train': float(res['seconds_train']),
                    'score': float(res['seconds_score']), 'total': res['seconds_total']},
        'jax': jax.__version__, 'device': str(jax.devices()[0]),
    }
    rows = []
    for j, L in enumerate(res['lags']):
        r = {'lag': int(L), 'H_hat': float(res['nll'][:, j].mean()),
             'SE_H': float(res['nll'][:, j].std(ddof=1) / np.sqrt(res['nll'].shape[0])),
             'H_exact': float(ref['H'][L]), 'train_loss': float(res['train_loss'][-1])}
        if dec is not None and L < L_grid:
            r.update(D_hat=float(dec['D'][L]), SE_D=float(dec['SE_D'][L]), D_exact=float(ref['D'][L]))
        rows.append(r)

    print('\n lag   H_hat    H_exact   dH      D_hat     SE_D     D_exact   (D_hat-D)/SE')
    for r in rows:
        dh = r['H_hat'] - r['H_exact']
        if 'D_hat' in r:
            z = (r['D_hat'] - r['D_exact']) / r['SE_D']
            print(f" {r['lag']:3d}  {r['H_hat']:.4f}  {r['H_exact']:.4f}  {dh:+.4f}  "
                  f"{r['D_hat']:+.5f}  {r['SE_D']:.5f}  {r['D_exact']:.5f}  {z:+.1f}")
        else:
            print(f" {r['lag']:3d}  {r['H_hat']:.4f}  {r['H_exact']:.4f}  {dh:+.4f}")
    if sel is not None:
        keys = ['L_star', 'L_tol', 'L_cv', 'C', 'rho', 'K_star', 'tol_capped', 'limited_by']
        summary['selection'] = {k: (float(sel[k]) if isinstance(sel[k], (float, np.floating)) else sel[k]) for k in keys}
        if args.mode == 'fixed':
            e = ex.exact_decrements(A, H, Q_TRUE, R, 40)
            summary['exact'] = {'rho_KF': e['rho_KF'], 'rho_KF_sq': e['rho_KF'] ** 2,
                                'L_tol_exact': ex.exact_tolerance_lag(A, H, Q_TRUE, R, args.T, K)}
        print('\nselection:', summary['selection'])
        if 'exact' in summary:
            print('exact    :', summary['exact'])

    summary['rows'] = rows
    path = os.path.join(out_dir, f'I1_step3_{tag}')
    with open(path + '.json', 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    np.savez(path + '.npz', nll=res['nll'], lags=res['lags'], heldout_idx=ho,
             z_heldout=z_ho, D_exact_per_theta=ref['D_per_theta'], H_exact_per_theta=ref['H_per_theta'])
    print(f'\nsaved {path}.json / .npz  (total {res["seconds_total"]:.0f}s)')


if __name__ == '__main__':
    main()
