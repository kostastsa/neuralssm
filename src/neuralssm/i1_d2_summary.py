r"""
I1, decision D2: summary of the runs of ``i1_d2_compare.py``.

For each seed and arm: the held-out curve, :math:`\hat D(L)` with its paired SE,
and :math:`L^\star` from ``select_lag`` on the fixed grid ``0..L_cap``. The
exact reference is the average over the held-out draws of :math:`\theta` of the
exact :math:`D(L;\theta)` and :math:`H(L;\theta)` (``lgssm_exact``).

The ``perlag`` arm was made with code that decision D2 removed. Its files stay in
``data/shared_results/D2`` and this script reads them.

Writes (``--out`` prefix, in ``data/shared_results/``)

* ``<out>_seeds.csv``: one row per (seed, arm).
* ``<out>_lags.csv``: one row per (arm, lag), mean over seeds.
* ``<out>.png``: the figure.

Usage (from ``src/neuralssm``, ``jaxenv`` active)::

    python i1_d2_summary.py --seeds 0-19 --arms perlag,ld1,ld5 --out I1_D2_wide_summary
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lgssm_exact as ex  # noqa: E402
import misc  # noqa: E402
from i1_d2_compare import out_path  # noqa: E402
from i1_step3_lgssm import A, H, R, exact_reference, q_from_z  # noqa: E402
from inference import lag_selection as ls  # noqa: E402

COMPARE_LAGS = (0, 2, 5, 10)


def parse_seeds(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load(args, seed, arm):
    base = out_path(args.mode, args.n_sims, args.T, args.L_cap, seed, arm, args.tag)
    if not os.path.exists(base + '.npz'):
        return None
    import json
    z = np.load(base + '.npz')
    with open(base + '.json') as f:
        meta = json.load(f)
    return {'nll': z['nll'], 'ho': z['heldout_idx'], 'z': z['z_heldout'], 'meta': meta,
            'train_loss': z['train_loss']}


def exact_tolerance_from_mean(Dbar, T, K):
    tail = np.cumsum(Dbar[::-1])[::-1]
    ok = np.flatnonzero(T * tail <= K)
    return int(ok[0]) if len(ok) else len(Dbar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='prior')
    ap.add_argument('--n_sims', type=int, default=1000)
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--L_cap', type=int, default=20)
    ap.add_argument('--seeds', type=str, required=True)
    ap.add_argument('--arms', type=str, required=True)
    ap.add_argument('--ref', type=str, default='perlag', help='arm the others are compared with')
    ap.add_argument('--tag', type=str, default=None)
    ap.add_argument('--out', type=str, default='I1_D2_wide_summary')
    args = ap.parse_args()
    seeds, arms = parse_seeds(args.seeds), args.arms.split(',')
    K = ls.kl_tolerance(args.T)
    Lg = args.L_cap

    rows, per_lag, paired = [], {a: [] for a in arms}, {a: [] for a in arms}
    for seed in seeds:
        runs = {a: load(args, seed, a) for a in arms}
        if any(r is None for r in runs.values()):
            print(f'seed {seed}: missing arms {[a for a, r in runs.items() if r is None]}, skipped')
            continue
        ho0 = runs[arms[0]]['ho']
        assert all(np.array_equal(r['ho'], ho0) for r in runs.values()), 'split differs between arms'
        ref = exact_reference(runs[arms[0]]['z'], Lg)
        wide = exact_reference(runs[arms[0]]['z'], 80)
        L_exact = exact_tolerance_from_mean(wide['D'], args.T, K)
        for a, r in runs.items():
            nll = r['nll']
            dec = ls.decrements_from_nll(nll)
            sel = ls.select_lag(nll, args.T, K)
            rows.append({'seed': seed, 'arm': a, 'L_star': sel['L_star'], 'L_tol': sel['L_tol'],
                         'L_cv': sel['L_cv'], 'fit_status': sel['fit_status'], 'rho_hat': sel['rho'],
                         'L_exact_tol_mean_theta': L_exact,
                         'seconds_train': r['meta']['seconds_train'],
                         'H0_minus_exact': dec['H'][0] - ref['H'][0],
                         'rmse_D_vs_exact': float(np.sqrt(np.mean((dec['D'] - ref['D']) ** 2))),
                         'H_bias_at_Lcap': dec['H'][Lg] - ref['H'][Lg]})
            per_lag[a].append(pd.DataFrame({'seed': seed, 'lag': np.arange(Lg + 1),
                                            'H_hat': dec['H'], 'H_exact': ref['H'],
                                            'D_hat': np.append(dec['D'], np.nan),
                                            'SE_D': np.append(dec['SE_D'], np.nan),
                                            'D_exact': np.append(ref['D'], np.nan)}))
            # per-trajectory decrements for the paired comparison with the reference arm
            paired[a].append(nll[:, :-1] - nll[:, 1:])

    df = pd.DataFrame(rows)
    out_dir = os.path.join(misc.get_root(), 'shared_results')
    df.to_csv(os.path.join(out_dir, args.out + '_seeds.csv'), index=False)

    lag_rows = []
    for a in arms:
        big = pd.concat(per_lag[a])
        for L, g in big.groupby('lag'):
            lag_rows.append({'arm': a, 'lag': L, 'n_seeds': len(g), 'H_bias': (g.H_hat - g.H_exact).mean(),
                             'H_bias_se': (g.H_hat - g.H_exact).std(ddof=1) / np.sqrt(len(g)),
                             'D_hat': g.D_hat.mean(), 'D_exact': g.D_exact.mean(),
                             'D_bias': (g.D_hat - g.D_exact).mean(),
                             'D_bias_se': (g.D_hat - g.D_exact).std(ddof=1) / np.sqrt(len(g)),
                             'SE_D_median': g.SE_D.median()})
    lags = pd.DataFrame(lag_rows)
    lags.to_csv(os.path.join(out_dir, args.out + '_lags.csv'), index=False)

    # ---- report -------------------------------------------------------------------------------
    n = df.seed.nunique()
    print(f'\n{n} seeds, K*={K:.3g}, mean-over-theta exact tolerance lag {int(df.L_exact_tol_mean_theta.iloc[0])}')
    print(df.groupby('arm').agg(L_star_mean=('L_star', 'mean'), L_star_sd=('L_star', 'std'),
                                L_tol=('L_tol', 'mean'), L_cv=('L_cv', 'mean'),
                                unresolved=('fit_status', lambda s: float((s != 'ok').mean())),
                                rmse_D=('rmse_D_vs_exact', 'mean'), H0_bias=('H0_minus_exact', 'mean'),
                                H_bias_Lcap=('H_bias_at_Lcap', 'mean'),
                                train_s=('seconds_train', 'mean')).round(4).to_string())
    ref = args.ref
    if ref in arms:
        piv = df.pivot(index='seed', columns='arm', values='L_star')
        for a in arms:
            if a == ref:
                continue
            d = (piv[a] - piv[ref]).abs()
            print(f'\n{a} against {ref}: |L*_{a} - L*_{ref}| <= 1 in {100 * (d <= 1).mean():.0f} % of seeds '
                  f'(mean signed difference {np.mean(piv[a] - piv[ref]):+.2f}); '
                  f'train time ratio {df[df.arm == a].seconds_train.mean() / df[df.arm == ref].seconds_train.mean():.2f}')
            print(f'   {"lag":>3} {"D_exact":>9} {"D_" + ref:>10} {"D_" + a:>10} {"ratio":>7} '
                  f'{"diff":>9} {"se(diff)":>9} {"z":>6}')
            Dr = np.array([x for x in paired[ref]])
            for L in COMPARE_LAGS:
                dref = np.array([p[:, L].mean() for p in paired[ref]])
                da = np.array([p[:, L].mean() for p in paired[a]])
                diff = da - dref
                se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
                dex = lags[(lags.arm == a) & (lags.lag == L)].D_exact.iloc[0]
                print(f'   {L:>3} {dex:9.5f} {dref.mean():10.5f} {da.mean():10.5f} '
                      f'{da.mean() / dref.mean():7.3f} {diff.mean():+9.5f} {se:9.5f} {diff.mean() / se:+6.1f}')

    # ---- figure -------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a in arms:
        g = lags[lags.arm == a]
        ax[0].errorbar(g.lag, g.D_bias, g.D_bias_se, marker='o', ms=3, capsize=2, label=a)
        ax[1].errorbar(g.lag, g.H_bias, g.H_bias_se, marker='o', ms=3, capsize=2, label=a)
        ax[2].plot(g.lag, g.SE_D_median, marker='o', ms=3, label=a)
    for k in (0, 1):
        ax[k].axhline(0, color='k', lw=0.8)
    ax[0].set_title(r'$\hat D(L)-D(L)$, mean over seeds')
    ax[1].set_title(r'$\hat H(L)-H(L)$, mean over seeds')
    ax[2].set_title(r'median paired SE of $\hat D(L)$')
    ax[2].set_yscale('log')
    for a_ in ax:
        a_.set_xlabel('lag $L$')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, args.out + '.png'), dpi=140)
    print('\nwrote', os.path.join(out_dir, args.out + '_{seeds.csv,lags.csv}'), 'and .png')


if __name__ == '__main__':
    main()
