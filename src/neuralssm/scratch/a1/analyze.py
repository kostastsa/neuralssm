"""A1 / A5 analysis (decision P-10): T-SNL posteriors against the exact Kalman posterior.

Reads the trials that main.py trials wrote for the descriptor files of make_descs.py and computes, for
each test bed, arm and trial:
  * w1        Wasserstein-1 distance between the T-SNL posterior samples of z = 0.5 log Q and the exact
              posterior of z (grid, Kalman filter with the simulator's start x_1 ~ N(0, P1))
  * bias_sd   (mean of the samples - exact posterior mean) / exact posterior sd
  * sd_ratio  sd of the samples / exact posterior sd
  * w1_trunc  W1 between the exact *truncated* posterior at the arm's lag and the exact posterior
              (truncation bias alone, no network error)
  * L         the lag of the trial (L* for lag: auto), and for auto the fields of lag_selection.json
Writes data/shared_results/A1_posterior_vs_lag.csv, A1_summary.csv and A1_posterior_vs_lag.png.

Run from src/neuralssm: python scratch/a1/analyze.py [--trials 1 20]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import experiment_descriptor as ed  # noqa: E402
import misc  # noqa: E402
import util.io  # noqa: E402
from make_descs import ARMS, TEST_BEDS, OUT as DESC_DIR  # noqa: E402

P1 = 0.1                       # initial state variance (lgssm_params: initial_covariance = 0.1 I)
Z_GRID = np.linspace(-6.0, 4.0, 2001)
OUT_DIR = os.path.join(ROOT, 'data', 'shared_results')
DEFAULT_MODEL = (0.9, 0.1)     # (a, R) when the descriptor has no override


def kf_loglik(y, a, R, Q, P1=P1):
    """Exact log p(y_{1:T} | Q) for x_1 ~ N(0, P1), x_t = a x_{t-1} + q, y_t = x_t + r. Q is an array."""
    m = np.zeros_like(Q); P = np.full_like(Q, P1); ll = np.zeros_like(Q)
    for yt in y:
        S = P + R
        ll += -0.5 * (np.log(2 * np.pi * S) + (yt - m) ** 2 / S)
        K = P / S
        m = m + K * (yt - m); P = (1 - K) * P
        m = a * m; P = a * a * P + Q
    return ll


def kf_loglik_trunc(y, a, R, Q, L, P1=P1):
    """Exact truncated log-likelihood sum_t log p(y_t | y_{max(1,t-L):t-1}, Q).

    The window starts from the unconditional law of x_{t-L} under the simulator (variance V_{t-L}), which is
    the law that the T-SNL training windows see.
    """
    T = len(y)
    V = np.empty((T,) + Q.shape); V[0] = P1
    for t in range(1, T):
        V[t] = a * a * V[t - 1] + Q
    ll = np.zeros_like(Q)
    for t in range(T):
        s = max(0, t - L)
        m = np.zeros_like(Q); P = V[s].copy()
        for u in range(s, t):
            S = P + R; K = P / S
            m = m + K * (y[u] - m); P = (1 - K) * P
            m = a * m; P = a * a * P + Q
        S = P + R
        ll += -0.5 * (np.log(2 * np.pi * S) + (y[t] - m) ** 2 / S)
    return ll


def grid_posterior(ll):
    lp = -0.5 * Z_GRID ** 2 + ll          # prior z ~ N(0, 1)
    w = np.exp(lp - lp.max()); w /= np.trapz(w, Z_GRID)
    return w


def moments(w):
    mu = np.trapz(Z_GRID * w, Z_GRID)
    return mu, np.sqrt(np.trapz((Z_GRID - mu) ** 2 * w, Z_GRID))


def cdf(w):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(Z_GRID))])
    return c / c[-1]


def w1_samples(z, F):
    Fs = np.searchsorted(np.sort(z), Z_GRID, side='right') / len(z)
    return np.trapz(np.abs(Fs - F), Z_GRID)


def load_samples(trial_dir):
    post = util.io.load(os.path.join(trial_dir, 'posterior'))
    cond = np.asarray(post[1] if isinstance(post, (tuple, list)) else post)
    return cond.reshape(cond.shape[0], -1)[:, 0]


def test_bed_of(a, R):
    for k, (ta, tr) in TEST_BEDS.items():
        if (ta if ta is not None else DEFAULT_MODEL[0]) == a and (tr if tr is not None else DEFAULT_MODEL[1]) == R:
            return k
    raise ValueError((a, R))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', nargs=2, type=int, default=[1, 20])
    args = ap.parse_args()
    files = [os.path.join(DESC_DIR, f'a1_lag_{arm}.txt') for arm in ARMS + ['oracle']]
    rows, cache = [], {}
    for f in files:
        arm_file = os.path.basename(f)[len('a1_lag_'):-4]
        for desc in ed.parse(util.io.load_txt(f)):
            sim = desc.sim
            a = sim.dyn_weight if sim.dyn_weight is not None else DEFAULT_MODEL[0]
            R = sim.emission_cov if sim.emission_cov is not None else DEFAULT_MODEL[1]
            tb = test_bed_of(a, R)
            exp_dir = os.path.join(misc.get_root(), 'experiments', desc.get_dir(), 'sample_gt_False')
            for trial in range(args.trials[0], args.trials[1] + 1):
                tdir = os.path.join(exp_dir, str(trial))
                if not os.path.exists(os.path.join(tdir, 'exp_desc.txt')):
                    continue
                (true_ps, true_cps), y = util.io.load(os.path.join(tdir, 'gt'))
                y = np.asarray(y).reshape(-1)
                key = (tb, trial)
                if key not in cache:   # the observations are the same in all arms (seed s = trial number)
                    w = grid_posterior(kf_loglik(y, a, R, np.exp(2 * Z_GRID)))
                    cache[key] = dict(y=y, w=w, F=cdf(w), mom=moments(w), trunc={})
                c = cache[key]
                if not np.allclose(c['y'], y):
                    print(f'WARNING {tb} trial {trial} {arm_file}: observations differ from the first arm')
                sel = {}
                if desc.inf.lag_is_auto():
                    with open(os.path.join(tdir, 'lag_selection.json')) as fh:
                        sel = json.load(fh)
                    L = int(sel['L_star'])
                else:
                    L = int(desc.inf.lag)
                if L not in c['trunc']:
                    wt = grid_posterior(kf_loglik_trunc(c['y'], a, R, np.exp(2 * Z_GRID), L))
                    c['trunc'][L] = np.trapz(np.abs(cdf(wt) - c['F']), Z_GRID)
                z = load_samples(tdir)
                mu, sd = c['mom']
                rows.append(dict(
                    test_bed=tb, a=a, R=R, arm=arm_file, trial=trial, seed=trial, L=L,
                    w1=w1_samples(z, c['F']), bias_sd=(z.mean() - mu) / sd, sd_ratio=z.std() / sd,
                    w1_trunc=c['trunc'][L], z_true=float(np.asarray(true_cps).reshape(-1)[0]),
                    exact_mean=mu, exact_sd=sd,
                    trial_time_s=float(util.io.load(os.path.join(tdir, 'trial_time'))),
                    L0=sel.get('L0'), L_tol=sel.get('L_tol'), L_cv=sel.get('L_cv'),
                    fit_status=sel.get('fit_status'), resolution_failure=sel.get('resolution_failure'),
                    damaged_fraction=sel.get('damaged_fraction'),
                ))
    if not rows:
        print('no finished trials'); return
    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, 'A1_posterior_vs_lag.csv'), index=False)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    summ = df.groupby(['test_bed', 'arm']).agg(
        n=('w1', 'size'), L_mean=('L', 'mean'), L_min=('L', 'min'), L_max=('L', 'max'),
        w1_median=('w1', 'median'), w1_mean=('w1', 'mean'), w1_se=('w1', se),
        bias_sd_mean=('bias_sd', 'mean'), sd_ratio_mean=('sd_ratio', 'mean'),
        w1_trunc_mean=('w1_trunc', 'mean'), time_mean_s=('trial_time_s', 'mean')).reset_index()
    summ.to_csv(os.path.join(OUT_DIR, 'A1_summary.csv'), index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 20):
        print(summ.round(4).to_string(index=False))
    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tbs = [t for t in TEST_BEDS if t in set(df.test_bed)]
    fig, axes = plt.subplots(1, len(tbs), figsize=(4.4 * len(tbs), 3.6), squeeze=False)
    for ax, tb in zip(axes[0], tbs):
        d = df[df.test_bed == tb]
        fixed = d[~d.arm.isin(['auto', 'oracle'])]
        if len(fixed):
            s = fixed.groupby('L').agg(m=('w1', 'mean'), se=('w1', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
                                        t=('w1_trunc', 'mean'))
            ax.errorbar(s.index, s.m, yerr=s.se, marker='D', color='green', label='T-SNL, fixed L')
            ax.plot(s.index, s.t, ls='--', color='gray', label='exact truncated (no NN)')
        for arm, mk, col in (('auto', 'o', 'black'), ('oracle', '*', 'tab:orange')):
            e = d[d.arm == arm]
            if len(e):
                ax.errorbar(e.L.mean(), e.w1.mean(), xerr=e.L.std(ddof=0),
                            yerr=e.w1.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else 0,
                            marker=mk, ms=9, ls='none', color=col, label=f'{arm} (mean L {e.L.mean():.1f})')
        ax.set_xscale('symlog', linthresh=1); ax.set_xlabel('lag L'); ax.set_ylabel('W1 to exact posterior of z')
        ax.set_title(f'{tb}: a={d.a.iloc[0]}, R={d.R.iloc[0]}, {d.trial.nunique()} seeds', fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'A1_posterior_vs_lag.png')
    fig.savefig(path, dpi=150)
    print('figure:', path)


if __name__ == '__main__':
    main()
