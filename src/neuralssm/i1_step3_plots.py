r"""
Figure and tidy CSV for the I1 step-3 runs (``i1_step3_lgssm.py``).

Reads ``data/shared_results/I1_step3_<mode>_N*_T*_*_seed*.npz`` and writes
``I1_step3_diagnostics.csv`` (one row per run, lag) and ``I1_step3_decrements.png``.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import misc  # noqa: E402
from inference import lag_selection as ls  # noqa: E402
import lgssm_exact as ex  # noqa: E402

d = os.path.join(misc.get_root(), 'shared_results')
files = sorted(glob.glob(os.path.join(d, 'I1_step3_*_N*_T*_seed*.npz')))
rows, fig_data = [], []
for f in files:
    tag = os.path.basename(f)[len('I1_step3_'):-4]
    z = np.load(f)
    nll, mode, T = z['nll'], tag.split('_')[0], int(tag.split('_T')[1].split('_')[0])
    dec = ls.decrements_from_nll(nll)
    Dex = z['D_exact_per_theta'].mean(0)
    Hex = z['H_exact_per_theta'].mean(0)
    sel = ls.select_lag(nll, T, ls.kl_tolerance(T))
    for L in range(nll.shape[1]):
        r = dict(run=tag, mode=mode, lag=L, H_hat=dec['H'][L], SE_H=dec['SE_H'][L], H_exact=Hex[L])
        if L < nll.shape[1] - 1:
            r.update(D_hat=dec['D'][L], SE_D=dec['SE_D'][L], D_exact=Dex[L],
                     z=(dec['D'][L] - Dex[L]) / dec['SE_D'][L], significant=bool(sel['mask'][L]))
        rows.append(r)
    fig_data.append((mode, dec, Dex, Hex, sel))
pd.DataFrame(rows).to_csv(os.path.join(d, 'I1_step3_diagnostics.csv'), index=False)

fig, ax = plt.subplots(2, len(fig_data), figsize=(5.2 * len(fig_data), 6.4), squeeze=False)
for j, (mode, dec, Dex, Hex, sel) in enumerate(fig_data):
    L = np.arange(len(dec['D']))
    a = ax[0, j]
    a.axhline(0, color='k', lw=0.5)
    a.plot(L, Dex, 'k-', label='exact $D(L)$')
    a.errorbar(L, dec['D'], 2 * dec['SE_D'], fmt='o', ms=3.5, color='tab:green', capsize=2,
               label=r'$\hat D(L)\pm2\,$SE (paired)')
    a.set_yscale('symlog', linthresh=1e-3)
    a.axvline(sel['L_star'], color='tab:red', ls='--', lw=1, label=f"$L^\\star={sel['L_star']}$")
    a.set_title(f'{mode}: decrements'); a.set_xlabel('lag $L$'); a.legend(fontsize=8)
    b = ax[1, j]
    Lh = np.arange(len(dec['H']))
    b.errorbar(Lh, dec['H'] - Hex, dec['SE_H'], fmt='o', ms=3.5, color='tab:blue', capsize=2)
    b.axhline(0, color='k', lw=0.5)
    b.set_ylim(-0.05, 0.08)
    b.set_title(r'$\hat H(L)-H(L)$ (exact), bars: unpaired SE'); b.set_xlabel('lag $L$')
fig.tight_layout()
fig.savefig(os.path.join(d, 'I1_step3_decrements.png'), dpi=150)
print('wrote', len(rows), 'rows,', len(files), 'runs')
