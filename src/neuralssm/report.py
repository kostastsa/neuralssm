"""
Static HTML report for a set of experiments.

Reads the per-trial results already on disk (posterior/results, gt, trial_time, seed.txt and,
if `errors` was run, the saved error metrics) and writes one self-contained HTML file:
a sortable summary table, error / time strip plots, and per-experiment details with rank
histograms and a trial table.

Usage:
    python main.py report <sample_gt> <start> <end> <files...> [--out report.html]
"""
import base64
import html
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import experiment_descriptor as ed
import misc
import util.io
from util.numerics import compute_all_errors

METRICS = ['kde_error', 'min_error', 'bias', 'sdev']
COLORS = {'abc': '#5aa9ff', 'mcmc': '#ff6b6b', 'snl': '#ff9f6b', 'tsnl': '#5fd38d'}
BG, FG, GRID = '#14161c', '#e4e6eb', '#343843'
DARK_RC = {'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
           'axes.edgecolor': GRID, 'axes.labelcolor': FG, 'text.color': FG,
           'xtick.color': FG, 'ytick.color': FG, 'grid.color': GRID}
LABELS = {'abc': 'SMC-ABC', 'mcmc': 'BPF-MCMC', 'snl': 'SNL', 'tsnl': 'T-SNL'}


def _alg_key(inf):
    if isinstance(inf, ed.TSNL_Descriptor):
        return 'tsnl'
    if isinstance(inf, ed.SNL_Descriptor):
        return 'snl'
    if isinstance(inf, ed.ABC_Descriptor):
        return 'abc'
    return 'mcmc'


def _load_posterior(inf, exp_dir):
    if isinstance(inf, (ed.SNL_Descriptor, ed.TSNL_Descriptor)):
        return util.io.load(os.path.join(exp_dir, 'posterior'))[1]
    return util.io.load(os.path.join(exp_dir, 'results'))[0]


def _load_trial(exp_desc, sample_gt, trial):
    """Returns a dict of metrics for one trial, or None if the trial has no results."""
    exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir(),
                           f'sample_gt_{sample_gt}', str(trial))
    if not (os.path.exists(os.path.join(exp_dir, 'posterior.pkl'))
            or os.path.exists(os.path.join(exp_dir, 'results.pkl'))):
        return None

    row = {'trial': trial}
    try:
        (_, true_cps), _ = util.io.load(os.path.join(exp_dir, 'gt'))
        cps = np.asarray(_load_posterior(exp_desc.inf, exp_dir))
        true_cps = np.asarray(true_cps)
    except Exception as e:  # unreadable trial: report it rather than crash the whole page
        row['error'] = repr(e)[:120]
        return row

    saved = {m: os.path.join(exp_dir, m + '.pkl') for m in METRICS}
    if all(os.path.exists(p) for p in saved.values()):
        for m in METRICS:
            row[m] = float(np.asarray(util.io.load(saved[m][:-4])))
    else:
        kde, mine, b, _, s, *_ = compute_all_errors(cps, true_cps)
        row.update(kde_error=float(kde), min_error=float(mine), bias=float(b), sdev=float(s))

    cps2 = cps.reshape(cps.shape[0], -1)
    row['ranks'] = (cps2 < true_cps.reshape(1, -1)).mean(axis=0)

    for name, path in [('time', 'trial_time.pkl'), ('log_sims', 'num_sims.pkl')]:
        p = os.path.join(exp_dir, path)
        row[name] = float(np.asarray(util.io.load(p[:-4]))) if os.path.exists(p) else np.nan
    seed_path = os.path.join(exp_dir, 'seed.txt')
    row['seed'] = util.io.load_txt(seed_path).strip() if os.path.exists(seed_path) else ''
    return row


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    return '<img src="data:image/png;base64,%s">' % base64.b64encode(buf.getvalue()).decode()


def _distinguishing_labels(entries):
    """Short row labels: the descriptor path parts that differ among rows of the same method."""
    parts = {i: e['desc'].sim.get_dir().split('/') + e['desc'].inf.get_dir().split('/')
             for i, e in enumerate(entries)}
    labels = {}
    for i, e in enumerate(entries):
        same = [j for j, f in enumerate(entries) if f['alg'] == e['alg'] and len(parts[j]) == len(parts[i])]
        keep = [k for k in range(len(parts[i])) if len({parts[j][k] for j in same}) > 1]
        all_sims = {tuple(parts[j][:len(e['desc'].sim.get_dir().split('/'))]) for j in range(len(entries))}
        extra = [] if len(all_sims) == 1 else [e['desc'].sim.get_dir().split('/')[-1]]
        labels[i] = LABELS[e['alg']] + (': ' + ' | '.join(extra + [parts[i][k] for k in keep])
                                        if keep or extra else '')
    return labels


def _fmt(x):
    return '–' if x is None or not np.isfinite(x) else f'{x:.3g}'


def _stat(rows, key):
    v = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])])
    if v.size == 0:
        return None, None, None
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


@plt.rc_context(DARK_RC)
def _strip_plot(entries, labels, key, title):
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(entries) + 1))
    rng = np.random.default_rng(0)
    for i, e in enumerate(entries):
        v = np.array([r[key] for r in e['rows'] if key in r and np.isfinite(r[key])])
        if v.size == 0:
            continue
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, v.size), s=8, alpha=0.5, color=COLORS[e['alg']])
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=FG, lw=1.5)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([labels[i] for i in range(len(entries))], fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    return _png(fig)


@plt.rc_context(DARK_RC)
def _rank_plot(rows, color):
    ranks = np.array([r['ranks'] for r in rows if 'ranks' in r])
    if ranks.size == 0:
        return ''
    n_par = ranks.shape[1]
    fig, axes = plt.subplots(1, n_par, figsize=(1.8 * n_par + 0.4, 1.6), squeeze=False)
    n_bins = 10
    lo, hi = np.percentile(np.random.default_rng(0).binomial(len(ranks), 1 / n_bins, 10000), [2.5, 97.5])
    for d, ax in enumerate(axes[0]):
        ax.hist(ranks[:, d], bins=n_bins, range=(0, 1), color=color)
        ax.axhspan(lo, hi, color=FG, alpha=0.18)
        ax.set_title(f'param {d}', fontsize=7)
        ax.tick_params(labelsize=6)
    return _png(fig)


def build_report(exp_descs, sample_gt, start, end):
    entries = []
    for desc in exp_descs:
        rows = [r for t in range(start, end + 1) if (r := _load_trial(desc, sample_gt, t)) is not None]
        entries.append({'desc': desc, 'alg': _alg_key(desc.inf), 'rows': rows})
    labels = _distinguishing_labels(entries)
    n_req = end - start + 1

    table = []
    for i, e in enumerate(entries):
        good = [r for r in e['rows'] if 'error' not in r]
        cells = [f'<td><span class="dot" style="background:{COLORS[e["alg"]]}"></span>{html.escape(labels[i])}</td>',
                 f'<td data-v="{len(good)}">{len(good)} / {n_req}</td>']
        for m in METRICS + ['time']:
            med, q1, q3 = _stat(good, m)
            cells.append(f'<td data-v="{med if med is not None else ""}" title="IQR {_fmt(q1)} – {_fmt(q3)}">{_fmt(med)}</td>')
        ls = _stat(good, 'log_sims')[0]
        cells.append(f'<td data-v="{ls if ls is not None else ""}">{_fmt(np.exp(ls)) if ls is not None else "–"}</td>')
        table.append('<tr>' + ''.join(cells) + '</tr>')
    head = ''.join(f'<th>{h}</th>' for h in
                   ['experiment', 'trials'] + [m + ' (median)' for m in METRICS] + ['time s', 'num_sims'])

    ranks = ''.join(
        f'<div class="card"><div class="cap"><span class="dot" style="background:{COLORS[e["alg"]]}"></span>'
        f'{html.escape(labels[i])}</div>{_rank_plot(e["rows"], COLORS[e["alg"]])}</div>'
        for i, e in enumerate(entries))

    plots = _strip_plot(entries, labels, 'kde_error', 'kde_error per trial (white tick = median)') \
        + _strip_plot(entries, labels, 'time', 'trial time (s)')

    details = []
    for i, e in enumerate(entries):
        trs = ''.join(
            f'<tr><td>{r["trial"]}</td><td>{html.escape(r["seed"]) if "seed" in r else ""}</td>'
            + (f'<td colspan="5">{html.escape(r["error"])}</td>' if 'error' in r else
               ''.join(f'<td>{_fmt(r[m])}</td>' for m in METRICS + ['time']))
            + '</tr>' for r in e['rows'])
        missing = [t for t in range(start, end + 1) if t not in {r['trial'] for r in e['rows']}]
        details.append(
            f'<details><summary>{html.escape(labels[i])}</summary>'
            f'<pre>{html.escape(e["desc"].pprint())}</pre>'
            f'<p>missing trials: {missing if missing else "none"}</p>'
            f'<table class="t"><tr><th>trial</th><th>seed</th>'
            + ''.join(f'<th>{m}</th>' for m in METRICS + ['time']) + f'</tr>{trs}</table></details>')

    return PAGE.format(title=f'Experiment report (sample_gt={sample_gt}, trials {start}–{end})',
                       head=head, rows=''.join(table), plots=plots, ranks=ranks, details=''.join(details))


def report(args):
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    page = build_report(exp_descs, args.sample_gt, args.start, args.end)
    out = args.out or os.path.join(
        misc.get_root(), 'reports',
        f'{os.path.splitext(os.path.basename(args.files[0]))[0]}_{args.sample_gt}_{args.start}_{args.end}.html')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        f.write(page)
    print(f'[report] Saved → {out}')


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark">
<style>
:root{{--bg:#14161c;--panel:#1c1f27;--fg:#e4e6eb;--mut:#9097a5;--line:#343843}}
body{{font:14px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:var(--fg);background:var(--bg)}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px;color:var(--mut);font-weight:600}} p{{color:var(--mut)}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{cursor:pointer;background:var(--panel);position:sticky;top:0}}
.t{{width:auto;font-size:12px}} .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
details{{margin:6px 0}} summary{{cursor:pointer}} pre{{background:var(--panel);padding:8px;font-size:12px;overflow:auto}}
img{{max-width:100%}} .plots,.ranks{{display:flex;flex-wrap:wrap;gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px 10px}} .cap{{font-size:12px;margin-bottom:4px}}
</style></head><body><h1>{title}</h1>
<p>Medians over completed trials; hover a cell for the IQR. Click a header to sort.</p>
<table id="s"><tr>{head}</tr>{rows}</table>
<h2>Per-trial spread</h2><div class="plots">{plots}</div>
<h2>Calibration (rank histograms; shaded band = 95% under uniformity)</h2><div class="ranks">{ranks}</div>
<h2>Details</h2>{details}
<script>
document.querySelectorAll('#s th').forEach((th,c)=>th.onclick=()=>{{
 const t=document.getElementById('s'),rs=[...t.rows].slice(1),d=th.dataset.d=th.dataset.d==='1'?-1:1;
 const v=r=>{{const x=r.cells[c].dataset.v;return x===undefined||x===''?NaN:parseFloat(x)}};
 rs.sort((a,b)=>c==0?d*a.cells[0].textContent.localeCompare(b.cells[0].textContent):d*((isNaN(v(a))?1e99:v(a))-(isNaN(v(b))?1e99:v(b))));
 rs.forEach(r=>t.tBodies[0].appendChild(r));}});
</script></body></html>"""
