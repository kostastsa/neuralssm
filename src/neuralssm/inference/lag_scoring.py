r"""
Lag-dropout network and held-out scoring for lag selection (request I1).

One conditional MAF, the *lag-dropout network*, evaluates every candidate lag
:math:`\ell\in\{0,\dots,L_{\rm grid}\}` (decision D2, 2026-10-21). It is trained on the
round-0 trajectories and scored on held-out trajectories. The output is the
array ``nll[n, ell]`` that :func:`inference.lag_selection.select_lag` consumes.
Training and scoring run in JAX. The selection itself runs on the host (NumPy).

Density estimated at lag :math:`\ell`
-------------------------------------
.. math:: q^{(\ell)}_\phi(y_t\mid y_{t-\ell:t-1},\theta),

with :math:`d`-dimensional target :math:`y_t`. The network has the lag
:math:`L_{\rm grid}`. Its conditioning vector has the size :math:`p+L_{\rm grid}+L_{\rm grid}d`
for every :math:`\ell`:

.. math:: (\theta,\ m_{L_{\rm grid}},\dots,m_1,\ \tilde y_{t-L_{\rm grid}},\dots,\tilde y_{t-1}),
          \qquad m_k=1[k\le\ell],\quad \tilde y_{t-k}=m_k\,y_{t-k}.

The observations older than :math:`\ell` are set to zero and the indicator
:math:`m` tells the network which lag it evaluates. During training every row
gets its own lag, drawn uniformly from :math:`\{0,\dots,L_{\rm grid}\}`. The rows
before masking have the production layout of ``util.train.lag_ds``:
``[theta, y_{t-L_grid}, ..., y_t]``.

Held-out score
--------------
For held-out trajectory :math:`n` and lag :math:`\ell`,

.. math::
    \mathrm{nll}_{n\ell}=-\frac{1}{T-L_{\rm cap}}\sum_{t=L_{\rm cap}+1}^{T}
    \log q^{(\ell)}_\phi\bigl(y^{(n)}_t\mid y^{(n)}_{t-\ell:t-1},\theta^{(n)}\bigr).

:math:`L_{\rm cap}` is the largest lag that the method can examine
(:math:`\lfloor T/4\rfloor` by default). The *scoring window*
:math:`t\in\{L_{\rm cap}+1,\dots,T\}` is the same for every lag
:math:`\ell\le L_{\rm cap}`, and it does not depend on the grid
:math:`\{0,\dots,L_{\rm grid}\}`. The zero-padding that ``reshape_emissions``
applies at :math:`t\le\ell` does not enter the comparison.

Entry points
------------
* :func:`select_lag_round0` (Option A, ``lag: auto``, and round 0 of Option B): stage 0
  and stage 1 on the round-0 simulations, with grid growth.
* :func:`reselect_lag` (Option B, ``lag: ada``): stage 1 on the new simulations of a
  round, scored with the network of the previous round.
* :class:`LagMaskedModel`: the network, fixed at one lag, seen as a plain T-SNL model.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import nnx

from inference import lag_selection as ls
from maf.density_models import MAF
from maf.layers import BatchNormLayer
from util.train import train_step


# Hyperparameters of the T-SNL experiments (create_exps.py, alg == 'tsnl').
PRODUCTION_HP = dict(nmade=5, dhidden=32, nhidden=5, act_fun='relu',
                     random_order=False, reverse=True, batch_norm=False, dropout=False,
                     epochs=20, lr=1e-4, weight_decay=1e-4, batch_size=128)


def hp_from_model_desc(model_desc):
    """
    Candidate hyperparameters from a ``MAF_Descriptor``, so the candidates have the
    architecture and the training budget of the T-SNL network. Batch size 128 and
    weight decay 1e-4 are fixed in ``SequentialNeuralLikelihood``.
    """
    return dict(PRODUCTION_HP, nmade=model_desc.nmades, dhidden=model_desc.dhidden,
                nhidden=model_desc.nhidden, act_fun=model_desc.act_fun,
                random_order=model_desc.random_order, reverse=model_desc.reverse,
                batch_norm=model_desc.batch_norm, dropout=model_desc.dropout,
                epochs=model_desc.nepochs, lr=model_desc.lr)


def make_windows(emissions, L, first=0):
    r"""
    Lagged windows of each trajectory.

    Row :math:`t` is :math:`(y_{t-L},\dots,y_t)` flattened, oldest first, with
    zeros for indices before the start (the convention of
    ``util.train.reshape_emissions``).

    Parameters
    ----------
    emissions : array (N, T, d)
    L : int
    first : int
        First target index (0-based). ``0`` gives all ``T`` windows;
        ``first = L_cap`` gives the scoring window.

    Returns
    -------
    array (N, T - first, (L + 1) * d)
    """
    N, T, d = emissions.shape
    pad = jnp.concatenate([jnp.zeros((N, L, d), emissions.dtype), emissions], axis=1)
    win = jnp.concatenate([pad[:, k:k + T] for k in range(L + 1)], axis=2)
    return win[:, first:]


def make_rows(cond_params, emissions, L, first=0):
    """Rows ``[theta, y_{t-L}, ..., y_t]``, shape (N, T - first, p + (L+1) d)."""
    win = make_windows(emissions, L, first)
    cp = jnp.broadcast_to(cond_params[:, None, :], win.shape[:2] + cond_params.shape[1:])
    return jnp.concatenate([cp, win], axis=2)


def freeze_batch_stats(model, ref_rows):
    r"""
    Switch to evaluation mode and fix the batch-norm statistics from
    ``ref_rows`` (a large random subset of the training rows).

    In evaluation mode ``BatchNormLayer`` takes its statistics from the first
    batch it sees. Passing one large reference batch first makes the scores
    independent of the batch that is scored.
    """
    model.eval()
    for layer in model.layers:
        if isinstance(layer, BatchNormLayer):
            layer.batch_mean = None
    model.loss_fn(ref_rows)


@nnx.jit
def _nll_per_row(model, x):
    out, log_det = model(x)
    u = out[:, model.dcond:]
    return 0.5 * jnp.sum(u ** 2, axis=1) + 0.5 * u.shape[1] * jnp.log(2 * jnp.pi) - log_det


def split_trajectories(key, n, holdout_frac=0.15):
    """Random split by trajectory. Returns (train_idx, heldout_idx)."""
    perm = np.asarray(jr.permutation(key, n))
    n_h = max(2, int(round(holdout_frac * n)))
    return perm[n_h:], perm[:n_h]


# ---------------------------------------------------------------------------
# Lag-dropout network (decision D2)
# ---------------------------------------------------------------------------
#
# One network of lag L_grid evaluates every lag ell in 0..L_grid. Its input row is
#     [theta, m_{L_grid}, ..., m_1, y_{t-L_grid}, ..., y_{t-1}, y_t]
# with the lag indicator m_k = 1 if k <= ell and 0 otherwise, and with the
# observations older than ell set to zero. The row has p + L_grid + L_grid*d
# conditioning columns, whatever ell is.

# Training budget of the lag-dropout network, as a multiple of the epochs of the
# descriptor (see select_lag_round0). 4 is the value with the smallest held-out NLL
# error among 1, 4 and 10 on one tuning seed (seed 1000, d4 LGSSM,
# I1_D2_tuning_summary_lags.csv).
LD_EPOCHS_FACTOR = 4.0

# Number of XLA backend compilations in this process (all functions). Option B logs the
# difference around each phase of a round, to show that a change of the lag does not
# compile the training step again.
_COMPILES = [0]


def _count_compile(event, duration_secs, **kwargs):
    if event == '/jax/core/compile/backend_compile_duration':
        _COMPILES[0] += 1


jax.monitoring.register_event_duration_secs_listener(_count_compile)


def n_compilations():
    """Number of XLA backend compilations in this process so far."""
    return _COMPILES[0]


def lag_mask(ell, L_grid):
    r"""
    Lag indicator :math:`m_k=1[k\le\ell]`, :math:`k=L_{\rm grid},\dots,1`
    (oldest first, the order of the observations in the row).

    Parameters
    ----------
    ell : int array (n,)
    L_grid : int

    Returns
    -------
    bool array (n, L_grid)
    """
    k = L_grid - jnp.arange(L_grid)
    return k[None, :] <= jnp.asarray(ell)[:, None]


@partial(jax.jit, static_argnames=('p', 'L_grid', 'd'))
def apply_lag_mask(rows, ell, p, L_grid, d):
    r"""
    Lag-dropout input from full-lag rows.

    Parameters
    ----------
    rows : array (n, p + (L_grid + 1) d)
        Rows ``[theta, y_{t-L_grid}, ..., y_t]`` as :func:`make_rows` returns
        them for ``L = L_grid``.
    ell : int array (n,) or int
        Lag of each row. The observations older than ``ell`` are set to zero.
    p, L_grid, d : int

    Returns
    -------
    array (n, p + L_grid + L_grid d + d)
        ``[theta, m, masked y_{t-L_grid..t-1}, y_t]``.
    """
    n = rows.shape[0]
    ell = jnp.broadcast_to(jnp.asarray(ell), (n,))
    keep = lag_mask(ell, L_grid)
    past = rows[:, p:p + L_grid * d].reshape(n, L_grid, d) * keep[:, :, None]
    return jnp.concatenate([rows[:, :p], keep.astype(rows.dtype), past.reshape(n, L_grid * d),
                            rows[:, p + L_grid * d:]], axis=1)


def build_dropout_candidate(key, L_grid, n_params, d, hp=PRODUCTION_HP):
    r"""
    MAF for the lag-dropout scheme: data dimension ``d``, conditioning size
    ``n_params + L_grid + L_grid*d`` (the parameters, the lag indicator and the
    masked past).
    """
    return MAF(din=d, nmade=hp['nmade'], dhidden=hp['dhidden'], nhidden=hp['nhidden'],
               act_fun=hp['act_fun'], dcond=n_params + L_grid + L_grid * d, rngs=nnx.Rngs(key),
               random_order=hp['random_order'], reverse=hp['reverse'],
               batch_norm=hp.get('batch_norm', False), dropout=hp.get('dropout', False))


def train_dropout(key, model, rows, p, L_grid, d, hp=PRODUCTION_HP, epochs=None):
    r"""
    Train the lag-dropout network on full-lag ``rows`` by minibatch AdamW.

    Each row of a minibatch gets its own lag, drawn uniformly from
    :math:`\{0,\dots,L_{\rm grid}\}`. The loader shuffles every epoch and drops
    the incomplete last batch, as in :func:`train_candidate`.

    Returns
    -------
    list of float
        Mean training loss per epoch.
    """
    bs = hp['batch_size']
    n = rows.shape[0]
    n_batches = n // bs
    epochs = hp['epochs'] if epochs is None else epochs
    optimizer = nnx.Optimizer(model, optax.adamw(hp['lr'], weight_decay=hp['weight_decay']))
    model.train()
    losses = []
    for _ in range(epochs):
        key, sub, sub_ell = jr.split(key, 3)
        perm = jr.permutation(sub, n)
        ells = jr.randint(sub_ell, (n_batches, bs), 0, L_grid + 1)
        ep = []
        for b in range(n_batches):
            batch = apply_lag_mask(rows[perm[b * bs:(b + 1) * bs]], ells[b], p, L_grid, d)
            ep.append(train_step(model, optimizer, batch))
        losses.append(float(jnp.mean(jnp.stack(ep))))
    return losses


def heldout_nll_all_lags(model, cond_params, emissions, L_grid, L_cap, chunk=8192):
    r"""
    Per-trajectory mean NLL of the lag-dropout network at every lag
    :math:`0,\dots,L_{\rm grid}` on the scoring window :math:`t>L_{\rm cap}`.

    Returns
    -------
    array (N, L_grid + 1)
    """
    N, T, d = emissions.shape
    p = cond_params.shape[1]
    rows = make_rows(cond_params, emissions, L_grid, first=L_cap)
    m = rows.shape[1]
    flat = rows.reshape(-1, rows.shape[2])
    cols = []
    for ell in range(L_grid + 1):
        out = jnp.concatenate([_nll_per_row(model, apply_lag_mask(flat[i:i + chunk], ell, p, L_grid, d))
                               for i in range(0, flat.shape[0], chunk)])
        cols.append(np.asarray(out.reshape(N, m).mean(axis=1)))
    return np.stack(cols, axis=1)


def score_dropout(key, cond_params, emissions, L_cap, L_grid, hp=PRODUCTION_HP,
                  holdout_frac=0.15, train_windows='all', epochs_factor=LD_EPOCHS_FACTOR,
                  logger=print, n_ref=4096, split=None):
    r"""
    Train one lag-dropout network of lag ``L_grid`` and score every lag
    :math:`0,\dots,L_{\rm grid}` on the held-out trajectories.

    The randomness comes from ``jax.random.fold_in(key, L_grid)``, so the
    result depends on the final grid only and not on the path of the growth
    (the network is trained again from the start when the grid grows).

    Parameters
    ----------
    key : PRNGKey
    cond_params : array (N, p)
        Round-0 parameters in the unconstrained form that T-SNL conditions on.
    emissions : array (N, T, d)
    L_cap : int
        The largest lag that the method can examine. It sets the scoring
        window :math:`t>L_{\rm cap}` for every lag.
    hp : dict
        Hyperparameters, default :data:`PRODUCTION_HP`.
    holdout_frac : float
        Fraction of trajectories held out (split by trajectory). Not used if
        ``split`` is given.
    train_windows : {'all', 'common'}
        ``'all'`` trains on every window with zero padding at the start, as
        production does. ``'common'`` trains on the scoring window only.
    n_ref : int
        Number of training rows used to freeze the batch-norm statistics.
    split : tuple of arrays, optional
        ``(train_idx, heldout_idx)``. Default: a random split from ``key``.
    L_grid : int
        Lag of the network, at most ``L_cap``.
    epochs_factor : float
        The network trains for ``round(epochs_factor * hp['epochs'])`` epochs.

    Returns
    -------
    dict
        ``nll`` (n_heldout, L_grid + 1), ``lags``, ``heldout_idx``,
        ``train_loss`` (loss of each epoch), ``seconds_train``, ``seconds_score``,
        ``seconds_total``.
    """
    cond_params = jnp.asarray(cond_params)
    emissions = jnp.asarray(emissions)
    N, T, d = emissions.shape
    p = cond_params.shape[1]
    if L_grid > L_cap:
        raise ValueError(f'L_grid={L_grid} is above L_cap={L_cap}')
    if split is None:
        split = split_trajectories(jr.fold_in(key, 1_000_000), N, holdout_frac)
    tr, ho = split
    first = 0 if train_windows == 'all' else L_cap
    epochs = max(1, int(round(epochs_factor * hp['epochs'])))

    t_all = time.perf_counter()
    kmodel, ktrain, kref = jr.split(jr.fold_in(key, L_grid), 3)
    model = build_dropout_candidate(kmodel, L_grid, p, d, hp)
    rows = make_rows(cond_params[tr], emissions[tr], L_grid, first).reshape(-1, p + (L_grid + 1) * d)
    losses = train_dropout(ktrain, model, rows, p, L_grid, d, hp, epochs)
    kref, kell = jr.split(kref)
    sel = jr.choice(kref, rows.shape[0], (min(n_ref, rows.shape[0]),), replace=False)
    ref = apply_lag_mask(rows[sel], jr.randint(kell, (sel.shape[0],), 0, L_grid + 1), p, L_grid, d)
    freeze_batch_stats(model, ref)
    jax.block_until_ready(jax.tree_util.tree_leaves(nnx.state(model)))
    t1 = time.perf_counter()
    nll = heldout_nll_all_lags(model, cond_params[ho], emissions[ho], L_grid, L_cap)
    t2 = time.perf_counter()
    logger(f'lag dropout, L_grid={L_grid}, {epochs} epochs: final train loss {losses[-1]:.4f}  '
           f'({t1 - t_all:.1f}s train, {t2 - t1:.1f}s score)\n')
    del model, rows
    jax.clear_caches()
    return {'nll': nll, 'lags': np.arange(L_grid + 1), 'heldout_idx': ho,
            'train_loss': np.array(losses), 'seconds_train': t1 - t_all,
            'seconds_score': t2 - t1, 'seconds_total': time.perf_counter() - t_all}


# ---------------------------------------------------------------------------
# Option B: the network keeps its shape, the lag is data
# ---------------------------------------------------------------------------

class LagDropoutMask:
    """
    Training hook of Option B. Called on a minibatch of full-lag rows
    ``[theta, y_{t-L_max}, ..., y_t]``, it draws a lag for each row, uniformly
    from ``0..L_max``, and returns the lag-dropout input (:func:`apply_lag_mask`).
    The key advances at every call.
    """

    def __init__(self, key, n_params, L_max, d):
        self.key, self.p, self.L_max, self.d = key, int(n_params), int(L_max), int(d)

    def __call__(self, batch):
        self.key, sub = jr.split(self.key)
        ell = jr.randint(sub, (batch.shape[0],), 0, self.L_max + 1)
        return apply_lag_mask(batch, ell, self.p, self.L_max, self.d)


class LagMaskedModel:
    r"""
    The lag-dropout network, fixed at the lag ``ell``, seen as a plain T-SNL
    model of lag ``ell``.

    It takes and gives what a MAF of lag ``ell`` takes and gives: ``loss_fn``
    receives rows ``[theta, y_{t-ell}, ..., y_t]`` (the layout of
    ``util.train.reshape_emissions`` at lag ``ell``), and ``generate`` receives
    ``[theta, y_{t-ell}, ..., y_{t-1}]``. The rows are padded with zeros to the
    lag :math:`L_{\max}` of the network and masked at ``ell``, so the code that
    evaluates the T-SNL likelihood (``logdensity_fn``, ``loglik_fn``,
    ``generate_emissions``) works with ``lag=ell`` and without change.

    Parameters
    ----------
    base : MAF
        The lag-dropout network of lag ``L_max``.
    ell : int
        Lag of the view, ``0 <= ell <= L_max``.
    L_max, n_params, emission_dim : int
    """

    def __init__(self, base, ell, L_max, n_params, emission_dim):
        if not 0 <= ell <= L_max:
            raise ValueError(f'lag {ell} is outside 0..{L_max}')
        self.base, self.ell, self.L_max = base, int(ell), int(L_max)
        self.p, self.d = int(n_params), int(emission_dim)

    def with_lag(self, ell):
        return LagMaskedModel(self.base, ell, self.L_max, self.p, self.d)

    def _pad(self, past):
        return jnp.zeros((past.shape[0], (self.L_max - self.ell) * self.d), past.dtype)

    def loss_fn(self, x):
        full = jnp.concatenate([x[:, :self.p], self._pad(x), x[:, self.p:]], axis=1)
        return self.base.loss_fn(apply_lag_mask(full, self.ell, self.p, self.L_max, self.d))

    def generate(self, key, num_samples, cond_samples):
        theta, past = cond_samples[:, :self.p], cond_samples[:, self.p:]
        keep = lag_mask(jnp.full((cond_samples.shape[0],), self.ell), self.L_max).astype(cond_samples.dtype)
        full = jnp.concatenate([theta, keep, self._pad(past), past], axis=1)
        return self.base.generate(key, num_samples, full)

    def eval(self):
        self.base.eval()

    def train(self):
        self.base.train()


def reference_rows(key, cond_params, emissions, L_max, n_ref=4096):
    """
    About ``n_ref`` lag-dropout rows (random trajectories, all their windows, a
    random lag for each row) to freeze the batch-norm statistics.
    """
    N, T, d = emissions.shape
    p = cond_params.shape[1]
    kt, kl = jr.split(key)
    m = int(min(N, max(1, -(-n_ref // T))))
    idx = jr.choice(kt, N, (m,), replace=False)
    rows = make_rows(cond_params[idx], emissions[idx], L_max).reshape(-1, p + (L_max + 1) * d)
    return apply_lag_mask(rows, jr.randint(kl, (rows.shape[0],), 0, L_max + 1), p, L_max, d)


def reselect_lag(key, model, cond_params, emissions, ref_cond_params, ref_emissions, L_max, L_cap, T,
                 eta=0.1, n_ref=4096, square=False, L_prev=None, hold=True, return_nll=False):
    r"""
    Option B: select the lag of a round from the new simulations of that round.

    The trajectories are held out by construction. The network was trained on
    earlier rounds and has not seen them. They are scored at every lag
    :math:`0,\dots,L_{\max}` on the window :math:`t>L_{\rm cap}`
    (:func:`heldout_nll_all_lags`), and :func:`inference.lag_selection.select_lag`
    gives :math:`L^\star_r`. The grid is fixed at :math:`0..L_{\max}`, because
    the network cannot grow. If the grid limits the result, ``grid_limited`` is
    true and the caller flags the run as capped. Stage 0 runs on the new
    emissions, for the flag ``resolution_failure``.

    The batch-norm statistics of the network are frozen on ``n_ref`` rows of the
    earlier rounds (``ref_cond_params``, ``ref_emissions``), not on the
    trajectories that are scored.

    **Rule against a failed round** (decision E5). The selection is *usable* if
    ``fit_status`` is not ``'none'`` and every held-out NLL is finite. If it is not
    usable and ``hold`` is true, the lag of the previous round ``L_prev`` is kept.
    ``L_star`` is then ``L_prev``, ``L_star_raw`` is the value of the routine, and
    ``held`` is true. In a dependent system even :math:`\hat D(0)` is significant
    when the scores are usable. ``'none'`` then means that the scores failed, for
    example because of trajectories at extreme parameters. A selection with
    ``fit_status`` ``'unresolved'`` is usable.

    Parameters
    ----------
    L_prev : int, optional
        Lag of the previous round. Without it the rule does not act.
    hold : bool
        Apply the rule (default true).
    return_nll : bool
        Return ``(record, nll)`` with the array of per-trajectory scores, shape
        (N, L_max + 1).

    Returns
    -------
    dict
        JSON-friendly. ``L_star`` (the lag to use), ``L_star_raw``, ``held``,
        ``usable``, ``L_tol``, ``L_cv``, ``fit_status``,
        ``n_prefix``, ``n_prefix_stage0``, ``L0``, ``resolution_failure``,
        ``grid_limited``, ``at_L_max``, the flags of ``grid_limits``,
        ``D_hat``, ``SE_D``, ``H_hat``, ``SE_H``, ``SE_Hdiff``, ``n_scored``,
        ``damaged_fraction``, ``n_damaged``, ``damaged_nats``, ``damaged_warning`` (diagnostic of
        :func:`inference.lag_selection.damage_diagnostic`, it does not change the lag),
        ``seconds_reselect``.
    """
    t0 = time.perf_counter()
    cond_params, emissions = jnp.asarray(cond_params), jnp.asarray(emissions)
    K_star = ls.kl_tolerance(T, eta=eta)
    freeze_batch_stats(model, reference_rows(key, jnp.asarray(ref_cond_params), jnp.asarray(ref_emissions),
                                             L_max, n_ref))
    nll = heldout_nll_all_lags(model, cond_params, emissions, L_max, L_cap)
    finite = bool(np.all(np.isfinite(nll)))
    safe = nll if finite else np.nan_to_num(nll, nan=1e6, posinf=1e6, neginf=-1e6)   # the rule below holds the lag
    sel = ls.select_lag(safe, T, K_star)
    st0 = ls.stage0_lag(np.asarray(emissions), T, K_star, square=square)
    flags = ls.grid_limits(sel, L_max)
    usable = bool(sel['fit_status'] != 'none' and finite)
    held = bool(hold and L_prev is not None and not usable)
    L_used = int(L_prev) if held else sel['L_star']
    out = {
        'L_star': L_used, 'L_star_raw': sel['L_star'], 'held': held, 'usable': usable,
        **ls.damage_diagnostic(nll),
        'L_tol': sel['L_tol'], 'L_cv': sel['L_cv'], 'L0': st0['L0'],
        'C_hat': sel['C'], 'rho_hat': sel['rho'], 'fit_status': sel['fit_status'],
        'n_prefix': sel['n_prefix'], 'n_prefix_stage0': st0['n_prefix'],
        'resolution_failure': ls.resolution_failure(st0['n_prefix'], sel['n_prefix']),
        'grid_limited': flags['limited'], 'grid_flags': flags,
        'at_L_max': bool(L_used >= L_max), 'limited_by': sel['limited_by'],
        'tol_capped': sel['tol_capped'], 'L_min': sel['L_min'],
        'D_hat': sel['D'], 'SE_D': sel['SE_D'], 'H_hat': sel['H'], 'SE_H': sel['SE_H'],
        'SE_Hdiff': sel['SE_Hdiff'], 'n_scored': int(emissions.shape[0]),
        'seconds_reselect': time.perf_counter() - t0,
    }
    return (_jsonable(out), nll.astype(np.float32)) if return_nll else _jsonable(out)


# ---------------------------------------------------------------------------
# Option A entry point: stage 0 + stage 1 on the round-0 simulations
# ---------------------------------------------------------------------------

def _jsonable(x):
    """Recursively convert arrays and NaN to JSON-friendly values (NaN becomes null)."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


def select_lag_round0(key, cond_params, emissions, T, hp, eta=0.1, holdout_frac=0.15, square=False,
                      L_cap=None, logger=print, ld_epochs_factor=None, return_nll=False):
    r"""
    Select the lag from the round-0 simulations. This is Option A (``lag: auto``),
    and round 0 of Option B (``lag: ada``).

    1. **Stage 0** (:func:`inference.lag_selection.stage0_lag`) on the round-0
       emissions as the pilot batch gives :math:`L_0` and the first grid
       :math:`L_{\rm grid}=\min(\max(2L_0,2),L_{\rm cap})`.
    2. **Candidate** (:func:`score_dropout`): one lag-dropout network of lag
       :math:`L_{\rm grid}`, scored on the held-out trajectories over the
       scoring window :math:`t>L_{\rm cap}`.
    3. **Stage 1** (:func:`inference.lag_selection.select_lag`).
    4. **Grid growth** (:func:`inference.lag_selection.grow_grid`): if the grid
       limits the result, :math:`L_{\rm grid}` doubles (at most
       :math:`L_{\rm cap}`), the network is trained again at the new shape, and
       stage 1 runs again.
    5. **Flag** (:func:`inference.lag_selection.resolution_failure`): true if
       stage 0 sees at least three significant lags and stage 1 sees fewer.

    The candidate network is discarded. The caller builds the T-SNL network.

    Parameters
    ----------
    key : PRNGKey
    cond_params : array (N, p), emissions : array (N, T, d)
        Round-0 data as ``util.train.get_sds`` returns it.
    T : int
        Series length.
    hp : dict
        Candidate hyperparameters, see :func:`hp_from_model_desc`.
    eta : float
        Total-variation tolerance, :math:`K^\star=2\eta^2`.
    ld_epochs_factor : float, optional
        Training budget of the network as a multiple of ``hp['epochs']``.
        Default :data:`LD_EPOCHS_FACTOR`.
    holdout_frac : float
        Fraction of trajectories held out.
    square : bool
        Apply the stage-0 proxy to :math:`y_t^2` (volatility models).
    L_cap : int, optional
        The largest lag that the method can examine. Default
        :math:`\lfloor T/4\rfloor`.
    logger : callable
        Receives progress lines.
    return_nll : bool
        Return ``(record, nll)`` with the held-out scores of the final grid,
        shape (n_heldout, L_grid + 1).

    Returns
    -------
    dict
        JSON-friendly. Main entries: ``L_star``, ``L_tol``, ``L_cv``, ``L0``,
        ``L_cap``, ``L_grid`` (final), ``L_grid_start``, ``grid_history``,
        ``grid_capped``, ``resolution_failure``, ``n_prefix``, ``damaged_fraction`` (and
        ``n_damaged``, ``damaged_warning``),
        ``C_hat``, ``rho_hat``, ``K_star``, ``fit_status``, ``D_hat``, ``SE_D``,
        ``H_hat``, ``SE_H``, ``SE_Hdiff``, ``seconds``.
    """
    ld_epochs_factor = LD_EPOCHS_FACTOR if ld_epochs_factor is None else ld_epochs_factor

    t0 = time.perf_counter()
    N = int(emissions.shape[0])
    L_cap = int(T // 4 if L_cap is None else L_cap)
    if L_cap < 2:
        raise ValueError(f'series too short for lag selection: T={T}, L_cap={L_cap} (need at least 2)')
    K_star = ls.kl_tolerance(T, eta=eta)
    st0 = ls.stage0_lag(np.asarray(emissions), T, K_star, square=square)
    L_grid0 = int(min(st0['L_grid'], L_cap))
    t_stage0 = time.perf_counter() - t0
    logger(f"lag selection: stage 0 L0={st0['L0']} ({st0['fit_status']}), first grid 0..{L_grid0}, "
           f"L_cap={L_cap}, K*={K_star:.4g}\n")

    key, ksplit, kscore = jr.split(key, 3)
    split = split_trajectories(ksplit, N, holdout_frac)
    seconds = {'train': 0.0, 'score': 0.0}
    n_networks = [0]

    def score_fn(L_grid):
        res = score_dropout(kscore, cond_params, emissions, L_cap, L_grid, hp, holdout_frac,
                            train_windows='all', epochs_factor=ld_epochs_factor, logger=logger,
                            split=split)
        n_networks[0] += 1
        seconds['train'] += float(np.sum(res['seconds_train']))
        seconds['score'] += float(np.sum(res['seconds_score']))
        return res['nll']

    grown = ls.grow_grid(score_fn, L_grid0, L_cap, T, K_star)
    sel, L_grid = grown['sel'], grown['L_grid']
    for h in grown['history']:
        logger(f"lag selection: grid 0..{h['L_grid']}: L*={h['L_star']} (L_tol={h['L_tol']}, "
               f"L_cv={h['L_cv']}), grid limits the result: {h['limited']}\n")
    logger(f"lag selection: L*={sel['L_star']} (L_tol={sel['L_tol']}, L_cv={sel['L_cv']}, "
           f"fit {sel['fit_status']}, limited by {sel['limited_by']}"
           f"{', grid at its cap' if grown['grid_capped'] else ''})\n")
    failure = ls.resolution_failure(st0['n_prefix'], sel['n_prefix'])
    if failure:
        logger(f"WARNING lag selection: resolution failure. Stage 0 sees {st0['n_prefix']} significant lags, "
               f"stage 1 sees {sel['n_prefix']}. L*={sel['L_star']} is a lower bound. More data or a "
               f"longer training are necessary.\n")

    damage = ls.damage_diagnostic(grown['nll'])
    if damage['damaged_warning']:
        logger(f"WARNING lag selection: {damage['n_damaged']} of {grown['nll'].shape[0]} held-out trajectories "
               f"({100 * damage['damaged_fraction']:.0f} %) have a damaged score (a lag is worse than lag 0 by more "
               f"than {damage['damaged_nats']:g} nat per observation, or not finite). The network does not describe "
               f"them. The lag is not changed by this warning.\n")

    n_held = int(grown['nll'].shape[0])
    out = {
        'option': 'A', 'L_star': sel['L_star'], 'L_final': sel['L_star'],
        'L_tol': sel['L_tol'], 'L_cv': sel['L_cv'], 'L0': st0['L0'], 'L_cap': L_cap,
        'L_grid': L_grid, 'L_grid_start': L_grid0, 'grid_history': grown['history'],
        'grid_capped': grown['grid_capped'], 'C_hat': sel['C'], 'rho_hat': sel['rho'],
        'K_star': K_star, 'eta': eta, 'T': int(T), 'fit_status': sel['fit_status'],
        'n_prefix': sel['n_prefix'], 'n_prefix_stage0': st0['n_prefix'], 'resolution_failure': failure,
        **damage,
        'tol_capped': sel['tol_capped'],
        'limited_by': sel['limited_by'], 'L_min': sel['L_min'],
        'D_hat': sel['D'], 'SE_D': sel['SE_D'], 'H_hat': sel['H'], 'SE_H': sel['SE_H'],
        'SE_Hdiff': sel['SE_Hdiff'], 'fit_mask': sel['mask'],
        'stage0': {k: st0[k] for k in ('L0', 'L_grid', 'C', 'rho', 'D_G', 'n_prefix',
                                       'fit_status', 'noise_floor')},
        'n_trajectories': N, 'n_heldout': n_held, 'n_train_trajectories': N - n_held,
        'hp': hp, 'holdout_frac': holdout_frac, 'n_networks': n_networks[0],
        'ld_epochs_factor': ld_epochs_factor,
        'seconds': {'stage0': t_stage0, 'candidates_train': seconds['train'],
                    'candidates_score': seconds['score'],
                    'total': time.perf_counter() - t0},
    }
    return (_jsonable(out), grown['nll'].astype(np.float32)) if return_nll else _jsonable(out)


def save_selection(sel, exp_dir, scores=None):
    r"""
    Write ``lag_selection.json`` and ``lag_selection.png`` into ``exp_dir``. If ``scores`` is a
    dict ``{round: array}`` of per-trajectory held-out scores, also write
    ``lag_selection_nll.npz`` (arrays ``round0``, ``round1``, ..., float32), so that any rule
    can be tested later on the same scores without training again.

    The figure has two panels: the decrements :math:`\hat D(L)\pm2\,\mathrm{SE}`
    with the fitted line :math:`\hat C\hat\rho^L` and the chosen
    :math:`L^\star`, and :math:`\hat H(L)-\hat H(L_{\min})` with its paired SE.
    These are the round-0 selection. For Option B (``sel['option'] == 'B'``) a
    third panel shows :math:`L^\star_r` for each round with the cap
    :math:`L_{\max}`. The title of the first panel carries the warning if
    ``resolution_failure`` is true.
    """
    import json
    import os

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(os.path.join(exp_dir, 'lag_selection.json'), 'w') as f:
        json.dump(sel, f, indent=1)
    if scores:
        np.savez(os.path.join(exp_dir, 'lag_selection_nll.npz'),
                 **{f'round{r}': np.asarray(v, dtype=np.float32) for r, v in scores.items()})

    D = np.array([np.nan if v is None else v for v in sel['D_hat']])
    se = np.array([np.nan if v is None else v for v in sel['SE_D']])
    H = np.array(sel['H_hat'], dtype=float)
    sed = np.array([np.nan if v is None else v for v in sel['SE_Hdiff']])
    L = np.arange(len(D))
    is_b = sel.get('option') == 'B'
    fig, ax = plt.subplots(1, 3 if is_b else 2, figsize=(14 if is_b else 9.5, 3.6))
    ax[0].errorbar(L, D, 2 * se, fmt='o', ms=3.5, color='tab:green', capsize=2,
                   label=r'$\hat D(L)\pm 2\,$SE')
    if sel['C_hat'] is not None and sel['rho_hat'] is not None:
        ax[0].plot(L, sel['C_hat'] * sel['rho_hat'] ** L, 'k-', lw=1,
                   label=rf"$\hat C\hat\rho^L$, $\hat\rho={sel['rho_hat']:.2f}$")
    ax[0].axvline(sel['L_star'], color='tab:red', ls='--', lw=1, label=rf"$L^\star={sel['L_star']}$")
    ax[0].set_yscale('symlog', linthresh=1e-3)
    ax[0].set_xlabel('lag $L$')
    title = f"decrements ({sel['fit_status']} fit, {sel['limited_by']}-limited)"
    if sel.get('resolution_failure'):
        title += '\nRESOLUTION FAILURE: $L^\\star$ is a lower bound'
    if sel.get('damaged_warning'):
        title += f"\nDAMAGED SCORES: {100 * sel['damaged_fraction']:.0f} % of the trajectories"
    ax[0].set_title(title, color='tab:red' if sel.get('resolution_failure') or sel.get('damaged_warning') else 'k', fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].errorbar(np.arange(len(H)), H - H.min(), sed, fmt='o', ms=3.5, color='tab:blue', capsize=2)
    ax[1].axvline(sel['L_star'], color='tab:red', ls='--', lw=1)
    ax[1].set_xlabel('lag $L$')
    ax[1].set_title(r'$\hat H(L)-\min\hat H$, paired SE')
    if is_b:
        r = np.arange(len(sel['L_star_r']))
        ax[2].plot(r, sel['L_star_r'], 'o-', color='tab:purple', label=r'$L^\star_r$')
        ax[2].axhline(sel['L_max'], color='k', ls=':', lw=1, label=rf"$L_{{\max}}={sel['L_max']}$")
        bad = [i for i, rd in enumerate(sel['rounds']) if rd.get('resolution_failure')]
        if bad:
            ax[2].plot(bad, [sel['L_star_r'][i] for i in bad], 'rx', ms=9, label='resolution failure')
        dam = [i for i, rd in enumerate(sel['rounds']) if rd.get('damaged_warning')]
        if dam:
            ax[2].plot(dam, [sel['L_star_r'][i] for i in dam], 's', mfc='none', mec='tab:orange', ms=11, label='damaged scores')
        held = [i for i, rd in enumerate(sel['rounds']) if rd.get('held')]
        if held:
            ax[2].plot(held, [sel['L_star_r'][i] for i in held], 'k+', ms=11, label='lag held')
        ax[2].set_xlabel('round $r$ (0-based)')
        ax[2].set_xticks(r)
        ax[2].set_title(r'lag used in each round')
        ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, 'lag_selection.png'), dpi=140)
    plt.close(fig)
