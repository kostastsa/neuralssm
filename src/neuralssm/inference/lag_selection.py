r"""
Automatic lag selection for T-SNL (request I1).

The functions in this module are pure: they take arrays and return arrays or
dictionaries, and they use no neural network. The networks only produce the
held-out negative log-likelihood array that :func:`select_lag` consumes.

Notation
--------
``T``        series length.
``d``        observation dimension.
``L``        lag, so the conditional of :math:`y_t` uses :math:`y_{t-L:t-1}`.
``H(L)``     expected conditional NLL (nats per step) given ``L`` lags,

.. math:: H(L) = \mathbb{E}\bigl[-\log p(y_t \mid y_{t-L:t-1})\bigr].

``D(L)``     decrement of ``H``,

.. math:: D(L) = H(L) - H(L+1) \ge 0.

``K*``       tolerance on the total KL divergence between the SSM law of
             :math:`y_{1:T}` and its order-``L`` Markov approximation. For a
             total-variation tolerance :math:`\eta`, Pinsker's inequality gives
             :math:`K^\star = 2\eta^2`. For a per-step tolerance
             :math:`\varepsilon`, :math:`K^\star = T\varepsilon`.

Working model
-------------
The decrements are modelled as geometric,

.. math:: D(L) \approx C\,\rho^{L},

where :math:`(C,\rho)` are fitted from data. This is a working model and not a
certificate (assumption (A1) of Proposition 1 fails for the paper's models).
Under this model the truncation error summed over ``T`` steps is

.. math:: K_{T,L} \le T\sum_{m \ge L} D(m) = \frac{T\,C\,\rho^{L}}{1-\rho},

and the tolerance lag is the smallest ``L`` with
:math:`T C \rho^{L}/(1-\rho) \le K^\star`.

Spec: ``requests/I1_attachments/lag_selection_guidance.md``, Sec. 3.3 and 4.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

def kl_tolerance(T, eta=None, eps=None):
    r"""
    KL tolerance :math:`K^\star` from a TV level or a per-step level.

    .. math::
        K^\star = 2\eta^2 \quad\text{(TV level } \eta\text{)}, \qquad
        K^\star = T\varepsilon \quad\text{(per-step level } \varepsilon\text{)}.

    The first form follows from Pinsker's inequality
    :math:`\|P-Q\|_{TV} \le \sqrt{\mathrm{KL}(P\|Q)/2}`.

    Parameters
    ----------
    T : int
        Series length. Used only with ``eps``.
    eta : float, optional
        Total-variation tolerance. Default 0.1 when neither argument is given.
    eps : float, optional
        Per-step KL tolerance. Mutually exclusive with ``eta``.

    Returns
    -------
    float
        :math:`K^\star` in nats.
    """
    if eta is not None and eps is not None:
        raise ValueError("give eta or eps, not both")
    if eps is not None:
        return float(T * eps)
    if eta is None:
        eta = 0.1
    return float(2.0 * eta ** 2)


# ---------------------------------------------------------------------------
# Stage 0: Gaussian proxy
# ---------------------------------------------------------------------------

def sample_autocovariances(y, max_lag):
    r"""
    Pooled sample autocovariances of one or more series.

    .. math::
        \hat\Gamma(h) = \frac{1}{N\,T}\sum_{n=1}^{N}\sum_{t=h+1}^{T}
        (y^{(n)}_t-\bar y)(y^{(n)}_{t-h}-\bar y)^{\top},
        \qquad h = 0,\dots,\text{max\_lag},

    with :math:`\bar y` the mean over all series and times. The divisor
    :math:`N T` (and not :math:`N(T-h)`) keeps the block Toeplitz matrix
    positive semi-definite, so the Yule-Walker system is well posed.

    Parameters
    ----------
    y : array, shape (T, d), (T,) or (N, T, d)
        Observations. A 1-D or 2-D array is one series.
    max_lag : int
        Largest lag ``h``.

    Returns
    -------
    array, shape (max_lag + 1, d, d)
        ``G[h]`` is :math:`\hat\Gamma(h)=\mathbb{E}[y_t y_{t-h}^\top]`.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[None, :, None]
    elif y.ndim == 2:
        y = y[None]
    N, T, d = y.shape
    if max_lag >= T:
        raise ValueError("max_lag must be smaller than the series length")
    y = y - y.reshape(-1, d).mean(axis=0)
    G = np.zeros((max_lag + 1, d, d))
    for h in range(max_lag + 1):
        G[h] = np.einsum('nti,ntj->ij', y[:, h:], y[:, :T - h]) / (N * T)
    return G


def innovation_covariances(G, order_max):
    r"""
    Innovation covariances :math:`V_k` of the best linear predictor of order
    :math:`k = 0,\dots,\text{order\_max}`.

    The order-``k`` predictor is :math:`\hat y_t=\sum_{j=1}^{k}A_j y_{t-j}`. The
    Yule-Walker equations are

    .. math::
        \Gamma(i) = \sum_{j=1}^{k} A_j\,\Gamma(i-j), \quad i=1,\dots,k,
        \qquad \Gamma(-h)=\Gamma(h)^{\top},

    that is :math:`[A_1,\dots,A_k]\,M_k=[\Gamma(1),\dots,\Gamma(k)]` with the
    block Toeplitz matrix :math:`(M_k)_{ji}=\Gamma(i-j)`. The innovation
    covariance is

    .. math:: V_k = \Gamma(0) - \sum_{j=1}^{k} A_j\,\Gamma(j)^{\top},
              \qquad V_0 = \Gamma(0).

    The system is solved directly for each ``k`` (cost :math:`O(k^3 d^3)`),
    which gives the same :math:`V_k` as Whittle's recursion
    (Brockwell and Davis 1991, Sec. 11.4). The orders in this application are
    small.

    Parameters
    ----------
    G : array, shape (order_max + 1, d, d)
        Autocovariances from :func:`sample_autocovariances`.
    order_max : int
        Largest order ``k``.

    Returns
    -------
    array, shape (order_max + 1, d, d)
    """
    d = G.shape[1]

    def gamma(h):
        return G[h] if h >= 0 else G[-h].T

    V = np.zeros((order_max + 1, d, d))
    V[0] = G[0]
    for k in range(1, order_max + 1):
        M = np.block([[gamma(i - j) for i in range(1, k + 1)]
                      for j in range(1, k + 1)])
        R = np.concatenate([G[i] for i in range(1, k + 1)], axis=1)
        try:
            A = np.linalg.solve(M.T, R.T).T      # R @ inv(M)
        except np.linalg.LinAlgError:            # near-singular at high order
            A = np.linalg.lstsq(M.T, R.T, rcond=None)[0].T
        V[k] = G[0] - sum(A[:, (j - 1) * d:j * d] @ G[j].T
                          for j in range(1, k + 1))
    return V


def gaussian_proxy_decrements(y, L_proxy, square=False):
    r"""
    Stage 0: Gaussian proxy :math:`D_G(L)` from the innovation covariances.

    .. math::
        D_G(L) = \tfrac12\log\frac{\det V_L}{\det V_{L+1}},
        \qquad L = 0,\dots,L_{\rm proxy}.

    For a scalar series :math:`V_{L+1}=V_L(1-\phi_{L+1}^2)` with
    :math:`\phi_{L+1}` the partial autocorrelation at lag :math:`L+1`, so

    .. math:: D_G(L) = -\tfrac12\log\bigl(1-\phi_{L+1}^2\bigr).

    This is the Gaussian entropy decrement
    :math:`H(L)-H(L+1)` with :math:`H(L)=\tfrac12\log(2\pi e\,V_L)` and is
    exact for Gaussian processes (guidance Sec. 3.3). For non-Gaussian
    models it is only a linear proxy. For volatility-type models (SVSSM) the
    series is nearly uncorrelated and the proxy is close to zero. Then use
    ``square=True`` to apply it to :math:`y_t^2`.

    Parameters
    ----------
    y : array, shape (T, d), (T,) or (N, T, d)
        Observed series or a pilot batch of simulated series.
    L_proxy : int
        Largest lag of the proxy. The fit uses orders up to ``L_proxy + 1``.
    square : bool
        Apply the proxy to :math:`y_t^2` (element-wise).

    Returns
    -------
    array, shape (L_proxy + 1,)
        :math:`D_G(0),\dots,D_G(L_{\rm proxy})`. Entries below zero (estimation
        noise, possible when the sample covariance is nearly singular) are
        kept as they are: the caller decides the noise level.
    """
    y = np.asarray(y, dtype=float)
    if square:
        y = y ** 2
    G = sample_autocovariances(y, L_proxy + 1)
    V = innovation_covariances(G, L_proxy + 1)
    logdet = np.array([np.linalg.slogdet(v)[1] for v in V])
    return 0.5 * (logdet[:-1] - logdet[1:])


# ---------------------------------------------------------------------------
# Forgetting curve and lag rules
# ---------------------------------------------------------------------------

def fit_forgetting_curve(D, SE=None, noise_floor=0.0, n_sigma=2.0, prefix=True,
                         min_points=3):
    r"""
    Fit the geometric working model :math:`\log D(L)=\log C+L\log\rho`.

    A lag is *significant* if

    .. math:: D(L) > n_\sigma\,\mathrm{SE}(L) \quad\text{and}\quad D(L) > \text{noise\_floor}.

    ``SE=None`` means no standard errors (stage 0): only ``noise_floor``
    applies. The fit is ordinary least squares of :math:`\log D` on :math:`L`
    over the lags in the fitted set. The slope is :math:`\log\hat\rho` and the
    intercept :math:`\log\hat C`.

    **Prefix rule** (``prefix=True``, default). The fitted set is the first run
    of consecutive significant lags starting at :math:`L=0`,

    .. math:: \{0,\dots,k-1\},\qquad k=\min\{L:\ L\text{ not significant}\}.

    An isolated significant lag further out is noise, and one such lag can
    flatten the regression line (seen on the LGSSM: lag 11 in a run where the
    exact decrement is :math:`<10^{-5}`). This is a deviation from the note,
    which regresses over all significant lags (``prefix=False``).

    A fit needs at least ``min_points`` lags. With fewer, the decay rate is
    not resolved and ``rho`` is ``nan``.

    Parameters
    ----------
    D : array, shape (n,)
        Decrements :math:`D(0),\dots,D(n-1)`.
    SE : array, shape (n,), optional
        Paired standard errors of ``D``.
    noise_floor : float
        Smallest value of ``D`` that counts as signal.
    n_sigma : float
        Number of standard errors (default 2).
    prefix : bool
        Use the first run of consecutive significant lags from :math:`L=0`.
    min_points : int
        Minimum number of lags for a fit (default 3).

    Returns
    -------
    dict
        ``C``, ``rho`` (``nan`` if the fit is not possible or if the slope is
        not negative, so :math:`\hat\rho\ge1`), ``mask`` (boolean array of the
        lags used), ``n_used`` and ``n_prefix`` (the length :math:`k` of the
        run of consecutive significant lags from 0).
    """
    D = np.asarray(D, dtype=float)
    sig = D > noise_floor
    if SE is not None:
        sig &= D > n_sigma * np.asarray(SE, dtype=float)
    k = int(np.argmin(sig)) if not sig.all() else len(sig)
    if prefix:
        mask = np.zeros_like(sig)
        mask[:k] = True
    else:
        mask = sig
    out = {'C': np.nan, 'rho': np.nan, 'mask': mask, 'n_used': int(mask.sum()),
           'n_prefix': k}
    if mask.sum() < min_points:
        return out
    L = np.arange(len(D))[mask]
    slope, intercept = np.polyfit(L, np.log(D[mask]), 1)
    if slope >= 0:
        return out
    out['C'], out['rho'] = float(np.exp(intercept)), float(np.exp(slope))
    return out


def L_from_rate(C, rho, T, K_star):
    r"""
    Tolerance lag from a fitted geometric curve.

    .. math::
        L_{\rm tol}=\min\Bigl\{L\ge0:\;\frac{T\,C\,\rho^{L}}{1-\rho}\le K^\star\Bigr\}
        =\max\Bigl(0,\Bigl\lceil
        \frac{\log\bigl(T C/((1-\rho)K^\star)\bigr)}{\log(1/\rho)}
        \Bigr\rceil\Bigr).

    Parameters
    ----------
    C, rho : float
        Fitted constants, :math:`0<\rho<1`.
    T : int
        Series length.
    K_star : float
        Tolerance, see :func:`kl_tolerance`.

    Returns
    -------
    int
        :math:`L_{\rm tol}`. ``numpy.inf`` if the curve is not valid
        (``rho`` is ``nan`` or not in (0, 1)).
    """
    if not (np.isfinite(rho) and 0.0 < rho < 1.0 and np.isfinite(C) and C > 0):
        return np.inf
    arg = T * C / ((1.0 - rho) * K_star)
    if arg <= 1.0:
        return 0
    return int(np.ceil(np.log(arg) / np.log(1.0 / rho)))


def L_from_rho(rho, kappa, T, delta):
    r"""
    A-priori lag of Proposition 1 for models where :math:`(\kappa,\rho)` are
    known (finite-state HMM, circular random walk test beds).

    .. math:: L=\min\Bigl\{L:\;\frac{T\,\kappa\,\rho^{L}}{1-\rho}\le\delta\Bigr\}.

    This is :func:`L_from_rate` with ``C=kappa`` and ``K_star=delta``. It is
    not used as a code path for the paper's models (assumption (A1) fails).
    """
    return L_from_rate(kappa, rho, T, delta)


def stage0_lag(y, T, K_star, noise_floor=None, square=False, L_proxy=60):
    r"""
    Stage 0: initial lag :math:`L_0` and first grid :math:`L_{\rm grid}=\max(2L_0,\,2)`.

    Compute :math:`D_G` on a lag range up to ``L_cap``, fit the geometric
    curve with :func:`fit_forgetting_curve`, and set :math:`L_0` with
    :func:`L_from_rate`. The caller limits the grid to :math:`L_{\rm cap}`.

    Parameters
    ----------
    y : array
        Series for :func:`gaussian_proxy_decrements`.
    T : int
        Series length used in the tolerance rule.
    K_star : float
        Tolerance.
    noise_floor : float, optional
        Smallest :math:`D_G` that counts as signal. Default
        :math:`3d^2/n` with :math:`n` the number of observations used
        (series times length). Under no dependence, the estimated decrement at
        one lag is about :math:`d^2/(2n)` on average.
    square : bool
        Apply the proxy to :math:`y_t^2`.
    L_proxy : int
        Largest lag of the proxy (clipped to ``len(y) // 4``). High orders make
        the Yule-Walker system near singular, so keep it moderate.

    Returns
    -------
    dict
        ``L0``, ``L_grid``, ``C``, ``rho``, ``D_G`` (array), ``n_used``,
        ``n_prefix`` (run of consecutive lags above the noise floor from lag
        0), ``fit_status`` and ``noise_floor``. As in :func:`select_lag`:
        ``'none'`` (nothing above the floor, ``L0 = 0``), ``'unresolved'``
        (fewer than three lags, ``L0`` = the run length), ``'ok'``, and
        ``'invalid'`` (non-negative slope: ``L0 = L_proxy``, conservative).
    """
    y = np.asarray(y)
    n_time = y.shape[-2] if y.ndim == 3 else y.shape[0]
    L_proxy = int(min(L_proxy, max(n_time // 4, 1)))
    D = gaussian_proxy_decrements(y, L_proxy, square=square)
    if noise_floor is None:
        d = y.shape[-1] if y.ndim >= 2 else 1
        noise_floor = 3.0 * d ** 2 / (y.size / d)
    fit = fit_forgetting_curve(D, noise_floor=noise_floor)
    k = fit['n_prefix']
    if k == 0:
        status, L0 = 'none', 0
    elif k < 3:
        status, L0 = 'unresolved', k
    else:
        L0 = L_from_rate(fit['C'], fit['rho'], T, K_star)
        status = 'ok' if np.isfinite(fit['rho']) else 'invalid'
        if not np.isfinite(L0):
            L0 = L_proxy
    L0 = int(min(L0, L_proxy))
    return {'L0': L0, 'L_grid': max(2 * L0, 2), 'C': fit['C'], 'rho': fit['rho'],
            'D_G': D, 'n_used': fit['n_used'], 'n_prefix': k, 'fit_status': status,
            'noise_floor': float(noise_floor)}


# ---------------------------------------------------------------------------
# Stage 1: held-out selection
# ---------------------------------------------------------------------------

def decrements_from_nll(nll):
    r"""
    Held-out NLL curve, decrements and paired standard errors.

    ``nll[n, L]`` is the mean NLL per step of held-out trajectory ``n`` at lag
    ``L``, computed on the *same* targets
    :math:`t\in\{L_{\rm cap}+1,\dots,T\}` for every ``L``. With ``N``
    trajectories:

    .. math::
        \hat H(L)=\frac1N\sum_n \mathrm{nll}_{nL},\qquad
        \hat D(L)=\hat H(L)-\hat H(L+1),

    .. math::
        \mathrm{SE}_D(L)=\frac{1}{\sqrt N}\,
        \mathrm{sd}_n\bigl(\mathrm{nll}_{nL}-\mathrm{nll}_{n,L+1}\bigr),
        \qquad
        \mathrm{SE}_H(L)=\frac{1}{\sqrt N}\,\mathrm{sd}_n(\mathrm{nll}_{nL}).

    The decrement uses the pairing over trajectories (same data, two lags),
    which removes the large between-trajectory variance. ``sd`` is the sample
    standard deviation with ``ddof=1``.

    Parameters
    ----------
    nll : array, shape (N, L_grid + 1)

    Returns
    -------
    dict
        ``H`` and ``SE_H`` (length ``L_grid + 1``), ``D`` and ``SE_D``
        (length ``L_grid``).
    """
    nll = np.asarray(nll, dtype=float)
    N = nll.shape[0]
    if N < 2:
        raise ValueError("need at least two held-out trajectories")
    diff = nll[:, :-1] - nll[:, 1:]
    return {
        'H': nll.mean(axis=0),
        'SE_H': nll.std(axis=0, ddof=1) / np.sqrt(N),
        'D': diff.mean(axis=0),
        'SE_D': diff.std(axis=0, ddof=1) / np.sqrt(N),
    }


def paired_se_to_min(nll):
    r"""
    Paired standard error of :math:`\hat H(L)-\hat H(L_{\min})` for every lag.

    Returns ``(j, se)`` with :math:`j=L_{\min}=\arg\min_L\hat H(L)` and

    .. math:: \mathrm{SE}_\Delta(L)=\frac{1}{\sqrt N}\,
        \mathrm{sd}_n\bigl(\mathrm{nll}_{nL}-\mathrm{nll}_{nL_{\min}}\bigr)

    (``ddof=1``; zero at :math:`L_{\min}`).
    """
    nll = np.asarray(nll, dtype=float)
    j = int(np.argmin(nll.mean(axis=0)))
    se = (nll - nll[:, [j]]).std(axis=0, ddof=1) / np.sqrt(nll.shape[0])
    return j, se


def one_se_lag(nll, paired=True):
    r"""
    One-standard-error rule (Hastie et al. 2009, Sec. 7.10).

    With :math:`L_{\min}=\arg\min_L \hat H(L)`, the *paired* rule is

    .. math::
        L_{\rm cv}=\min\Bigl\{L:\ \hat H(L)-\hat H(L_{\min})
        \le \mathrm{SE}_{\Delta}(L)\Bigr\},\qquad
        \mathrm{SE}_{\Delta}(L)=\frac{1}{\sqrt N}\,
        \mathrm{sd}_n\bigl(\mathrm{nll}_{nL}-\mathrm{nll}_{nL_{\min}}\bigr).

    The two lags are scored on the same trajectories, so the trajectory
    variance cancels in the difference, as for the decrements. The *unpaired*
    rule (``paired=False``) compares with the standard error of
    :math:`\hat H(L_{\min})` itself, which is dominated by the variation
    between trajectories. On the LGSSM it returned lags 2 and 1 where the
    paired rule returned 4 and 2 (exact tolerance lag: 4).

    Parameters
    ----------
    nll : array, shape (N, L_grid + 1)
        Per-trajectory mean held-out NLL.
    paired : bool
        Use the paired standard error (default).

    Returns
    -------
    int
        :math:`L_{\rm cv}`.
    """
    nll = np.asarray(nll, dtype=float)
    H = nll.mean(axis=0)
    j, se = paired_se_to_min(nll)
    if not paired:
        se = np.full(len(H), nll[:, j].std(ddof=1) / np.sqrt(nll.shape[0]))
    return int(np.flatnonzero(H - H[j] <= se)[0])


def select_lag(nll, T, K_star, n_sigma=2.0, paired_cv=True, prefix=True, min_points=3):
    r"""
    Stage 1: select :math:`L^\star` from the held-out NLL array.

    Steps (guidance Sec. 4, steps 3-5):

    1. :func:`decrements_from_nll` gives :math:`\hat H,\hat D,\mathrm{SE}`.
    2. :func:`fit_forgetting_curve` gives :math:`(\hat C,\hat\rho)` from the
       first run of consecutive lags with :math:`\hat D(L)>n_\sigma\mathrm{SE}_D(L)`.
    3. :math:`L_{\rm tol}` from :func:`L_from_rate` and :math:`L_{\rm cv}`
       from :func:`one_se_lag` (paired).
    4. .. math:: L^\star=\min(L_{\rm tol},\,L_{\rm cv}).

    Let :math:`k` be the length of the run of consecutive significant lags
    from 0. The value of :math:`L_{\rm tol}` and the entry ``fit_status``
    depend on :math:`k`:

    * ``'none'``, :math:`k=0`: even :math:`\hat D(0)` is not significant,
      so :math:`L_{\rm tol}=0`.
    * ``'unresolved'``, :math:`0<k<` ``min_points``: the decay rate cannot be
      fitted. :math:`L_{\rm tol}=k`, the first lag whose decrement is not
      resolved. The tolerance is not certified.
    * ``'ok'``: the fit gives :math:`L_{\rm tol}`. If it is larger than
      :math:`L_{\rm grid}`, it is clipped and ``tol_capped`` is true.
    * ``'invalid'``: the fitted slope is not negative, so
      :math:`L_{\rm tol}=L_{\rm grid}` and ``tol_capped`` is true.

    The first three rows and the prefix rule are choices of this
    implementation (not in the note). ``paired_cv=False`` and
    ``prefix=False`` give the note's wording.

    Parameters
    ----------
    nll : array, shape (N, L_grid + 1)
        Per-trajectory mean held-out NLL at lags ``0..L_grid``.
    T : int
        Series length.
    K_star : float
        Tolerance, see :func:`kl_tolerance`.
    n_sigma : float
        Number of standard errors for the significance test (default 2).
    paired_cv : bool
        Use the paired one-SE rule (default).
    prefix, min_points : see :func:`fit_forgetting_curve`.

    Returns
    -------
    dict
        ``L_star``, ``L_tol``, ``L_cv``, ``C``, ``rho``, ``K_star``,
        ``fit_status``, ``n_prefix``, ``tol_capped``, ``limited_by``
        (``'tol'`` or ``'cv'``), ``L_min`` (lag of the smallest :math:`\hat H`),
        and the arrays ``H``, ``SE_H``, ``D``, ``SE_D``, ``SE_Hdiff`` (paired SE
        of :math:`\hat H(L)-\hat H(L_{\min})`), ``mask`` (lags used in the fit).
    """
    out = decrements_from_nll(nll)
    L_grid = len(out['H']) - 1
    fit = fit_forgetting_curve(out['D'], out['SE_D'], n_sigma=n_sigma, prefix=prefix,
                               min_points=min_points)
    L_cv = one_se_lag(nll, paired=paired_cv)
    L_min, se_diff = paired_se_to_min(nll)

    k = fit['n_prefix'] if prefix else fit['n_used']
    if k == 0:
        status, L_tol = 'none', 0
    elif k < min_points:
        status, L_tol = 'unresolved', k
    else:
        L_tol = L_from_rate(fit['C'], fit['rho'], T, K_star)
        status = 'ok' if np.isfinite(fit['rho']) else 'invalid'
    tol_capped = (not np.isfinite(L_tol)) or L_tol > L_grid
    L_tol = L_grid if tol_capped else int(L_tol)

    L_star = min(L_tol, L_cv)
    out.update(L_star=int(L_star), L_tol=L_tol, L_cv=L_cv, C=fit['C'],
               rho=fit['rho'], K_star=float(K_star), fit_status=status,
               n_prefix=int(fit['n_prefix']), tol_capped=bool(tol_capped),
               L_min=L_min, SE_Hdiff=se_diff,
               limited_by='cv' if L_cv < L_tol else 'tol', mask=fit['mask'])
    return out


def select_lag_by_theta_bins(nll, bin_var, T, K_star, n_bins=4, quantile=1.0,
                             n_sigma=2.0):
    r"""
    Step 6: account for the dependence of the forgetting rate on
    :math:`\theta`.

    The held-out trajectories are sorted by the scalar ``bin_var`` (for
    example the component of :math:`\theta` that drives the forgetting, or a
    projection of it) and split into ``n_bins`` bins of equal size. The rule
    :func:`select_lag` runs in each bin, giving :math:`L^\star_b`. The result
    is a quantile of the per-bin lags, which mirrors the
    :math:`\sup_\theta` of Proposition 1:

    .. math:: L^\star=\bigl\lceil Q_q\bigl(L^\star_1,\dots,L^\star_B\bigr)\bigr\rceil,

    with ``quantile=1.0`` the maximum.

    Parameters
    ----------
    nll : array, shape (N, L_grid + 1)
    bin_var : array, shape (N,)
        One scalar per held-out trajectory.
    T, K_star, n_sigma : see :func:`select_lag`.
    n_bins : int
        Number of bins.
    quantile : float
        Quantile of the per-bin lags in [0, 1].

    Returns
    -------
    dict
        ``L_star``, ``L_bins`` (array of per-bin lags) and ``bins`` (list of
        the :func:`select_lag` results).
    """
    nll = np.asarray(nll, dtype=float)
    order = np.argsort(np.asarray(bin_var))
    groups = np.array_split(order, n_bins)
    bins = [select_lag(nll[g], T, K_star, n_sigma) for g in groups]
    L_bins = np.array([b['L_star'] for b in bins])
    return {'L_star': int(np.ceil(np.quantile(L_bins, quantile))),
            'L_bins': L_bins, 'bins': bins}


# ---------------------------------------------------------------------------
# Resolution failure and the cap of Option B
# ---------------------------------------------------------------------------

def resolution_failure(n_stage0, n_stage1, min_points=3):
    r"""
    Warning flag: stage 1 cannot resolve a dependence that stage 0 sees.

    Stage 0 (the Gaussian proxy) and stage 1 (the trained networks) count the
    run of consecutive lags, from lag 0, whose decrement is above the noise
    (``n_prefix``). The flag is true if stage 0 has at least ``min_points``
    such lags and stage 1 has fewer. Then the data show a dependence that the
    held-out decrements of the networks cannot resolve one lag at a time. The
    fit of stage 1 is missing or unresolved, and :math:`L^\star` is a lower
    bound. More data (or a longer training) are necessary.

    The default ``min_points = 3`` is the number of points that the fit of
    :func:`select_lag` needs.

    Parameters
    ----------
    n_stage0, n_stage1 : int
        ``n_prefix`` of :func:`stage0_lag` and of :func:`select_lag`.
    min_points : int

    Returns
    -------
    bool
    """
    return bool(n_stage0 >= min_points and n_stage1 < min_points)


def ada_lag_max(L_star0, L_cap):
    r"""
    Cap :math:`L_{\max}` of Option B: the lag of the network, fixed for the whole
    run. It is :math:`\min(\max(2L^\star_0,\,2),\,L_{\rm cap})`, with
    :math:`L^\star_0` the lag selected on the round-0 simulations. The factor 2
    is the margin of the plan (``lag_selection_plan.md``, Sec. 3 step 1) and
    the floor 2 is the smallest grid of :func:`grow_grid`.
    """
    return int(min(max(2 * int(L_star0), 2), int(L_cap)))


# A trajectory is damaged if one of its lags scores worse than its own lag 0 by more than this
# many nats per observation, or if a score is not finite. At 1 nat, at most 0.7 % of the clean
# trajectories of the d4 LGSSM are flagged in any of the 24 clean sets that I checked (150 to 500
# trajectories each), and 29 % of the trajectories of the failed round of the first Option B run.
DAMAGE_NATS = 1.0

# The diagnostic warns if more than this fraction of the trajectories is damaged.
DAMAGE_WARN = 0.05


def damaged_trajectories(nll, nats=DAMAGE_NATS):
    r"""
    Which held-out trajectories have a damaged score?

    Trajectory :math:`n` is damaged if a score is not finite, or if

    .. math:: \max_{L\ge1}\bigl(\mathrm{nll}_{nL}-\mathrm{nll}_{n0}\bigr) > \text{nats}.

    The network at lag 0 is a valid model of the same trajectory, so a lag that is worse
    by more than ``nats`` per observation shows that the network does not describe the
    trajectory (for example a simulation at an extreme parameter, outside the data that
    the network has seen). The test cannot see damage that is the same at every lag.

    Parameters
    ----------
    nll : array, shape (N, L_grid + 1)
    nats : float

    Returns
    -------
    bool array, shape (N,)
    """
    nll = np.asarray(nll, dtype=float)
    with np.errstate(invalid='ignore'):
        excess = (nll[:, 1:] - nll[:, [0]]).max(axis=1)
    return ~np.all(np.isfinite(nll), axis=1) | (excess > nats)


def damage_diagnostic(nll, nats=DAMAGE_NATS, warn=DAMAGE_WARN):
    r"""
    Diagnostic of a selection (decision E13). It never changes the lag.

    Returns
    -------
    dict
        ``damaged_fraction``, ``n_damaged``, ``damaged_nats`` (the threshold) and
        ``damaged_warning`` (true if the fraction is above ``warn``).
    """
    mask = damaged_trajectories(nll, nats)
    frac = float(mask.mean())
    return {'damaged_fraction': frac, 'n_damaged': int(mask.sum()), 'damaged_nats': float(nats),
            'damaged_warning': bool(frac > warn)}


# ---------------------------------------------------------------------------
# Growing the grid (procedure (b))
# ---------------------------------------------------------------------------

def grid_limits(sel, L_grid):
    r"""
    Does the grid :math:`\{0,\dots,L_{\rm grid}\}` limit the result of ``select_lag``?

    The grid limits the result if at least one of these holds:

    * ``tol_capped``: the fit gives a tolerance lag above :math:`L_{\rm grid}`,
      or the fit is not valid (the slope is not negative).
    * ``cv_at_edge``: :math:`L_{\rm cv}=L_{\rm grid}`. The held-out NLL is still
      the lowest at the edge of the grid.
    * ``all_significant``: every decrement :math:`\hat D(0),\dots,\hat D(L_{\rm grid}-1)`
      is significant. The decay is not finished at the edge.

    Parameters
    ----------
    sel : dict
        Output of :func:`select_lag` for the grid ``0..L_grid``.
    L_grid : int

    Returns
    -------
    dict
        The three flags and ``limited`` (true if one of them is true).
    """
    flags = {'tol_capped': bool(sel['tol_capped']),
             'cv_at_edge': bool(sel['L_cv'] == L_grid),
             'all_significant': bool(sel['n_prefix'] >= L_grid)}
    flags['limited'] = any(flags.values())
    return flags


def grow_grid(score_fn, L_grid0, L_cap, T, K_star, **select_kw):
    r"""
    Procedure (b): start with a small grid and make it bigger only when it
    limits the result.

    1. Score the lags :math:`0,\dots,L_{\rm grid}` with ``score_fn`` and run
       :func:`select_lag`.
    2. If :func:`grid_limits` says that the grid limits the result and
       :math:`L_{\rm grid}<L_{\rm cap}`, set
       :math:`L_{\rm grid}\leftarrow\min(2L_{\rm grid},L_{\rm cap})`, score the
       new grid again, and go to 1.
    3. Stop when the grid does not limit the result, or when
       :math:`L_{\rm grid}=L_{\rm cap}`. In the second case, ``grid_capped``
       is true if the grid still limits the result.

    The lag-dropout network has the shape of the grid, so it is trained again
    when the grid grows, and ``score_fn`` returns all the lags each time. Every
    lag is scored on the same targets, :math:`t>L_{\rm cap}`, and the same
    held-out trajectories, so the scores of different grids can be compared.

    Parameters
    ----------
    score_fn : callable
        ``score_fn(L_grid)`` returns the array of per-trajectory held-out NLL
        of the lags ``0..L_grid``, shape ``(N, L_grid + 1)``.
    L_grid0 : int
        First grid (at least 2, at most ``L_cap``).
    L_cap : int
        The largest lag that the method can examine (at least 2).
    T, K_star, select_kw
        See :func:`select_lag`.

    Returns
    -------
    dict
        ``nll`` (``(N, L_grid + 1)``), ``sel`` (output of :func:`select_lag`
        for the final grid), ``L_grid`` (final), ``history`` (one dict for each
        grid with ``L_grid``, ``L_star``, ``L_tol``, ``L_cv`` and the flags of
        :func:`grid_limits`), ``grid_capped``.
    """
    L_cap = int(L_cap)
    if L_cap < 2:
        raise ValueError('L_cap must be at least 2 (the series is too short)')
    L_grid = int(min(max(L_grid0, 2), L_cap))
    history = []
    while True:
        nll = np.asarray(score_fn(L_grid), dtype=float)
        sel = select_lag(nll, T, K_star, **select_kw)
        flags = grid_limits(sel, L_grid)
        history.append({'L_grid': L_grid, 'L_star': sel['L_star'], 'L_tol': sel['L_tol'],
                        'L_cv': sel['L_cv'], **flags})
        if not flags['limited'] or L_grid >= L_cap:
            break
        L_grid = min(2 * L_grid, L_cap)
    return {'nll': nll, 'sel': sel, 'L_grid': L_grid, 'history': history,
            'grid_capped': bool(flags['limited'] and L_grid >= L_cap)}
