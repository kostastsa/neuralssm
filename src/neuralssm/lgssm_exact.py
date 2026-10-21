r"""
Exact quantities for the scalar LGSSM, used to validate lag selection (I1)
and as the reference for tasks A1 and A5.

Model
-----
.. math::
    x_t = a\,x_{t-1} + w_t,\quad w_t\sim N(0,Q),\qquad
    y_t = h\,x_t + v_t,\quad v_t\sim N(0,R),\qquad |a|<1,

in the stationary regime (:math:`x_0\sim N(0,Q/(1-a^2))`).

Kalman steady state
-------------------
Let :math:`P` be the steady-state variance of the one-step state prediction.
It solves the scalar Riccati equation

.. math:: P = a^2\Bigl(P-\frac{h^2P^2}{h^2P+R}\Bigr)+Q,

the innovation variance is :math:`S=h^2P+R`, and the steady-state filter gain
is :math:`K=P h/S`. The innovation :math:`e_t=y_t-h\hat x_{t|t-1}` is white
with variance :math:`S`, and

.. math:: y_t = a\,y_{t-1}+e_t-a(1-Kh)\,e_{t-1},

so :math:`y_t` is a stationary ARMA(1,1) process with AR coefficient :math:`a`,
MA coefficient :math:`\theta=-a(1-Kh)` and innovation variance :math:`S`
(an AR(1) plus white noise; Granger and Morris 1976).

Forgetting rate
---------------
The Kalman filter forgets its initial condition at the pathwise rate

.. math:: \rho_{KF}=|a(1-Kh)|=|\theta|.

The expected NLL decrement :math:`D(L)` decays at the squared rate
:math:`\rho_{KF}^2` (guidance Sec. 3.2).

Exact decrements
----------------
The autocovariances of :math:`y_t` are
:math:`\gamma(0)=h^2\frac{Q}{1-a^2}+R` and
:math:`\gamma(k)=h^2\frac{Q}{1-a^2}a^k` for :math:`k\ge1`. For a Gaussian
process the conditional entropy given :math:`L` lags is
:math:`H(L)=\tfrac12\log(2\pi e V_L)` with :math:`V_L` the order-``L``
innovation variance (Durbin-Levinson), so

.. math:: D(L)=\tfrac12\log\frac{V_L}{V_{L+1}}=-\tfrac12\log(1-\phi_{L+1}^2),

with :math:`\phi_k` the PACF. Here :math:`V_L` and :math:`\phi_k` are computed
by the Durbin-Levinson recursion on the exact autocovariances, independent of
``lag_selection.innovation_covariances``.
"""

import numpy as np


def kalman_steady_state(a, h, Q, R, tol=1e-14, max_iter=100000):
    r"""
    Steady-state prediction variance, innovation variance and gain.

    Iterates the Riccati recursion :math:`P\leftarrow a^2(P-h^2P^2/(h^2P+R))+Q`
    to convergence.

    Returns
    -------
    dict
        ``P`` (prediction variance), ``S`` (innovation variance
        :math:`h^2P+R`), ``K`` (filter gain :math:`Ph/S`), ``rho_KF``
        (:math:`|a(1-Kh)|`) and ``theta_MA`` (:math:`-a(1-Kh)`).
    """
    if abs(a) >= 1:
        raise ValueError("need |a| < 1 for a stationary model")
    P = Q / (1.0 - a ** 2)
    for _ in range(max_iter):
        S = h ** 2 * P + R
        P_new = a ** 2 * (P - h ** 2 * P ** 2 / S) + Q
        if abs(P_new - P) < tol * max(1.0, abs(P)):
            P = P_new
            break
        P = P_new
    S = h ** 2 * P + R
    K = P * h / S
    theta = -a * (1.0 - K * h)
    return {'P': P, 'S': S, 'K': K, 'rho_KF': abs(theta), 'theta_MA': theta}


def observation_autocov(a, h, Q, R, hmax):
    r"""
    Autocovariances :math:`\gamma(0),\dots,\gamma(h_{\max})` of the observations
    (see the module docstring).
    """
    px = Q / (1.0 - a ** 2)
    g = h ** 2 * px * a ** np.arange(hmax + 1)
    g[0] += R
    return g


def durbin_levinson(gamma, kmax):
    r"""
    Durbin-Levinson recursion (Brockwell and Davis 1991, Prop. 5.2.1).

    .. math::
        \phi_{kk}=\frac{\gamma(k)-\sum_{j=1}^{k-1}\phi_{k-1,j}\gamma(k-j)}{V_{k-1}},
        \qquad V_k=V_{k-1}(1-\phi_{kk}^2),\quad V_0=\gamma(0).

    Returns
    -------
    phi : array (kmax,)
        The PACF :math:`\phi_{11},\dots,\phi_{kmax,kmax}`.
    V : array (kmax + 1,)
        Innovation variances :math:`V_0,\dots,V_{kmax}`.
    """
    phi = np.zeros(kmax)
    V = np.zeros(kmax + 1)
    V[0] = gamma[0]
    prev = np.zeros(0)
    for k in range(1, kmax + 1):
        num = gamma[k] - np.dot(prev, gamma[k - 1:0:-1]) if k > 1 else gamma[1]
        phi_kk = num / V[k - 1]
        cur = np.concatenate([prev - phi_kk * prev[::-1], [phi_kk]])
        phi[k - 1] = phi_kk
        V[k] = V[k - 1] * (1.0 - phi_kk ** 2)
        prev = cur
    return phi, V


def exact_decrements(a, h, Q, R, L_max):
    r"""
    Exact :math:`D(L)`, :math:`H(L)` and the PACF of the scalar LGSSM.

    .. math::
        D(L)=-\tfrac12\log(1-\phi_{L+1}^2),\qquad
        H(L)=\tfrac12\log(2\pi e\,V_L).

    Returns
    -------
    dict
        ``D`` (length ``L_max + 1``, :math:`L=0,\dots,L_{\max}`), ``H``
        (length ``L_max + 2``), ``pacf`` (length ``L_max + 1``), ``V``,
        plus the output of :func:`kalman_steady_state`.
    """
    g = observation_autocov(a, h, Q, R, L_max + 1)
    phi, V = durbin_levinson(g, L_max + 1)
    out = kalman_steady_state(a, h, Q, R)
    out.update(D=-0.5 * np.log1p(-phi ** 2), H=0.5 * np.log(2 * np.pi * np.e * V),
               pacf=phi, V=V)
    return out


def exact_tolerance_lag(a, h, Q, R, T, K_star, L_cap=2000):
    r"""
    Smallest :math:`L` with :math:`T\sum_{m\ge L}D(m)\le K^\star`, from the exact
    decrements. Reference for the truncation error of D4 (the sum is
    truncated at ``L_cap``).

    Returns
    -------
    int
        The lag, or ``L_cap`` if the tolerance is not met within the cap.
    """
    D = exact_decrements(a, h, Q, R, L_cap)['D']
    tail = np.cumsum(D[::-1])[::-1]          # tail[L] = sum_{m>=L} D(m)
    ok = np.flatnonzero(T * tail <= K_star)
    return int(ok[0]) if len(ok) else L_cap


def simulate(key_seed, a, h, Q, R, T, n_series=1):
    r"""
    Simulate ``n_series`` stationary series of length ``T``.

    Returns
    -------
    array, shape (n_series, T, 1)
    """
    rng = np.random.default_rng(key_seed)
    x = rng.standard_normal(n_series) * np.sqrt(Q / (1 - a ** 2))
    y = np.zeros((n_series, T, 1))
    w = rng.standard_normal((n_series, T)) * np.sqrt(Q)
    v = rng.standard_normal((n_series, T)) * np.sqrt(R)
    for t in range(T):
        x = a * x + w[:, t]
        y[:, t, 0] = h * x + v[:, t]
    return y
