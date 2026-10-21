"""Unit tests for the pure blocks of inference/lag_selection.py (request I1).

No network is used. Exact autocovariances are used wherever a closed form
exists, so the checks are to numerical precision and not statistical.
"""

import os
import sys
import unittest

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'neuralssm'))

from inference import lag_selection as ls  # noqa: E402


def durbin_levinson_pacf(gamma, kmax):
    """Independent scalar PACF (Brockwell and Davis, Prop. 5.2.1)."""
    phi = np.zeros(kmax + 1)
    prev = np.zeros(0)
    for k in range(1, kmax + 1):
        if k == 1:
            phi_kk = gamma[1] / gamma[0]
            cur = np.array([phi_kk])
        else:
            num = gamma[k] - np.dot(prev, gamma[k - 1:0:-1])
            den = gamma[0] - np.dot(prev, gamma[1:k])
            phi_kk = num / den
            cur = np.concatenate([prev - phi_kk * prev[::-1], [phi_kk]])
        phi[k] = phi_kk
        prev = cur
    return phi[1:]


def arma11_autocov(a, ma, sigma2, hmax):
    """Autocovariances of x_t = a x_{t-1} + e_t + ma e_{t-1}."""
    g0 = sigma2 * (1 + 2 * a * ma + ma ** 2) / (1 - a ** 2)
    g1 = a * g0 + ma * sigma2
    return np.array([g0, g1] + [a ** (h - 1) * g1 for h in range(2, hmax + 1)])


class TestTolerance(unittest.TestCase):

    def test_values(self):
        self.assertAlmostEqual(ls.kl_tolerance(100), 0.02)
        self.assertAlmostEqual(ls.kl_tolerance(100, eta=0.2), 0.08)
        self.assertAlmostEqual(ls.kl_tolerance(100, eps=1e-3), 0.1)
        with self.assertRaises(ValueError):
            ls.kl_tolerance(100, eta=0.1, eps=0.1)


class TestStage0(unittest.TestCase):

    def test_autocov_ar1(self):
        rng = np.random.default_rng(0)
        a, n = 0.6, 200000
        e = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = a * y[t - 1] + e[t]
        G = ls.sample_autocovariances(y, 3)
        g0 = 1 / (1 - a ** 2)
        np.testing.assert_allclose(G[:, 0, 0], g0 * a ** np.arange(4), rtol=0.03)

    def test_scalar_ar1_exact(self):
        a, s2 = 0.7, 2.0
        g = (s2 / (1 - a ** 2)) * a ** np.arange(8)
        V = ls.innovation_covariances(g[:, None, None], 6)[:, 0, 0]
        np.testing.assert_allclose(V[0], g[0])
        np.testing.assert_allclose(V[1:], s2, rtol=1e-10)

    def test_scalar_matches_pacf_arma11(self):
        a, ma, s2, kmax = 0.9, -0.5, 1.0, 12
        g = arma11_autocov(a, ma, s2, kmax + 1)
        V = ls.innovation_covariances(g[:, None, None], kmax + 1)[:, 0, 0]
        D = 0.5 * np.log(V[:-1] / V[1:])
        phi = durbin_levinson_pacf(g, kmax + 1)
        np.testing.assert_allclose(D, -0.5 * np.log(1 - phi ** 2),
                                   rtol=1e-6, atol=1e-14)
        self.assertTrue(np.all(D > 0))

    def test_var1_exact(self):
        A = np.array([[0.5, 0.2], [-0.1, 0.4]])
        S = np.array([[1.0, 0.3], [0.3, 0.5]])
        G0 = solve_discrete_lyapunov(A, S)
        G = np.stack([np.linalg.matrix_power(A, h) @ G0 for h in range(6)])
        V = ls.innovation_covariances(G, 4)
        np.testing.assert_allclose(V[0], G0)
        for k in range(1, 5):
            np.testing.assert_allclose(V[k], S, atol=1e-9)

    def test_proxy_on_simulated_ar1(self):
        rng = np.random.default_rng(1)
        a, n, N = 0.5, 400, 50
        y = np.zeros((N, n, 1))
        e = rng.standard_normal((N, n))
        for t in range(1, n):
            y[:, t, 0] = a * y[:, t - 1, 0] + e[:, t]
        D = ls.gaussian_proxy_decrements(y, 5)
        self.assertEqual(D.shape, (6,))
        self.assertAlmostEqual(D[0], -0.5 * np.log(1 - a ** 2), delta=0.02)
        self.assertTrue(np.all(np.abs(D[1:]) < 0.01))

    def test_stage0_default_noise_floor(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal((50, 200, 1))          # white noise: no dependence
        out = ls.stage0_lag(y, T=200, K_star=0.02)
        self.assertEqual(out['fit_status'], 'none')
        self.assertEqual(out['L0'], 0)

    def test_stage0_lag(self):
        # Geometric D via ARMA(1,1) with a slow AR root.
        rng = np.random.default_rng(2)
        a, ma, n, N = 0.9, -0.5, 500, 40
        e = rng.standard_normal((N, n))
        y = np.zeros((N, n))
        for t in range(1, n):
            y[:, t] = a * y[:, t - 1] + e[:, t] + ma * e[:, t - 1]
        out = ls.stage0_lag(y[..., None], T=n, K_star=0.02, noise_floor=1e-4)
        self.assertGreater(out['L0'], 3)
        self.assertEqual(out['L_grid'], max(2 * out['L0'], 2))
        self.assertTrue(0 < out['rho'] < 1)


class TestCurveAndRules(unittest.TestCase):

    def test_fit_recovers_geometric(self):
        L = np.arange(15)
        D = 0.3 * 0.7 ** L
        fit = ls.fit_forgetting_curve(D)
        self.assertAlmostEqual(fit['C'], 0.3)
        self.assertAlmostEqual(fit['rho'], 0.7)
        self.assertEqual(fit['n_used'], 15)

    def test_fit_uses_only_significant_lags(self):
        L = np.arange(15)
        D = 0.3 * 0.7 ** L
        SE = np.full(15, 0.02)
        fit = ls.fit_forgetting_curve(D, SE)
        self.assertTrue(np.all(D[fit['mask']] > 2 * SE[fit['mask']]))
        self.assertFalse(fit['mask'][-1])
        self.assertAlmostEqual(fit['rho'], 0.7)

    def test_fit_degenerate(self):
        fit = ls.fit_forgetting_curve(np.array([0.1, 0.0, 0.0]))
        self.assertTrue(np.isnan(fit['rho']))
        fit = ls.fit_forgetting_curve(np.array([0.1, 0.2, 0.3]))   # increasing
        self.assertTrue(np.isnan(fit['rho']))

    def test_L_from_rate_is_smallest(self):
        C, rho, T, K = 0.5, 0.8, 200, 0.02
        L = ls.L_from_rate(C, rho, T, K)
        f = lambda l: T * C * rho ** l / (1 - rho)  # noqa: E731
        self.assertLessEqual(f(L), K)
        self.assertGreater(f(L - 1), K)

    def test_L_from_rate_edge(self):
        self.assertEqual(ls.L_from_rate(1e-9, 0.5, 100, 0.02), 0)
        self.assertEqual(ls.L_from_rate(0.5, 1.2, 100, 0.02), np.inf)
        self.assertEqual(ls.L_from_rate(0.5, np.nan, 100, 0.02), np.inf)
        self.assertEqual(ls.L_from_rho(0.8, 0.5, 200, 0.02),
                         ls.L_from_rate(0.5, 0.8, 200, 0.02))

    def test_one_se_lag_unpaired_vs_paired(self):
        """A large shared offset inflates the unpaired SE and shortens the lag."""
        rng = np.random.default_rng(0)
        N = 200
        H = np.array([3.0, 2.0, 1.05, 1.0, 1.01, 1.02])
        nll = H + rng.standard_normal((N, 1)) * 1.0 + rng.standard_normal((N, 6)) * 0.01
        self.assertEqual(ls.one_se_lag(nll, paired=False), 2)   # SE ~ 0.07 swamps H(2)-H(3)=0.05
        self.assertEqual(ls.one_se_lag(nll, paired=True), 3)

    def test_prefix_rule_ignores_isolated_noise_lag(self):
        L = np.arange(15)
        D = 0.3 * 0.5 ** L
        SE = np.full(15, 0.002)
        D2 = D.copy()
        D2[11] = 0.01                                  # isolated spike above 2 SE
        pre = ls.fit_forgetting_curve(D2, SE, prefix=True)
        note = ls.fit_forgetting_curve(D2, SE, prefix=False)
        self.assertAlmostEqual(pre['rho'], 0.5, places=3)
        self.assertFalse(pre['mask'][11])
        self.assertTrue(note['mask'][11])
        self.assertGreater(note['rho'], 0.6)           # one noise lag flattens the line

    def test_min_points(self):
        fit = ls.fit_forgetting_curve(np.array([0.5, 0.1, 0.0, 0.0]), np.full(4, 0.01))
        self.assertEqual(fit['n_prefix'], 2)
        self.assertTrue(np.isnan(fit['rho']))
        fit = ls.fit_forgetting_curve(np.array([0.5, 0.1, 0.0, 0.0]), np.full(4, 0.01), min_points=2)
        self.assertAlmostEqual(fit['rho'], 0.2, places=6)


def synthetic_nll(N, L_grid, C, rho, noise, seed, h_inf=1.0):
    """H(L)=h_inf+C rho^L/(1-rho) with a per-trajectory offset shared by all
    lags (so the paired differences have small variance)."""
    rng = np.random.default_rng(seed)
    L = np.arange(L_grid + 1)
    H = h_inf + C * rho ** L / (1 - rho)
    shared = rng.standard_normal((N, 1)) * 0.5
    own = rng.standard_normal((N, L_grid + 1)) * noise
    return H + shared + own


class TestSelection(unittest.TestCase):

    def test_decrements_shapes_and_pairing(self):
        nll = synthetic_nll(100, 10, 0.3, 0.6, 0.002, seed=0)
        out = ls.decrements_from_nll(nll)
        self.assertEqual(out['H'].shape, (11,))
        self.assertEqual(out['D'].shape, (10,))
        # paired SE of D is small although the SE of H is large (shared offset)
        self.assertLess(out['SE_D'].max(), 0.01)
        self.assertGreater(out['SE_H'].min(), 0.02)
        with self.assertRaises(ValueError):
            ls.decrements_from_nll(nll[:1])

    def test_select_recovers_exact_tolerance_lag(self):
        C, rho, T = 0.3, 0.6, 200
        K = ls.kl_tolerance(T)                     # 0.02
        nll = synthetic_nll(400, 25, C, rho, 0.0005, seed=1)
        out = ls.select_lag(nll, T, K)
        exact = ls.L_from_rate(C, rho, T, K)
        self.assertAlmostEqual(out['rho'], rho, delta=0.1)
        self.assertLessEqual(abs(out['L_tol'] - exact), 2)
        self.assertEqual(out['L_star'], min(out['L_tol'], out['L_cv']))
        self.assertFalse(out['tol_capped'])
        self.assertEqual(out['fit_status'], 'ok')

    def test_select_unresolved(self):
        # Only lags 0 and 1 are above the noise: the decay rate cannot be fitted.
        nll = synthetic_nll(200, 10, 1.0, 0.01, 0.003, seed=9)
        out = ls.select_lag(nll, 100, 0.02)
        self.assertEqual(out['fit_status'], 'unresolved')
        self.assertEqual(out['L_tol'], out['n_prefix'])
        self.assertLess(out['n_prefix'], 3)

    def test_select_cv_limits(self):
        # Large noise: the NLL curve is flat within one SE early, so L_cv < L_tol.
        nll = synthetic_nll(40, 25, 0.3, 0.9, 0.3, seed=2)
        out = ls.select_lag(nll, 200, 0.02)
        self.assertLessEqual(out['L_star'], out['L_tol'])
        self.assertEqual(out['limited_by'], 'cv' if out['L_cv'] < out['L_tol'] else 'tol')

    def test_select_capped(self):
        # Slow forgetting and a small grid: the tolerance lag is beyond L_grid.
        nll = synthetic_nll(400, 8, 0.3, 0.97, 0.0005, seed=3)
        out = ls.select_lag(nll, 1000, 0.02)
        self.assertTrue(out['tol_capped'])
        self.assertEqual(out['L_tol'], 8)

    def test_select_no_signal(self):
        rng = np.random.default_rng(4)
        nll = 1.0 + 0.1 * rng.standard_normal((50, 1)) + 1e-5 * rng.standard_normal((50, 6))
        out = ls.select_lag(nll, 100, 0.02)
        self.assertEqual(out['L_tol'], 0)
        self.assertEqual(out['L_star'], 0)
        self.assertEqual(out['fit_status'], 'none')

    def test_determinism(self):
        nll = synthetic_nll(100, 15, 0.3, 0.7, 0.01, seed=5)
        a, b = ls.select_lag(nll, 200, 0.02), ls.select_lag(nll.copy(), 200, 0.02)
        for k in ('L_star', 'L_tol', 'L_cv', 'rho', 'C'):
            self.assertEqual(a[k], b[k])

    def test_theta_bins_take_max(self):
        # Half the trajectories forget fast (rho=0.3), half slowly (rho=0.8).
        T, K = 200, 0.02
        fast = synthetic_nll(200, 25, 0.3, 0.3, 0.0005, seed=6)
        slow = synthetic_nll(200, 25, 0.3, 0.8, 0.0005, seed=7)
        nll = np.concatenate([fast, slow])
        theta = np.concatenate([np.zeros(200), np.ones(200)]) + 0.01 * np.arange(400) / 400
        out = ls.select_lag_by_theta_bins(nll, theta, T, K, n_bins=2, quantile=1.0)
        self.assertEqual(out['L_star'], out['L_bins'].max())
        self.assertGreater(out['L_bins'][1], out['L_bins'][0] + 3)
        pooled = ls.select_lag(nll, T, K)['L_star']
        self.assertGreaterEqual(out['L_star'], pooled - 1)


class TestGridGrowth(unittest.TestCase):
    """Procedure (b): the grid grows only when it limits the result."""

    def make_score_fn(self, full):
        calls = []

        def score_fn(L_grid):
            calls.append(L_grid)
            return full[:, :L_grid + 1]
        return score_fn, calls

    def test_grid_limits_flags(self):
        base = dict(tol_capped=False, L_cv=2, n_prefix=1)
        self.assertFalse(ls.grid_limits(base, 8)['limited'])
        self.assertTrue(ls.grid_limits(dict(base, tol_capped=True), 8)['tol_capped'])
        self.assertTrue(ls.grid_limits(dict(base, L_cv=8), 8)['cv_at_edge'])
        self.assertTrue(ls.grid_limits(dict(base, n_prefix=8), 8)['all_significant'])
        self.assertTrue(ls.grid_limits(dict(base, n_prefix=8), 8)['limited'])

    def test_grows_when_decay_is_slow(self):
        full = synthetic_nll(400, 32, 0.3, 0.8, 0.0005, seed=11)
        score_fn, calls = self.make_score_fn(full)
        out = ls.grow_grid(score_fn, 4, 32, 200, ls.kl_tolerance(200))
        grids = [h['L_grid'] for h in out['history']]
        self.assertEqual(grids[0], 4)
        self.assertGreater(len(grids), 1)
        for a, b in zip(grids[:-1], grids[1:]):
            self.assertEqual(b, min(2 * a, 32))
        self.assertTrue(all(h['limited'] for h in out['history'][:-1]))
        self.assertEqual(calls, grids)                          # one scoring call for each grid

    def test_no_growth_when_not_limited(self):
        full = synthetic_nll(400, 16, 0.3, 0.2, 0.0005, seed=12)
        score_fn, calls = self.make_score_fn(full)
        out = ls.grow_grid(score_fn, 8, 16, 100, ls.kl_tolerance(100))
        self.assertEqual(len(out['history']), 1)
        self.assertEqual(out['L_grid'], 8)
        self.assertFalse(out['grid_capped'])
        self.assertEqual(calls, [8])

    def test_grid_capped_flag(self):
        full = synthetic_nll(400, 8, 0.3, 0.97, 0.0005, seed=13)
        score_fn, _ = self.make_score_fn(full)
        out = ls.grow_grid(score_fn, 4, 8, 1000, ls.kl_tolerance(1000))
        self.assertEqual(out['L_grid'], 8)
        self.assertTrue(out['grid_capped'])

    def test_result_does_not_depend_on_growth_path(self):
        full = synthetic_nll(300, 32, 0.3, 0.8, 0.0005, seed=14)
        K = ls.kl_tolerance(200)
        grown = ls.grow_grid(self.make_score_fn(full)[0], 2, 32, 200, K)
        direct = ls.grow_grid(self.make_score_fn(full)[0], grown['L_grid'], 32, 200, K)
        np.testing.assert_array_equal(grown['nll'], full[:, :grown['L_grid'] + 1])
        one_shot = ls.select_lag(full[:, :grown['L_grid'] + 1], 200, K)
        for k in ('L_star', 'L_tol', 'L_cv', 'rho'):
            self.assertEqual(grown['sel'][k], one_shot[k])
        self.assertEqual(grown['sel']['L_star'], direct['sel']['L_star'])

    def test_array_of_the_last_grid_replaces_the_old_one(self):
        """The network is trained again when the grid grows, so its scores depend on the grid."""
        full = synthetic_nll(400, 32, 0.3, 0.8, 0.0005, seed=11)

        def score_fn(L_grid):
            return full[:, :L_grid + 1] + 0.001 * L_grid

        K = ls.kl_tolerance(200)
        out = ls.grow_grid(score_fn, 4, 32, 200, K)
        self.assertGreater(len(out['history']), 1)
        np.testing.assert_allclose(out['nll'], full[:, :out['L_grid'] + 1] + 0.001 * out['L_grid'])
        direct = ls.grow_grid(score_fn, out['L_grid'], 32, 200, K)
        for k in ('L_star', 'L_tol', 'L_cv'):
            self.assertEqual(out['sel'][k], direct['sel'][k])

    def test_first_grid_is_clipped_to_the_cap_and_floor(self):
        full = synthetic_nll(100, 8, 0.3, 0.3, 0.001, seed=15)
        out = ls.grow_grid(self.make_score_fn(full)[0], 50, 8, 100, 0.02)
        self.assertEqual(out['history'][0]['L_grid'], 8)
        out = ls.grow_grid(self.make_score_fn(full)[0], 0, 8, 100, 0.02)
        self.assertEqual(out['history'][0]['L_grid'], 2)

    def test_cap_too_small(self):
        with self.assertRaises(ValueError):
            ls.grow_grid(lambda L_grid: np.zeros((10, L_grid + 1)), 2, 1, 10, 0.02)


class TestResolutionFailure(unittest.TestCase):

    def test_flag_table(self):
        self.assertTrue(ls.resolution_failure(27, 0))      # the A5 slow regime: stage 0 sees, stage 1 does not
        self.assertTrue(ls.resolution_failure(3, 2))
        self.assertFalse(ls.resolution_failure(3, 3))      # stage 1 resolves
        self.assertFalse(ls.resolution_failure(2, 0))      # stage 0 itself sees too little
        self.assertFalse(ls.resolution_failure(0, 0))      # no dependence at all
        self.assertTrue(ls.resolution_failure(5, 4, min_points=5))

    def test_returns_a_python_bool(self):
        self.assertIs(ls.resolution_failure(np.int64(9), np.int64(0)), True)

    def test_flag_from_a_flat_curve_and_a_clean_curve(self):
        T, K = 200, 0.02
        clean = synthetic_nll(400, 12, 0.3, 0.5, 0.0005, seed=21)
        flat = synthetic_nll(400, 12, 0.003, 0.5, 0.05, seed=22)          # decrements far below the noise
        n_clean = ls.select_lag(clean, T, K)['n_prefix']
        n_flat = ls.select_lag(flat, T, K)['n_prefix']
        self.assertGreaterEqual(n_clean, 3)
        self.assertLess(n_flat, 3)
        self.assertFalse(ls.resolution_failure(8, n_clean))
        self.assertTrue(ls.resolution_failure(8, n_flat))


class TestAdaLagMax(unittest.TestCase):

    def test_twice_the_first_lag_with_floor_and_cap(self):
        self.assertEqual(ls.ada_lag_max(5, 25), 10)
        self.assertEqual(ls.ada_lag_max(0, 25), 2)         # floor: the smallest grid
        self.assertEqual(ls.ada_lag_max(1, 25), 2)
        self.assertEqual(ls.ada_lag_max(20, 25), 25)       # cap L_cap
        self.assertEqual(ls.ada_lag_max(3, 4), 4)

    def test_returns_int(self):
        self.assertIsInstance(ls.ada_lag_max(np.int64(4), np.int64(30)), int)


class TestDamageDiagnostic(unittest.TestCase):
    """Decision E13: the fraction of trajectories with a damaged score. It only reports."""

    def clean(self, n=200, seed=0):
        return synthetic_nll(n, 8, 0.3, 0.5, 0.002, seed=seed)

    def test_clean_scores_have_no_damage(self):
        d = ls.damage_diagnostic(self.clean())
        self.assertEqual((d['n_damaged'], d['damaged_fraction'], d['damaged_warning']), (0, 0.0, False))
        self.assertEqual(d['damaged_nats'], ls.DAMAGE_NATS)

    def test_a_lag_worse_than_lag_zero_by_more_than_the_threshold_is_damage(self):
        nll = self.clean()
        nll[:10, 5] = nll[:10, 0] + 1.5                     # 10 of 200 trajectories: lag 5 is 1.5 nats above lag 0
        mask = ls.damaged_trajectories(nll)
        self.assertEqual(int(mask.sum()), 10)
        self.assertTrue(mask[:10].all())
        d = ls.damage_diagnostic(nll)
        self.assertAlmostEqual(d['damaged_fraction'], 0.05)
        self.assertFalse(d['damaged_warning'])              # the warning needs more than 5 %
        nll[10:11, 3] = nll[10:11, 0] + 5.0
        self.assertTrue(ls.damage_diagnostic(nll)['damaged_warning'])

    def test_a_small_excess_is_not_damage(self):
        nll = self.clean()
        nll[:, 4] = nll[:, 0] + 0.9                         # every trajectory, below the threshold
        self.assertEqual(int(ls.damaged_trajectories(nll).sum()), 0)

    def test_non_finite_scores_are_damage(self):
        nll = self.clean(n=20)
        nll[3, 2] = np.nan
        nll[7, 6] = np.inf
        self.assertEqual(sorted(np.flatnonzero(ls.damaged_trajectories(nll))), [3, 7])

    def test_the_threshold_is_an_argument(self):
        nll = self.clean()
        nll[:4, 2] = nll[:4, 0] + 0.5
        self.assertEqual(int(ls.damaged_trajectories(nll, nats=0.25).sum()), 4)
        self.assertEqual(int(ls.damaged_trajectories(nll, nats=1.0).sum()), 0)

    def test_it_does_not_change_the_selection(self):
        nll = self.clean()
        before = ls.select_lag(nll, 200, 0.02)
        nll2 = nll.copy()
        ls.damage_diagnostic(nll2)
        np.testing.assert_array_equal(nll, nll2)
        self.assertEqual(before['L_star'], ls.select_lag(nll2, 200, 0.02)['L_star'])


if __name__ == '__main__':
    unittest.main()
