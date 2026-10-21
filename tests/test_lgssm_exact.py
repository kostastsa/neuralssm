"""Tests for lgssm_exact.py and the exact check of the stage-0 proxy (request I1)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'neuralssm'))

import lgssm_exact as ex  # noqa: E402
from inference import lag_selection as ls  # noqa: E402


def arma11_autocov(a, ma, s2, hmax):
    g0 = s2 * (1 + 2 * a * ma + ma ** 2) / (1 - a ** 2)
    g1 = a * g0 + ma * s2
    return np.array([g0, g1] + [a ** (k - 1) * g1 for k in range(2, hmax + 1)])


class TestExact(unittest.TestCase):

    def test_riccati_fixed_point(self):
        for a, h, Q, R in [(0.9, 1, 0.1, 1.0), (0.99, 1, 0.1, 100.0), (-0.5, 2, 1.0, 0.3)]:
            s = ex.kalman_steady_state(a, h, Q, R)
            P = s['P']
            self.assertAlmostEqual(P, a ** 2 * (P - h ** 2 * P ** 2 / (h ** 2 * P + R)) + Q, places=10)
            self.assertAlmostEqual(s['K'], P * h / (h ** 2 * P + R))

    def test_arma11_representation(self):
        """AR(1)+noise equals ARMA(1,1) with theta=-a(1-Kh), sigma^2=S."""
        for a, h, Q, R in [(0.9, 1, 0.1, 1.0), (0.99, 1, 0.1, 100.0), (0.5, 1.5, 0.7, 0.2)]:
            s = ex.kalman_steady_state(a, h, Q, R)
            g = ex.observation_autocov(a, h, Q, R, 10)
            g_arma = arma11_autocov(a, s['theta_MA'], s['S'], 10)
            np.testing.assert_allclose(g, g_arma, rtol=1e-8)

    def test_durbin_levinson_vs_block_solver(self):
        g = ex.observation_autocov(0.9, 1, 0.1, 1.0, 21)
        phi, V = ex.durbin_levinson(g, 20)
        V2 = ls.innovation_covariances(g[:, None, None], 20)[:, 0, 0]
        np.testing.assert_allclose(V, V2, rtol=1e-8)

    def test_decrement_identity(self):
        e = ex.exact_decrements(0.9, 1, 0.1, 1.0, 30)
        np.testing.assert_allclose(e['D'], e['H'][:-1] - e['H'][1:], rtol=1e-8, atol=1e-14)
        self.assertTrue(np.all(e['D'] > 0))

    def test_decay_rate_is_rho_squared(self):
        e = ex.exact_decrements(0.9, 1, 0.1, 1.0, 60)
        D = e['D']
        slope = np.log(D[40] / D[30]) / 10
        self.assertAlmostEqual(np.exp(slope), e['rho_KF'] ** 2, delta=0.02 * e['rho_KF'] ** 2)

    def test_stage0_matches_exact(self):
        """Stage-0 proxy from a large sample vs the exact decrements."""
        a, h, Q, R = 0.9, 1.0, 0.1, 1.0
        y = ex.simulate(0, a, h, Q, R, T=2000, n_series=100)
        e = ex.exact_decrements(a, h, Q, R, 12)
        D_hat = ls.gaussian_proxy_decrements(y, 12)
        np.testing.assert_allclose(D_hat[:6], e['D'][:6], rtol=0.15, atol=1e-3)

    def test_exact_tolerance_lag_monotone_in_R(self):
        T, K = 100, 0.02
        L1 = ex.exact_tolerance_lag(0.99, 1, 0.1, 1.0, T, K)
        L2 = ex.exact_tolerance_lag(0.99, 1, 0.1, 100.0, T, K)
        self.assertGreater(L2, 3 * L1)


if __name__ == '__main__':
    unittest.main()
