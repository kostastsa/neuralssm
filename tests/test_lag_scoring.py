"""Tests for inference/lag_scoring.py (request I1, step 3).

Run from the repository root with the jaxenv environment active:
    python -m unittest tests.test_lag_scoring
"""

import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'neuralssm'))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from flax import nnx  # noqa: E402

from inference import lag_scoring as sc  # noqa: E402
from inference import lag_selection as ls  # noqa: E402
from maf.layers import BatchNormLayer  # noqa: E402
from util.train import reshape_emissions  # noqa: E402


class TestWindows(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = jnp.asarray(rng.standard_normal((3, 12, 2)), dtype=jnp.float32)

    def test_matches_production_reshape(self):
        for L in (0, 1, 4):
            ours = sc.make_windows(self.y, L)
            for n in range(3):
                np.testing.assert_allclose(ours[n], reshape_emissions(self.y[n], L))

    def test_common_targets_identical_across_lags(self):
        L_max = 5
        targets = [sc.make_windows(self.y, L, first=L_max)[..., -2:] for L in range(L_max + 1)]
        for t in targets[1:]:
            np.testing.assert_allclose(t, targets[0])
        self.assertEqual(targets[0].shape, (3, 12 - L_max, 2))

    def test_rows_layout(self):
        cond = jnp.arange(3, dtype=jnp.float32)[:, None] + 100
        rows = sc.make_rows(cond, self.y, L=2, first=2)
        self.assertEqual(rows.shape, (3, 10, 1 + 3 * 2))
        np.testing.assert_allclose(rows[1, :, 0], 101.0)
        np.testing.assert_allclose(rows[0, 0, 1:], self.y[0, 0:3].reshape(-1))


class _OldBatchNorm(BatchNormLayer):
    """Log-determinant of ``BatchNormLayer`` before the 2026-10-20 fix (train mode only)."""

    def __call__(self, x):
        out, ld = super().__call__(x)
        v = jnp.var(x, axis=0) + self.eps
        extra = self.dcond + jnp.sum(jnp.asarray(self.gamma)[:, :self.dcond] - 0.5 * jnp.log(v[:self.dcond]))
        return out, ld + extra


class TestBatchNormLogDet(unittest.TestCase):

    def test_logdet_is_data_columns_only(self):
        dcond, din = 4, 1
        rng = np.random.default_rng(1)
        x = jnp.asarray(np.concatenate([rng.normal(0, 3, (300, dcond)),
                                        rng.normal(0, 0.2, (300, din))], 1), dtype=jnp.float32)
        layer = BatchNormLayer(din, dcond)
        layer.use_running_average = False
        layer.gamma.value = jnp.full((1, din + dcond), 0.3)
        _, ld = layer(x)
        v = jnp.var(x, axis=0) + 1e-5
        np.testing.assert_allclose(ld, jnp.sum(0.3 - 0.5 * jnp.log(v[dcond:])), rtol=1e-5)

    def test_cond_gamma_does_not_move_logdet(self):
        dcond, din = 3, 1
        x = jr.normal(jr.PRNGKey(0), (50, dcond + din))
        layer = BatchNormLayer(din, dcond)
        layer.use_running_average = False
        _, ld0 = layer(x)
        layer.gamma.value = layer.gamma.value.at[0, :dcond].set(5.0)
        _, ld1 = layer(x)
        np.testing.assert_allclose(ld0, ld1)

    def test_backward_logdet_is_inverse(self):
        dcond, din = 2, 2
        x = jr.normal(jr.PRNGKey(1), (64, dcond + din))
        layer = BatchNormLayer(din, dcond)
        layer.use_running_average = True
        layer.gamma.value = layer.gamma.value + 0.2
        layer.set_batch_stats_func(x)
        y, ld = layer(x)
        x2, ld_b = layer.backward(y)
        np.testing.assert_allclose(x2[:, dcond:], x[:, dcond:], atol=1e-5)
        np.testing.assert_allclose(ld + ld_b, 0.0, atol=1e-5)

    def test_fix_leaves_training_unchanged(self):
        """Old and new log-determinants give the same trained function.

        The extra terms of the old version depend only on the conditioning-column
        gamma and on batch statistics of the raw inputs, so every other parameter
        receives the same gradient.
        """
        from maf.density_models import MAF
        import optax

        def build():
            m = MAF(din=1, nmade=2, dhidden=8, nhidden=2, act_fun='relu', dcond=3,
                    rngs=nnx.Rngs(jr.PRNGKey(0)), reverse=True)
            return m

        new, old = build(), build()
        for i, layer in enumerate(old.layers):
            if isinstance(layer, BatchNormLayer):
                old.layers[i] = _OldBatchNorm(1, 3)
        for m in (new, old):
            m.train()
        data = jr.normal(jr.PRNGKey(1), (20 * 64, 4))
        opts = [nnx.Optimizer(m, optax.adam(1e-3)) for m in (new, old)]
        for k in range(20):
            batch = data[k * 64:(k + 1) * 64]
            for m, o in zip((new, old), opts):
                grads = nnx.grad(lambda mm: mm.loss_fn(batch))(m)
                o.update(grads)
        fa = dict(nnx.to_flat_state(nnx.state(new, nnx.Param)))
        fb = dict(nnx.to_flat_state(nnx.state(old, nnx.Param)))
        n_compared = 0
        for path, va in fa.items():
            a, b = np.asarray(va.value), np.asarray(fb[path].value)
            if 'gamma' in path:
                a, b = a[:, 3:], b[:, 3:]
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6, err_msg=str(path))
            n_compared += 1
        self.assertGreaterEqual(n_compared, 16)
        gam_old = [np.asarray(v.value)[:, :3] for p, v in fb.items() if 'gamma' in p]
        self.assertTrue(all(g.min() > 0 for g in gam_old))      # old conditioning gammas drift up


class TestDensity(unittest.TestCase):

    def test_loss_fn_is_mean_nll_of_proper_density(self):
        Lg, p, d = 2, 1, 1
        model = sc.build_dropout_candidate(jr.PRNGKey(5), Lg, p, d, dict(sc.PRODUCTION_HP))
        ref = sc.apply_lag_mask(jr.normal(jr.PRNGKey(6), (2048, p + (Lg + 1) * d)), 1, p, Lg, d)
        sc.freeze_batch_stats(model, ref)
        np.testing.assert_allclose(model.loss_fn(ref[:500]), sc._nll_per_row(model, ref[:500]).mean(), rtol=1e-5)


class TestLagDropout(unittest.TestCase):
    """The lag-dropout network (decision D2)."""

    p, d, L_grid = 2, 2, 4

    def full_rows(self, n=6, seed=0):
        rng = np.random.default_rng(seed)
        return jnp.asarray(rng.standard_normal((n, self.p + (self.L_grid + 1) * self.d)), dtype=jnp.float32)

    def test_lag_mask_layout(self):
        """Oldest first: lags 4, 3, 2, 1. Lag k is kept if k <= ell."""
        m = np.asarray(sc.lag_mask(jnp.array([0, 1, 2, 4]), 4))
        np.testing.assert_array_equal(m, [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]])

    def test_extreme_lags(self):
        rows = self.full_rows()
        p, d, Lg = self.p, self.d, self.L_grid
        full = sc.apply_lag_mask(rows, Lg, p, Lg, d)
        none = sc.apply_lag_mask(rows, 0, p, Lg, d)
        self.assertEqual(full.shape, (6, p + Lg + Lg * d + d))
        np.testing.assert_allclose(full[:, :p], rows[:, :p])                       # theta
        np.testing.assert_allclose(full[:, p:p + Lg], 1.0)                         # indicator
        np.testing.assert_allclose(full[:, p + Lg:], rows[:, p:])                  # whole past and y_t
        np.testing.assert_allclose(none[:, p:p + Lg], 0.0)
        np.testing.assert_allclose(none[:, p + Lg:p + Lg + Lg * d], 0.0)           # past zeroed
        np.testing.assert_allclose(none[:, -d:], rows[:, -d:])                     # y_t kept

    def test_mask_keeps_the_last_ell_observations(self):
        """The kept block equals the past columns of the production row at lag ell."""
        cond = jnp.arange(4, dtype=jnp.float32)[:, None] * jnp.ones((1, self.p))
        y = jnp.asarray(np.random.default_rng(3).standard_normal((4, 12, self.d)), dtype=jnp.float32)
        p, d, Lg = self.p, self.d, self.L_grid
        big = sc.make_rows(cond, y, Lg, first=Lg).reshape(-1, p + (Lg + 1) * d)
        for ell in range(Lg + 1):
            ours = sc.apply_lag_mask(big, ell, p, Lg, d)
            ref = sc.make_rows(cond, y, ell, first=Lg).reshape(-1, p + (ell + 1) * d)
            kept = ours[:, p + Lg + (Lg - ell) * d:]          # last ell observations and y_t
            np.testing.assert_allclose(kept, ref[:, p:])
            np.testing.assert_allclose(ours[:, p + Lg:p + Lg + (Lg - ell) * d], 0.0)

    def test_masked_lag_does_not_change_the_output(self):
        """Observations older than ell change neither the input nor the density."""
        p, d, Lg = self.p, self.d, self.L_grid
        model = sc.build_dropout_candidate(jr.PRNGKey(0), Lg, p, d, dict(sc.PRODUCTION_HP))
        sc.freeze_batch_stats(model, jr.normal(jr.PRNGKey(1), (1024, p + Lg + Lg * d + d)))
        a = self.full_rows(n=64, seed=1)
        for ell in range(Lg + 1):
            b = a.at[:, p:p + (Lg - ell) * d].set(jr.normal(jr.PRNGKey(2), (64, (Lg - ell) * d)) * 50.0)
            xa, xb = (sc.apply_lag_mask(r, ell, p, Lg, d) for r in (a, b))
            np.testing.assert_array_equal(xa, xb)
            np.testing.assert_allclose(sc._nll_per_row(model, xa), sc._nll_per_row(model, xb))

    def test_different_lags_give_different_densities(self):
        """The indicator and the kept past matter: the network is not blind to ell."""
        p, d, Lg = self.p, self.d, self.L_grid
        model = sc.build_dropout_candidate(jr.PRNGKey(0), Lg, p, d, dict(sc.PRODUCTION_HP))
        sc.freeze_batch_stats(model, jr.normal(jr.PRNGKey(1), (1024, p + Lg + Lg * d + d)))
        rows = self.full_rows(n=64, seed=4)
        v = [np.asarray(sc._nll_per_row(model, sc.apply_lag_mask(rows, ell, p, Lg, d))) for ell in range(Lg + 1)]
        self.assertGreater(np.abs(v[0] - v[Lg]).max(), 1e-4)

    def test_density_integrates_to_one(self):
        p, d, Lg = 1, 1, 2
        model = sc.build_dropout_candidate(jr.PRNGKey(3), Lg, p, d, dict(sc.PRODUCTION_HP))
        sc.freeze_batch_stats(model, jr.normal(jr.PRNGKey(4), (2048, p + Lg + Lg * d + d)))
        grid = jnp.linspace(-30.0, 30.0, 60001)
        for ell in (0, 1, 2):
            full = jnp.concatenate([jnp.tile(jnp.array([0.3, -0.5, 0.8]), (grid.shape[0], 1)), grid[:, None]], 1)
            x = sc.apply_lag_mask(full, ell, p, Lg, d)
            integral = float(jnp.sum(jnp.exp(-sc._nll_per_row(model, x))) * (grid[1] - grid[0]))
            self.assertAlmostEqual(integral, 1.0, delta=2e-3)

    def data(self, N=40, T=40, seed=0):
        rng = np.random.default_rng(seed)
        y = np.zeros((N, T, 1), dtype=np.float32)
        for t in range(1, T):
            y[:, t, 0] = 0.7 * y[:, t - 1, 0] + rng.standard_normal(N)
        return jnp.zeros((N, 1)), jnp.asarray(y)

    def small_hp(self):
        return dict(sc.PRODUCTION_HP, epochs=1, nmade=2, dhidden=8, nhidden=2)

    def test_score_dropout_shape_and_determinism(self):
        cond, y = self.data()
        hp = self.small_hp()
        a = sc.score_dropout(jr.PRNGKey(0), cond, y, 4, 3, hp, logger=lambda s: None)
        b = sc.score_dropout(jr.PRNGKey(0), cond, y, 4, 3, hp, logger=lambda s: None)
        self.assertEqual(a['nll'].shape, (a['heldout_idx'].shape[0], 4))
        np.testing.assert_allclose(a['nll'], b['nll'], rtol=1e-5)
        self.assertTrue(np.all(np.isfinite(a['nll'])))
        with self.assertRaises(ValueError):
            sc.score_dropout(jr.PRNGKey(0), cond, y, 4, 5, hp, logger=lambda s: None)

    def test_score_depends_on_the_final_grid_only(self):
        """Same key and split, same L_grid: same array, whatever came before."""
        cond, y = self.data()
        hp = self.small_hp()
        split = sc.split_trajectories(jr.PRNGKey(5), 40, 0.25)
        sc.score_dropout(jr.PRNGKey(1), cond, y, 6, 2, hp, split=split, logger=lambda s: None)
        a = sc.score_dropout(jr.PRNGKey(1), cond, y, 6, 4, hp, split=split, logger=lambda s: None)
        b = sc.score_dropout(jr.PRNGKey(1), cond, y, 6, 4, hp, split=split, logger=lambda s: None)
        np.testing.assert_allclose(a['nll'], b['nll'], rtol=1e-5)

    def test_round0_selection_runs_grows_and_is_deterministic(self):
        cond, y = self.data(N=40, T=40)
        hp = self.small_hp()
        out = [sc.select_lag_round0(jr.PRNGKey(2), cond, y, 40, hp, logger=lambda s: None) for _ in range(2)]
        a, b = out
        self.assertEqual(a['option'], 'A')
        self.assertEqual(a['L_cap'], 10)                       # T // 4
        self.assertLessEqual(a['L_grid'], a['L_cap'])
        self.assertLessEqual(a['L_star'], a['L_grid'])
        self.assertEqual(a['grid_history'][0]['L_grid'], a['L_grid_start'])
        self.assertEqual(len(a['H_hat']), a['L_grid'] + 1)
        self.assertEqual(a['n_heldout'], 6)
        self.assertEqual(a['n_networks'], len(a['grid_history']))        # one network for each grid
        self.assertEqual(a['ld_epochs_factor'], sc.LD_EPOCHS_FACTOR)
        for k in ('L_star', 'L_grid', 'L_tol', 'L_cv'):
            self.assertEqual(a[k], b[k])

    def test_series_too_short(self):
        cond, y = self.data(N=10, T=6)
        with self.assertRaises(ValueError):
            sc.select_lag_round0(jr.PRNGKey(0), cond, y, 6, self.small_hp(), logger=lambda s: None)

    def test_round0_selection_returns_the_scores(self):
        cond, y = self.data(N=40, T=40)
        rec, nll = sc.select_lag_round0(jr.PRNGKey(2), cond, y, 40, self.small_hp(), logger=lambda s: None,
                                        return_nll=True)
        self.assertEqual(nll.shape, (rec['n_heldout'], rec['L_grid'] + 1))
        for k in ('damaged_fraction', 'n_damaged', 'damaged_nats', 'damaged_warning'):
            self.assertIn(k, rec)
        self.assertEqual(rec['damaged_fraction'], ls.damage_diagnostic(nll)['damaged_fraction'])

    def test_save_selection_writes_the_scores(self):
        import tempfile
        cond, y = self.data(N=40, T=40)
        rec, nll = sc.select_lag_round0(jr.PRNGKey(2), cond, y, 40, self.small_hp(), logger=lambda s: None,
                                        return_nll=True)
        with tempfile.TemporaryDirectory() as d:
            sc.save_selection(rec, d, scores={0: nll})
            z = np.load(os.path.join(d, 'lag_selection_nll.npz'))
            np.testing.assert_array_equal(z['round0'], nll)
            self.assertTrue(os.path.exists(os.path.join(d, 'lag_selection.json')))

    def test_round0_selection_reports_the_resolution_flag(self):
        cond, y = self.data(N=40, T=40)
        msgs = []
        a = sc.select_lag_round0(jr.PRNGKey(2), cond, y, 40, self.small_hp(), logger=msgs.append)
        self.assertIsInstance(a['resolution_failure'], bool)
        self.assertEqual(a['resolution_failure'], ls.resolution_failure(a['n_prefix_stage0'], a['n_prefix']))
        self.assertEqual(a['resolution_failure'], any('resolution failure' in m for m in msgs))
        json.dumps(a)                                           # the record is JSON-friendly


class TestLagMaskedModel(unittest.TestCase):
    """Option B: the lag-dropout network at one lag behaves as a T-SNL model of that lag."""

    p, d, L_max = 2, 2, 4

    def setUp(self):
        hp = dict(sc.PRODUCTION_HP, nmade=2, dhidden=8, nhidden=2)
        self.base = sc.build_dropout_candidate(jr.PRNGKey(0), self.L_max, self.p, self.d, hp)
        rng = np.random.default_rng(1)
        self.cond = jnp.asarray(rng.standard_normal((3, self.p)), dtype=jnp.float32)
        self.y = jnp.asarray(rng.standard_normal((3, 14, self.d)), dtype=jnp.float32)
        full = sc.make_rows(self.cond, self.y, self.L_max).reshape(-1, self.p + (self.L_max + 1) * self.d)
        ref = sc.apply_lag_mask(full, jr.randint(jr.PRNGKey(2), (full.shape[0],), 0, self.L_max + 1),
                                self.p, self.L_max, self.d)
        sc.freeze_batch_stats(self.base, ref)

    def view(self, ell):
        return sc.LagMaskedModel(self.base, ell, self.L_max, self.p, self.d)

    def test_loss_equals_the_masked_full_lag_loss(self):
        """loss_fn on lag-ell rows == loss of the network on the full-lag rows masked at ell."""
        for ell in range(self.L_max + 1):
            rows = sc.make_rows(self.cond, self.y, ell).reshape(-1, self.p + (ell + 1) * self.d)
            full = sc.make_rows(self.cond, self.y, self.L_max).reshape(-1, self.p + (self.L_max + 1) * self.d)
            want = sc._nll_per_row(self.base, sc.apply_lag_mask(full, ell, self.p, self.L_max, self.d)).mean()
            np.testing.assert_allclose(self.view(ell).loss_fn(rows), want, rtol=1e-5)

    def test_works_with_the_production_likelihood_functions(self):
        """loglik_fn at lag ell (reshape_emissions, zero padding) gives the sum of the masked row scores."""
        from util.train import loglik_fn
        for ell in (0, 2, self.L_max):
            got = loglik_fn(self.cond[0], self.y[0], self.view(ell), ell)
            full = sc.make_rows(self.cond[:1], self.y[:1], self.L_max).reshape(-1, self.p + (self.L_max + 1) * self.d)
            want = -sc._nll_per_row(self.base, sc.apply_lag_mask(full, ell, self.p, self.L_max, self.d)).sum()
            np.testing.assert_allclose(np.asarray(got).ravel()[0], want, rtol=1e-4)

    def test_generate_uses_only_the_visible_past(self):
        key = jr.PRNGKey(7)
        ell = 2
        cond = jnp.concatenate([self.cond[0], jnp.array([0.5, -0.5, 1.0, 2.0])])[None]    # theta, 2 lags of d=2
        out = self.view(ell).generate(key, 1, cond)
        self.assertEqual(out.shape[-1], self.p + self.L_max + self.L_max * self.d + self.d)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
        keep = sc.lag_mask(jnp.array([ell]), self.L_max).astype(cond.dtype)
        manual = jnp.concatenate([cond[:, :self.p], keep, jnp.zeros((1, (self.L_max - ell) * self.d)),
                                  cond[:, self.p:]], axis=1)
        np.testing.assert_allclose(out, self.base.generate(key, 1, manual))

    def test_lag_out_of_range(self):
        with self.assertRaises(ValueError):
            self.view(self.L_max + 1)
        self.assertEqual(self.view(1).with_lag(3).ell, 3)


class TestLagDropoutMask(unittest.TestCase):

    def test_draws_a_lag_for_each_row_and_advances_the_key(self):
        p, d, Lg = 2, 2, 5
        rows = jnp.asarray(np.random.default_rng(0).standard_normal((512, p + (Lg + 1) * d)), dtype=jnp.float32)
        mask = sc.LagDropoutMask(jr.PRNGKey(0), p, Lg, d)
        a, b = mask(rows), mask(rows)
        self.assertEqual(a.shape, (512, p + Lg + Lg * d + d))
        self.assertFalse(np.array_equal(a, b))                            # new lags at the next call
        ind = np.asarray(a[:, p:p + Lg])
        ell = ind.sum(axis=1).astype(int)                                  # thermometer code: sum = lag
        self.assertEqual(set(np.unique(ell)), set(range(Lg + 1)))          # all lags occur
        np.testing.assert_array_equal(ind, np.asarray(sc.lag_mask(jnp.asarray(ell), Lg)))
        past = np.asarray(a[:, p + Lg:p + Lg + Lg * d]).reshape(-1, Lg, d)
        np.testing.assert_array_equal(past[ind == 0], 0.0)                 # masked observations are zero


class TestReselect(unittest.TestCase):
    """Option B: the lag of a round from its new simulations."""

    def setUp(self):
        rng = np.random.default_rng(0)
        N, T = 30, 40
        y = np.zeros((N, T, 1), dtype=np.float32)
        for t in range(1, T):
            y[:, t, 0] = 0.7 * y[:, t - 1, 0] + rng.standard_normal(N)
        self.cond, self.y = jnp.zeros((N, 1)), jnp.asarray(y)
        hp = dict(sc.PRODUCTION_HP, nmade=2, dhidden=8, nhidden=2, epochs=1)
        self.L_max = 4
        self.model = sc.build_dropout_candidate(jr.PRNGKey(1), self.L_max, 1, 1, hp)

    def run_once(self):
        return sc.reselect_lag(jr.PRNGKey(2), self.model, self.cond[:20], self.y[:20], self.cond[20:], self.y[20:],
                               self.L_max, 10, 40)

    def test_record(self):
        rd = self.run_once()
        self.assertLessEqual(rd['L_star'], self.L_max)
        self.assertEqual(rd['n_scored'], 20)
        self.assertEqual(len(rd['H_hat']), self.L_max + 1)
        self.assertEqual(len(rd['D_hat']), self.L_max)
        self.assertEqual(rd['at_L_max'], rd['L_star'] >= self.L_max)
        self.assertEqual(rd['grid_limited'], rd['grid_flags']['limited'])
        self.assertEqual(rd['resolution_failure'], ls.resolution_failure(rd['n_prefix_stage0'], rd['n_prefix']))
        json.dumps(rd)

    def test_deterministic(self):
        a, b = self.run_once(), self.run_once()
        for k in ('L_star', 'L_tol', 'L_cv', 'H_hat'):
            self.assertEqual(a[k], b[k])

    def test_return_nll_gives_the_scores(self):
        rd, nll = sc.reselect_lag(jr.PRNGKey(2), self.model, self.cond[:20], self.y[:20], self.cond[20:], self.y[20:],
                                  self.L_max, 10, 40, return_nll=True)
        self.assertEqual(nll.shape, (20, self.L_max + 1))
        self.assertEqual(nll.dtype, np.float32)
        np.testing.assert_allclose(np.asarray(rd['H_hat']), nll.mean(axis=0), rtol=1e-5)

    def corrupt_run(self, **kw):
        """A selection whose scores fail: a few trajectories have a huge NLL at the lags above 0."""
        import unittest.mock as mock
        real = sc.heldout_nll_all_lags

        def bad(*a, **k):
            nll = np.array(real(*a, **k))
            nll[:3, 1:] += 1e4 * 4.0 ** np.arange(nll.shape[1] - 1)[None, :]
            return nll
        with mock.patch.object(sc, 'heldout_nll_all_lags', bad):
            return sc.reselect_lag(jr.PRNGKey(2), self.model, self.cond[:20], self.y[:20], self.cond[20:], self.y[20:],
                                   self.L_max, 10, 40, **kw)

    def test_hold_rule_keeps_the_previous_lag_when_the_selection_fails(self):
        rd = self.corrupt_run(L_prev=3)
        self.assertEqual(rd['fit_status'], 'none')
        self.assertFalse(rd['usable'])
        self.assertTrue(rd['held'])
        self.assertEqual((rd['L_star'], rd['L_star_raw']), (3, 0))

    def test_damage_diagnostic_warns_and_does_not_change_the_lag(self):
        rd = self.corrupt_run(L_prev=3, hold=False)
        self.assertEqual(rd['n_damaged'], 3)                               # the 3 corrupted trajectories
        self.assertAlmostEqual(rd['damaged_fraction'], 3 / 20)
        self.assertTrue(rd['damaged_warning'])
        self.assertEqual(rd['L_star'], rd['L_star_raw'])                   # a warning only
        clean = self.run_once()
        self.assertFalse(clean['damaged_warning'])

    def test_hold_rule_can_be_switched_off_and_needs_a_previous_lag(self):
        rd = self.corrupt_run(L_prev=3, hold=False)
        self.assertFalse(rd['held'])
        self.assertEqual(rd['L_star'], rd['L_star_raw'])
        self.assertFalse(self.corrupt_run()['held'])                       # no previous lag: nothing to keep

    def test_hold_rule_does_not_act_on_a_usable_selection(self):
        rd = self.run_once()
        rd2 = sc.reselect_lag(jr.PRNGKey(2), self.model, self.cond[:20], self.y[:20], self.cond[20:], self.y[20:],
                              self.L_max, 10, 40, L_prev=4)
        if rd['fit_status'] != 'none':
            self.assertFalse(rd2['held'])
            self.assertEqual(rd2['L_star'], rd['L_star'])

    def test_non_finite_scores_are_not_usable(self):
        import unittest.mock as mock
        real = sc.heldout_nll_all_lags

        def nan(*a, **k):
            nll = np.array(real(*a, **k))
            nll[0, 2] = np.nan
            return nll
        with mock.patch.object(sc, 'heldout_nll_all_lags', nan):
            rd = sc.reselect_lag(jr.PRNGKey(2), self.model, self.cond[:20], self.y[:20], self.cond[20:], self.y[20:],
                                 self.L_max, 10, 40, L_prev=2)
        self.assertFalse(rd['usable'])
        self.assertEqual((rd['L_star'], rd['held']), (2, True))

    def test_reference_rows_do_not_use_the_scored_trajectories(self):
        rows = sc.reference_rows(jr.PRNGKey(0), self.cond, self.y, self.L_max, n_ref=100)
        self.assertEqual(rows.shape[1], 1 + self.L_max + self.L_max + 1)
        self.assertLessEqual(rows.shape[0], 3 * 40)                       # 3 trajectories of 40 windows


class TestCompileCounter(unittest.TestCase):

    def test_counts_a_new_compilation_only_once(self):
        f = jax.jit(lambda x: x * 3.0 + 1.0)
        n0 = sc.n_compilations()
        f(jnp.ones(7)).block_until_ready()
        n1 = sc.n_compilations()
        f(jnp.zeros(7)).block_until_ready()
        n2 = sc.n_compilations()
        self.assertGreater(n1, n0)
        self.assertEqual(n2, n1)


if __name__ == '__main__':
    unittest.main()
