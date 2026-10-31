"""Regression test for decision P-12: the T-SNL log density uses the sum over the T windows.

``MAF.loss_fn`` returns the mean NLL of a batch (the training loss). Before the fix, ``util.train.logdensity_fn``
used this mean as the log-likelihood, so the MCMC sampled the likelihood to the power 1/T. The test checks
that logdensity_fn - log_prior equals the sum of the per-window log densities and equals ``loglik_fn``.

Run from the repository root with the jaxenv environment active:
    python -m unittest tests.test_tsnl_logdensity
"""

import os
import sys
import unittest

import numpy as np

REPO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(REPO, 'src', 'neuralssm'))

import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

import simulators.lgssm as lgssm  # noqa: E402
from maf.density_models import MAF  # noqa: E402
from util.param import log_prior  # noqa: E402
from util.train import logdensity_fn, loglik_fn, reshape_emissions  # noqa: E402

T = 30


def setup(lag, din):
    props = lgssm.setup(1, 1, 0, ['d4'])['props']
    model = MAF(din=din, nmade=2, dhidden=8, nhidden=2, act_fun='relu', dcond=(lag + 1) if lag >= 0 else 1,
                rngs=nnx.Rngs(0), reverse=True)
    model.eval()   # the MCMC uses the model in evaluation mode (eval_model in inference/nde.py)
    obs = jnp.asarray(np.random.default_rng(0).standard_normal((T, 1)), dtype=jnp.float32)
    return props, model, obs


class TestTSNLLogDensity(unittest.TestCase):

    def test_tsnl_sum_over_windows(self):
        for lag in (0, 1, 3):
            props, model, obs = setup(lag, din=1)
            cp = jnp.array([-0.7], dtype=jnp.float32)
            ll = float(logdensity_fn(cp, model, obs, props, lag) - log_prior(cp, props))
            rows = jnp.concatenate([jnp.tile(cp, (T, 1)), reshape_emissions(obs, lag)], axis=1)
            out, log_det = model(rows)                      # per-row output and log-determinant, one batch call
            u = out[:, model.dcond:]
            per_row_nll = jnp.sum(u ** 2, axis=1) / 2 + 0.5 * u.shape[1] * jnp.log(2 * jnp.pi) - log_det
            per_window = [-float(v) for v in per_row_nll]
            self.assertAlmostEqual(float(model.loss_fn(rows)), -float(np.mean(per_window)), places=4)
            self.assertEqual(rows.shape[0], T)
            self.assertAlmostEqual(ll, sum(per_window), places=2)
            # The old (wrong) value was the mean: it differs from the sum by a factor T.
            self.assertGreater(abs(ll - np.mean(per_window)), 1.0)
            self.assertAlmostEqual(ll, float(loglik_fn(cp, obs, model, lag).squeeze()), places=3)

    def test_snl_unchanged(self):
        props, model, obs = setup(-1, din=T)
        cp = jnp.array([-0.7], dtype=jnp.float32)
        ll = float(logdensity_fn(cp, model, obs, props, -1) - log_prior(cp, props))
        row = jnp.concatenate([cp[None], obs.flatten()[None]], axis=1)
        self.assertAlmostEqual(ll, -float(model.loss_fn(row)), places=4)


if __name__ == '__main__':
    unittest.main()
