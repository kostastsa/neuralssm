"""Regression test: an integer lag gives the same result before and after the I1 changes.

The original ``inference/nde.py`` (git HEAD when the I1 work started, commit 0545ada) and the current
one are run with the same keys, data and initial network. The posterior sample, the trained
parameters and the losses must be identical, so existing experiments reproduce unchanged.

Run from the repository root with the jaxenv environment active:
    python -m unittest tests.test_nde_integer_lag
"""

import os
import subprocess
import sys
import types
import unittest

import numpy as np

REPO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(REPO, 'src', 'neuralssm'))

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from flax import nnx  # noqa: E402

import simulators.lgssm as lgssm  # noqa: E402
from maf.density_models import MAF  # noqa: E402
import inference.nde as nde_new  # noqa: E402

BASE_COMMIT = '0545ada'


def load_original_nde():
    src = subprocess.run(['git', 'show', f'{BASE_COMMIT}:src/neuralssm/inference/nde.py'],
                         capture_output=True, text=True, cwd=REPO)
    if src.returncode != 0:
        raise unittest.SkipTest(f'cannot read nde.py at {BASE_COMMIT}: {src.stderr.strip()}')
    mod = types.ModuleType('nde_original')
    exec(compile(src.stdout, 'nde_original', 'exec'), mod.__dict__)
    return mod


class _Log:
    def write(self, s):
        pass


def run(module, lag):
    setup = lgssm.setup(1, 1, 0, ['d4'])
    props, ssm = setup['props'], setup['ssm']
    model = MAF(din=1, nmade=2, dhidden=8, nhidden=2, act_fun='relu', dcond=lag + 1,
                rngs=nnx.Rngs(0), reverse=True)
    learner = module.SequentialNeuralLikelihood(props, ssm, lag=lag)
    obs = jnp.asarray(np.random.default_rng(0).standard_normal((30, 1)), dtype=jnp.float32)
    model, (post, post_cond) = learner.learn_likelihood(
        jr.PRNGKey(3), obs, model, num_rounds=2, num_timesteps=30, num_samples=24,
        num_posterior_samples=10, train_on='all', sampler='rwm', mcmc_steps=40,
        logger=_Log(), num_epochs=1, learning_rate=1e-4)
    params = [np.asarray(v.value) for _, v in nnx.to_flat_state(nnx.state(model, nnx.Param))]
    return np.asarray(post_cond), params, [float(x) for x in learner.losses], learner.lag


class TestIntegerLagUnchanged(unittest.TestCase):

    def test_same_as_original(self):
        original = load_original_nde()
        for lag in (0, 3):
            old = run(original, lag)
            new = run(nde_new, lag)
            np.testing.assert_array_equal(old[0], new[0])
            self.assertEqual(len(old[1]), len(new[1]))
            for a, b in zip(old[1], new[1]):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(old[2], new[2])
            self.assertEqual(old[3], new[3])
            self.assertEqual(new[3], lag)


if __name__ == '__main__':
    unittest.main()
