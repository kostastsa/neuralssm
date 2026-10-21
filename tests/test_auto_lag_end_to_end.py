"""End-to-end runs of small ``lag: auto`` (Option A) and ``lag: ada`` (Option B) experiments through
ExperimentRunner (about 1 to 2 minutes each).

Opt-in, because it trains networks and writes a trial directory:
    NEURALSSM_SLOW=1 python -m unittest tests.test_auto_lag_end_to_end

The result directory (T = 40, 60 simulations) is removed afterwards, but only if the test created it.
"""

import json
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'neuralssm'))

import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

import experiment_descriptor as ed  # noqa: E402
import experiment_runner as er  # noqa: E402
import util.io  # noqa: E402
from inference import lag_selection as ls  # noqa: E402

SIM = ed.LGSSM_Descriptor().create_desc(1, 1, 40, None, 'd4')
MODEL_ARGS = (3, 16, 3, 'relu', False, True, False, False, 2, 1e-4)
FILES = ('lag_selection.json', 'lag_selection.png', 'lag_selection_nll.npz', 'model.pkl', 'mll.pkl', 'posterior.pkl')


def make(lag, n_rounds=2, **kw):
    inf = ed.TSNL_Descriptor().create_desc(MODEL_ARGS, 60, n_rounds, lag, 1.0, 'all', 'rwm', 100, **kw)
    return ed.ExperimentDescriptor(ed.ExperimentDescriptor().create_desc(SIM, inf))


@unittest.skipUnless(os.environ.get('NEURALSSM_SLOW') == '1', 'set NEURALSSM_SLOW=1 to run')
class TestAutoAdaEndToEnd(unittest.TestCase):

    def run_trial(self, desc):
        """Runs trial 1 and returns (trial directory, cleanup function)."""
        runner = er.ExperimentRunner(desc)
        sim_root = os.path.join(runner.exp_dir.split('nde')[0])
        existed = os.path.exists(sim_root)
        runner.run(trial=1, sample_gt=False, key=jr.PRNGKey(0), seed=0, n_post_samples=20)
        d = os.path.join(runner.exp_dir, 'sample_gt_False', '1')
        for f in FILES:
            self.assertTrue(os.path.exists(os.path.join(d, f)), f)
        return d, (lambda: None if existed else shutil.rmtree(sim_root, ignore_errors=True))

    def check_common(self, desc, d, option):
        sel = json.load(open(os.path.join(d, 'lag_selection.json')))
        self.assertEqual(sel['option'], option)
        self.assertLessEqual(sel['L_star'], sel['L_grid'])
        self.assertLessEqual(sel['L_grid'], sel['L_cap'])
        self.assertEqual(sel['L_cap'], 10)                 # T // 4 with T = 40
        self.assertEqual(sel['n_trajectories'], 60)
        self.assertIsInstance(sel['resolution_failure'], bool)
        self.assertIsInstance(sel['damaged_warning'], bool)
        self.assertTrue(0.0 <= sel['damaged_fraction'] <= 1.0)
        self.assertEqual(sel['resolution_failure'], ls.resolution_failure(sel['n_prefix_stage0'], sel['n_prefix']))
        self.assertNotIn('variant', sel)
        self.assertEqual(len(sel['rounds']), desc.inf.n_rounds)
        for rd in sel['rounds']:
            for phase in ('simulate', 'train', 'mcmc'):
                self.assertGreaterEqual(rd['seconds'][phase], 0.0)
                self.assertGreaterEqual(rd['compilations'][phase], 0)
        return sel

    def test_auto_run_writes_selection_and_uses_it(self):
        desc = make('auto')
        d, cleanup = self.run_trial(desc)
        try:
            sel = self.check_common(desc, d, 'A')
            self.assertEqual(ed.resolve_lag(desc.inf, d), sel['L_star'])
            graphdef, state = util.io.load(os.path.join(d, 'model'))
            model = nnx.merge(graphdef, state)
            self.assertIs(ed.resolve_model(desc.inf, d, model), model)           # plain T-SNL network
            self.assertEqual(model.dcond, 1 + sel['L_star'])                      # p = 1, d = 1
        finally:
            cleanup()

    def test_ada_run_reselects_in_every_round_with_one_network_shape(self):
        desc = make('ada', n_rounds=3)
        d, cleanup = self.run_trial(desc)
        try:
            sel = self.check_common(desc, d, 'B')
            L_max = sel['L_max']
            self.assertEqual(L_max, ls.ada_lag_max(sel['L_star'], sel['L_cap']))
            self.assertEqual(len(sel['L_star_r']), 3)
            self.assertEqual(len(sel['L_star_raw_r']), 3)
            self.assertEqual(sel['damaged_rounds'], [rd['round'] for rd in sel['rounds'] if rd['damaged_warning']])
            self.assertTrue(all('damaged_fraction' in rd for rd in sel['rounds']))
            for r in sel['held_rounds']:                                       # a held round keeps the previous lag
                self.assertEqual(sel['L_star_r'][r], sel['L_star_r'][r - 1])
            scores = np.load(os.path.join(d, 'lag_selection_nll.npz'))
            self.assertEqual(sorted(scores.files), ['round0', 'round1', 'round2'])
            self.assertEqual(scores['round1'].shape, (60, L_max + 1))
            self.assertEqual(sel['L_star_r'][0], sel['L_star'])
            self.assertEqual(sel['L_final'], sel['L_star_r'][-1])
            self.assertEqual(ed.resolve_lag(desc.inf, d), sel['L_final'])
            self.assertTrue(all(0 <= L <= L_max for L in sel['L_star_r']))
            self.assertIsInstance(sel['capped'], bool)
            self.assertEqual(sel['capped'], bool(sel['capped_rounds']))
            for r, rd in enumerate(sel['rounds']):
                self.assertEqual(rd['round'], r)
                self.assertEqual(rd['lag'], sel['L_star_r'][r])
                self.assertIn('select', rd['seconds'])
                if r >= 1:
                    self.assertEqual(rd['n_scored'], 60)          # the new simulations of the round
                    self.assertEqual(len(rd['H_hat']), L_max + 1)
            # the saved network is the lag-dropout network of lag L_max, in the loader's format
            graphdef, state = util.io.load(os.path.join(d, 'model'))
            base = nnx.merge(graphdef, state)
            self.assertEqual(base.dcond, 1 + L_max + L_max)                       # p + L_max + L_max * d
            view = ed.resolve_model(desc.inf, d, base)
            self.assertEqual((view.ell, view.L_max), (sel['L_final'], L_max))
        finally:
            cleanup()

    def test_ada_options_of_the_learner(self):
        """lag_opts: cap L_max, factor of the warm rounds, hold. None of them is in the descriptor."""
        import simulators.lgssm as lgssm
        from inference import lag_scoring as sc
        from inference.nde import SequentialNeuralLikelihood
        from maf.density_models import MAF

        class Log:
            def __init__(self):
                self.lines = []

            def write(self, s):
                self.lines.append(s)

        setup = lgssm.setup(1, 1, 0, ['d4'])
        hp = dict(sc.PRODUCTION_HP, nmade=2, dhidden=8, nhidden=2, epochs=1)
        lag_opts = {'tol': 0.1, 'hp': hp, 'L_max': 3, 'ld_epochs_factor_warm': 1, 'hold': True}
        learner = SequentialNeuralLikelihood(setup['props'], setup['ssm'], lag='ada', lag_opts=lag_opts)
        factory = lambda n_cond: MAF(din=1, nmade=2, dhidden=8, nhidden=2, act_fun='relu', dcond=n_cond,
                                     rngs=nnx.Rngs(0), reverse=True)
        obs = jnp.asarray(np.random.default_rng(0).standard_normal((40, 1)), dtype=jnp.float32)
        log = Log()
        model, _ = learner.learn_likelihood(
            jr.PRNGKey(3), obs, None, num_rounds=3, num_timesteps=40, num_samples=40, num_posterior_samples=10,
            train_on='all', sampler='rwm', mcmc_steps=60, logger=log, num_epochs=1, learning_rate=1e-4,
            model_factory=factory)
        sel = learner.lag_selection
        self.assertEqual(sel['L_max'], 3)                                   # the option replaces 2 * L*_0
        self.assertEqual(sel['L_max_source'], 'lag_opts')
        self.assertEqual(sel['ld_epochs_factor_warm'], 1)
        self.assertEqual(model.base.dcond, 1 + 3 + 3)
        self.assertEqual(sorted(learner.lag_scores), [0, 1, 2])
        self.assertTrue(all(0 <= L <= 3 for L in sel['L_star_r']))
        self.assertEqual(sel['L_final'], learner.lag)

    def test_snl_descriptor_and_integer_lag_still_run_without_selection_record(self):
        desc = make(3)
        d, cleanup = None, None
        runner = er.ExperimentRunner(desc)
        sim_root = os.path.join(runner.exp_dir.split('nde')[0])
        existed = os.path.exists(sim_root)
        try:
            runner.run(trial=1, sample_gt=False, key=jr.PRNGKey(0), seed=0, n_post_samples=20)
            d = os.path.join(runner.exp_dir, 'sample_gt_False', '1')
            self.assertFalse(os.path.exists(os.path.join(d, 'lag_selection.json')))
            self.assertEqual(ed.resolve_lag(desc.inf, d), 3)
        finally:
            if not existed:
                shutil.rmtree(sim_root, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
