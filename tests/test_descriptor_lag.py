"""Tests for the lag field of the T-SNL descriptor: integer, auto, ada (request I1).

Run from the repository root with the jaxenv environment active:
    python -m unittest tests.test_descriptor_lag
"""

import json
import os
import sys
import tempfile
import unittest

import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'neuralssm'))

import experiment_descriptor as ed  # noqa: E402
import util.misc  # noqa: E402

SIM = ed.LGSSM_Descriptor().create_desc(1, 1, 40, None, 'd4')
MODEL_ARGS = (5, 32, 5, 'relu', False, True, False, False, 20, 1e-4)


def make(lag, **kw):
    inf = ed.TSNL_Descriptor().create_desc(MODEL_ARGS, 100, 3, lag, 1.0, 'best', 'rwm', 500, **kw)
    return ed.ExperimentDescriptor().create_desc(SIM, inf)


def inf_text(lag, **kw):
    """The inference part only, whitespace removed, as ``TSNL_Descriptor.parse`` receives it."""
    text = make(lag, **kw)
    return util.misc.remove_whitespace(text).split('inf:')[1][:-1]


class TestIntegerLag(unittest.TestCase):

    def test_parse_and_path_unchanged(self):
        d = ed.ExperimentDescriptor(make(15))
        inf = d.inf
        self.assertEqual(inf.lag, 15)
        self.assertIsNone(inf.lag_tol)
        # the directory formula of the original code
        expected = os.path.join('nde', 'tsnl', 'samples_100_rounds_3_lag_15_subsample_1.0_train_on_best_'
                                'sampler_rwm_mcmc_steps_500', inf.model.get_id())
        self.assertEqual(inf.get_dir(), expected)

    def test_pprint_has_no_new_fields(self):
        inf = ed.ExperimentDescriptor(make(15)).inf
        self.assertNotIn('lag_tol', inf.pprint())

    def test_int_rejects_tol(self):
        with self.assertRaises(ValueError):
            make(15, lag_tol=0.1)
        text = inf_text('auto').replace('lag:auto', 'lag:15')
        with self.assertRaises(ed.ParseError):
            ed.TSNL_Descriptor().parse(text)


class TestAutoAda(unittest.TestCase):

    def test_auto_defaults(self):
        inf = ed.ExperimentDescriptor(make('auto')).inf
        self.assertEqual((inf.lag, inf.lag_tol), ('auto', 0.1))
        self.assertTrue(inf.lag_is_auto())
        self.assertIn('lag_auto_tol_0.1_subsample_1.0', inf.get_dir())

    def test_ada_defaults(self):
        inf = ed.ExperimentDescriptor(make('ada')).inf
        self.assertEqual((inf.lag, inf.lag_tol), ('ada', 0.1))
        self.assertTrue(inf.lag_is_auto())
        self.assertIn('lag_ada_tol_0.1_subsample_1.0', inf.get_dir())

    def test_explicit_tol(self):
        for lag, tol in (('auto', 0.05), ('ada', 0.2)):
            inf = ed.ExperimentDescriptor(make(lag, lag_tol=tol)).inf
            self.assertEqual(inf.lag_tol, tol)

    def test_tol_may_be_omitted_in_hand_written_text(self):
        text = inf_text('auto').replace(',lag_tol:0.1', '')
        self.assertNotIn('lag_tol', text)
        d = ed.TSNL_Descriptor()
        d.parse(text)
        self.assertEqual((d.lag, d.lag_tol), ('auto', 0.1))

    def test_variant_field_is_gone(self):
        """The lag-dropout network is the only scheme (decision D2): the old field does not parse."""
        text = inf_text('auto').replace('lag_tol:0.1,', 'lag_tol:0.1,lag_variant:perlag,')
        with self.assertRaises(ed.ParseError):
            ed.TSNL_Descriptor().parse(text)
        with self.assertRaises(TypeError):
            make('auto', lag_variant='dropout')

    def test_invalid_tol(self):
        for tol in (0.0, 1.0, 2.0):
            with self.assertRaises(ed.ParseError):
                ed.TSNL_Descriptor().parse(inf_text('auto', lag_tol=tol))

    def test_round_trip_through_pprint(self):
        for lag, kw in [(15, {}), ('auto', {}), ('ada', dict(lag_tol=0.2))]:
            inf = ed.ExperimentDescriptor(make(lag, **kw)).inf
            again = ed.TSNL_Descriptor(inf.pprint())
            self.assertEqual((again.lag, again.lag_tol, again.n_samples, again.mcmc_steps),
                             (inf.lag, inf.lag_tol, inf.n_samples, inf.mcmc_steps))
            self.assertEqual(again.get_dir(), inf.get_dir())

    def test_paths_are_distinct(self):
        dirs = {ed.ExperimentDescriptor(make(lag, **kw)).inf.get_dir()
                for lag, kw in [(15, {}), ('auto', {}), ('auto', dict(lag_tol=0.05)), ('ada', {}),
                                ('ada', dict(lag_tol=0.05))]}
        self.assertEqual(len(dirs), 5)


class TestResolveLag(unittest.TestCase):

    def test_integer_needs_no_file(self):
        inf = ed.ExperimentDescriptor(make(15)).inf
        self.assertEqual(ed.resolve_lag(inf, '/nonexistent'), 15)

    def test_auto_reads_json(self):
        inf = ed.ExperimentDescriptor(make('auto')).inf
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'lag_selection.json'), 'w') as f:
                json.dump({'L_star': 7}, f)
            self.assertEqual(ed.resolve_lag(inf, d), 7)

    def test_ada_prefers_final(self):
        inf = ed.ExperimentDescriptor(make('ada')).inf
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'lag_selection.json'), 'w') as f:
                json.dump({'L_star': 7, 'L_final': 5}, f)
            self.assertEqual(ed.resolve_lag(inf, d), 5)

    def test_snl_descriptor_passes_through(self):
        self.assertEqual(ed.resolve_lag(ed.SNL_Descriptor(), '/nonexistent'), -1)


class TestResolveModel(unittest.TestCase):

    def test_integer_and_auto_return_the_model_itself(self):
        sentinel = object()
        for lag in (15, 'auto'):
            inf = ed.ExperimentDescriptor(make(lag)).inf
            self.assertIs(ed.resolve_model(inf, '/nonexistent', sentinel), sentinel)

    def test_ada_wraps_the_network_at_the_last_lag(self):
        from inference import lag_scoring as sc
        inf = ed.ExperimentDescriptor(make('ada')).inf
        base = sc.build_dropout_candidate(jax.random.PRNGKey(0), 6, 1, 1)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'lag_selection.json'), 'w') as f:
                json.dump({'L_star': 4, 'L_final': 3, 'L_max': 6, 'n_params': 1, 'emission_dim': 1}, f)
            view = ed.resolve_model(inf, d, base)
        self.assertIsInstance(view, sc.LagMaskedModel)
        self.assertIs(view.base, base)
        self.assertEqual((view.ell, view.L_max), (3, 6))


if __name__ == '__main__':
    unittest.main()
