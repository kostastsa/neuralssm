# This module is adapted from the repository ([https://github.com/gpapamak/snl.git]),
# authored by George Papamakarios under the MIT License
import jax 
from jax import numpy as jnp 
from jax import random as jr # type: ignore
import orbax.checkpoint as ocp

import os
import shutil
import gc

import util.plot
import util.io
import util.math
from util.param import sample_prior, to_train_array
from util.misc import kde_error

from flax import nnx
from maf.density_models import MAF

import experiment_descriptor as ed
import misc

import inspect

class ExperimentRunner:
    """
    Runs experiments on likelihood-free inference of simulator models.
    """

    def __init__(self, exp_desc):
        """
        :param exp_desc: an experiment descriptor object
        """

        assert isinstance(exp_desc, ed.ExperimentDescriptor)

        self.exp_desc = exp_desc
        self.exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
        self.sim = misc.get_simulator(exp_desc.sim)

    def run(self, trial=0, sample_gt=False, n_post_samples=1000, key=jr.PRNGKey(0), seed=0):
        """
        Runs the experiment.
        :param rng: random number generator to use
        """

        print('\n' + '-' * 80)
        print('RUNNING EXPERIMENT, TRIAL {0}:\n'.format(trial))
        print(self.exp_desc.pprint())

        exp_dir = os.path.join(self.exp_dir, str(trial))

        if os.path.exists(exp_dir):
            raise misc.AlreadyExistingExperiment(self.exp_desc)

        util.io.make_folder(exp_dir)

        try:
            if isinstance(self.exp_desc.inf, ed.ABC_Descriptor):
                self._run_abc(exp_dir, sample_gt, key, seed)

            elif isinstance(self.exp_desc.inf, ed.PRT_MCMC_Descriptor):
                self._run_prt_mcmc(exp_dir, sample_gt, n_post_samples, key, seed)

            elif isinstance(self.exp_desc.inf, ed.SNL_Descriptor):
                self._run_snl(exp_dir, sample_gt, n_post_samples, key, seed)

            elif isinstance(self.exp_desc.inf, ed.TSNL_Descriptor):
                self._run_tsnl(exp_dir, sample_gt, n_post_samples, key, seed)

            else:
                raise TypeError('unknown inference descriptor')

        except:
            shutil.rmtree(exp_dir)
            raise

    def _run_abc(self, exp_dir, sample_gt, key, seed):
        """
        Runs the ABC experiments.
        """

        import inference.abc as abc

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        param_info = self.sim.get_param_info()
        sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)
        ssm = sim_setup['ssm']
        props = sim_setup['props']
        inputs = sim_setup['inputs']
        
        if sample_gt:
            key, subkey = jr.split(key)
            true_ps = sample_prior(key, props)[0]
            true_cps = to_train_array(true_ps, props)
            true_ps.from_unconstrained(props)
            _, obs_ys = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs) 
        else:
            true_ps, obs_ys = self.sim.get_ground_truth()

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            if isinstance(inf_desc, ed.SMC_ABC_Descriptor):

                abc_runner = abc.SMC(props, ssm)

                results = abc_runner.run(
                    key,
                    obs_ys,
                    eps_init=inf_desc.eps_init,
                    eps_last=inf_desc.eps_last,
                    eps_decay=inf_desc.eps_decay,
                    num_particles=inf_desc.n_samples,
                    logger=logger
                )

            else:
                raise TypeError('unknown ABC algorithm')
            
            samples, _, _, _, counts, _, _ = results
            error = kde_error(samples[-1], true_cps)
            num_simulations = counts * sim_desc.num_timesteps
            
            util.io.save(([true_ps, true_cps], obs_ys), os.path.join(exp_dir, 'gt'))
            util.io.save((error, num_simulations), os.path.join(exp_dir, 'error'))
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save(abc_runner.time_all_rounds, os.path.join(exp_dir, 'time_all_rounds'))
            util.io.save_txt(str(seed), os.path.join(exp_dir, 'seed.txt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))
            util.io.save_txt(inspect.getsource(param_info), os.path.join(exp_dir, 'param_info.txt'))

            del results
            abc_runner = None
            jax.clear_backends()
            gc.collect()


    def _run_prt_mcmc(self, exp_dir, sample_gt, n_post_samples, key, seed):
        """
        Runs the ABC experiments.
        """

        import inference.mcmc as mcmc

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        param_info = self.sim.get_param_info()
        sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)
        ssm = sim_setup['ssm']
        props = sim_setup['props']
        inputs = sim_setup['inputs']
        if sample_gt:
            key, subkey = jr.split(key)
            true_ps = sample_prior(key, props)[0]
            true_cps = to_train_array(true_ps, props)
            true_ps.from_unconstrained(props)
            _, obs_ys = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs) 

        else:
            true_ps, obs_ys = self.sim.get_ground_truth()

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            if isinstance(inf_desc, ed.PRT_MCMC_Descriptor):

                mcmc_runner = mcmc.BPF_MCMC(props, ssm)

                results = mcmc_runner.run(
                    key,
                    obs_ys,
                    num_prt = inf_desc.num_prt,
                    num_posterior_samples=n_post_samples,
                    mcmc_steps = inf_desc.mcmc_steps,
                    num_iters = inf_desc.num_iters,
                    logger=logger
                    )
                
            else:

                raise TypeError('unknown PRT_MCMC algorithm')

            error = kde_error(results[0], true_cps)
            num_simulations = inf_desc.num_prt * sim_desc.num_timesteps * inf_desc.mcmc_steps * inf_desc.num_iters

            util.io.save(([true_ps, true_cps], obs_ys), os.path.join(exp_dir, 'gt'))
            util.io.save((error, num_simulations), os.path.join(exp_dir, 'error'))
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save_txt(str(mcmc_runner.time), os.path.join(exp_dir, 'time.txt'))
            util.io.save_txt(str(seed), os.path.join(exp_dir, 'seed.txt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))
            util.io.save_txt(inspect.getsource(param_info), os.path.join(exp_dir, 'param_info.txt'))

            del results
            mcmc_runner = None
            jax.clear_backends()
            gc.collect()


    def _run_snl(self, exp_dir, sample_gt, n_post_samples, key, seed):
        """
        Runs the likelihood learner with MCMC.
        """

        import inference.nde as nde

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        param_info = self.sim.get_param_info()
        sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)
        ssm = sim_setup['ssm']
        props = sim_setup['props']
        inputs = sim_setup['inputs']
        key, subkey = jr.split(key)
        xparam = sample_prior(subkey, props)[0]

        if sample_gt:
            key, subkey = jr.split(key)
            true_ps = sample_prior(key, props)[0]
            true_cps = to_train_array(true_ps, props)
            true_ps.from_unconstrained(props)
            _, obs_ys = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs) 
        else:
            true_ps, obs_ys = self.sim.get_ground_truth()

        n_inputs = sim_desc.emission_dim * sim_desc.num_timesteps
        n_cond = to_train_array(xparam, props).shape[0]
        key_model, key_learner = jr.split(key)
        model = self._create_model(n_inputs, n_cond, key_model)
        learner = nde.SequentialNeuralLikelihood(props, ssm, lag=-1)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            model, (posterior_sample, posterior_cond_sample) = learner.learn_likelihood(
                key=key_learner,
                observations=obs_ys,
                model=model,
                num_rounds=inf_desc.n_rounds,
                num_timesteps=sim_desc.num_timesteps,
                num_samples=inf_desc.n_samples,
                num_posterior_samples=n_post_samples,
                train_on=inf_desc.train_on,
                mcmc_steps=inf_desc.mcmc_steps,
                logger=logger
            )

            error = kde_error(posterior_cond_sample, true_cps)
            num_simulations = inf_desc.n_rounds * inf_desc.n_samples * sim_desc.num_timesteps

            util.io.save(([true_ps, true_cps], obs_ys), os.path.join(exp_dir, 'gt'))
            util.io.save((error, num_simulations), os.path.join(exp_dir, 'error'))
            util.io.save(learner.all_params, os.path.join(exp_dir, 'all_params'))
            util.io.save(learner.all_emissions, os.path.join(exp_dir, 'all_emissions'))
            util.io.save(learner.all_cond_params, os.path.join(exp_dir, 'all_cond_params'))
            util.io.save((posterior_sample, posterior_cond_sample), os.path.join(exp_dir, 'posterior'))
            util.io.save(learner.time_all_rounds, os.path.join(exp_dir, 'time_all_rounds'))
            util.io.save(learner.all_dists.reshape(inf_desc.n_rounds, inf_desc.n_samples), os.path.join(exp_dir, 'all_dists'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))
            util.io.save_txt(str(seed), os.path.join(exp_dir, 'seed.txt'))
            util.io.save_txt(inspect.getsource(param_info), os.path.join(exp_dir, 'param_info.txt'))

            del posterior_sample, posterior_cond_sample
            learner = None
            model = None
            jax.clear_backends()
            gc.collect()

    def _run_tsnl(self, exp_dir, sample_gt, n_post_samples, key, seed):
        """
        Runs the likelihood learner with MCMC.
        """

        import inference.nde as nde

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        param_info = self.sim.get_param_info()
        sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)
        ssm = sim_setup['ssm']
        props = sim_setup['props']
        inputs = sim_setup['inputs']

        key, subkey = jr.split(key)
        xparam = sample_prior(subkey, props)[0]

        num_tiles=None
        subsample=False

        if inf_desc.subsample < 1.0:

            num_tiles = int(inf_desc.subsample * sim_desc.num_timesteps)
            subsample = True

        if sample_gt:

            key, subkey = jr.split(key)
            true_ps = sample_prior(key, props)[0]
            true_cps = to_train_array(true_ps, props)
            true_ps.from_unconstrained(props)
            true_ps, obs_ys = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs) 

        else:

            true_ps, obs_ys = self.sim.get_ground_truth()

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            n_inputs = sim_desc.emission_dim
            n_params = to_train_array(xparam, props).shape[0]
            n_cond = inf_desc.lag * sim_desc.emission_dim + n_params
            key_model, key_learner = jr.split(key)
            model = self._create_model(n_inputs, n_cond, key_model)
            learner = nde.SequentialNeuralLikelihood(props, ssm, lag=inf_desc.lag)

            _, (posterior_sample, posterior_cond_sample) = learner.learn_likelihood(
                key=key_learner,
                observations=obs_ys,
                model=model,
                num_rounds=inf_desc.n_rounds,
                num_timesteps=sim_desc.num_timesteps,
                num_samples=inf_desc.n_samples,
                num_posterior_samples=n_post_samples,
                train_on=inf_desc.train_on,
                mcmc_steps=inf_desc.mcmc_steps,
                logger=logger,
                num_tiles = num_tiles,
                subsample = subsample
            )
            
            error = kde_error(posterior_cond_sample, true_cps)
            num_simulations = inf_desc.n_rounds * inf_desc.n_samples * sim_desc.num_timesteps

            util.io.save(([true_ps, true_cps], obs_ys), os.path.join(exp_dir, 'gt'))
            util.io.save((error, num_simulations), os.path.join(exp_dir, 'error'))
            util.io.save(learner.all_params, os.path.join(exp_dir, 'all_params'))
            util.io.save(learner.all_emissions, os.path.join(exp_dir, 'all_emissions'))
            util.io.save(learner.all_cond_params, os.path.join(exp_dir, 'all_cond_params'))
            util.io.save(learner.time_all_rounds, os.path.join(exp_dir, 'time_all_rounds'))
            util.io.save(learner.all_dists.reshape(inf_desc.n_rounds, inf_desc.n_samples), os.path.join(exp_dir, 'all_dists'))
            util.io.save((posterior_sample, posterior_cond_sample), os.path.join(exp_dir, 'posterior'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))
            util.io.save_txt(str(seed), os.path.join(exp_dir, 'seed.txt'))
            util.io.save_txt(inspect.getsource(param_info), os.path.join(exp_dir, 'param_info.txt'))

            del posterior_sample, posterior_cond_sample
            learner = None
            model = None
            gc.collect()
            jax.clear_backends()

    def _create_model(self, n_inputs, n_cond, rng):
        """
        Given input and output sizes, creates and returns the model for the NDE experiments.
        """

        model_desc = self.exp_desc.inf.model

        if isinstance(model_desc, ed.MAF_Descriptor):

            return MAF(
                din = n_inputs,
                nmade = model_desc.nmades,
                dhidden=model_desc.dhidden,
                nhidden=model_desc.nhidden,
                act_fun=model_desc.act_fun,
                dcond = n_cond,
                rngs = nnx.Rngs(rng),
                random_order = model_desc.random_order,
                reverse=model_desc.reverse,
                dropout=model_desc.dropout,
                batch_norm=model_desc.batch_norm
            )

        else:
            raise TypeError('unknown model descriptor')
