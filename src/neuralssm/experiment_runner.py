# This module is adapted from the repository ([https://github.com/gpapamak/snl.git]),
# authored by George Papamakarios under the MIT License
import jax 
from jax import random as jr # type: ignore
import os
import gc
import util.plot
import util.io
import util.math
from util.param import sample_prior, to_train_array, get_unravel_fn, tree_from_params, join_trees, params_from_tree
from util.train import marg_loglik
from flax import nnx
from maf.density_models import MAF
import experiment_descriptor as ed
import misc
import time
import inspect

import matplotlib.pyplot as plt
import matplotlib_inline # type: ignore
import scienceplots # type: ignore

plt.style.use(['science', 'ieee'])
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

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

    def run(self, trial=0, sample_gt=False, plot_sims=False, n_post_samples=1000, key=jr.PRNGKey(0), seed=0):
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
                self._run_abc(exp_dir, sample_gt, plot_sims, key, seed)

            elif isinstance(self.exp_desc.inf, ed.PRT_MCMC_Descriptor):
                self._run_prt_mcmc(exp_dir, sample_gt, plot_sims, n_post_samples, key, seed)

            elif isinstance(self.exp_desc.inf, ed.SNL_Descriptor):
                self._run_snl(exp_dir, sample_gt, plot_sims, n_post_samples, key, seed)

            elif isinstance(self.exp_desc.inf, ed.TSNL_Descriptor):
                self._run_tsnl(exp_dir, sample_gt, plot_sims, n_post_samples, key, seed)

            else:
                raise TypeError('unknown inference descriptor')

        except:

            print('EXPERIMENT FAILED')

            raise

    def _run_abc(self, exp_dir, sample_gt, plot_sims, key, seed):
        """
        Runs the ABC experiments.
        """

        import inference.abc as abc

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        param_info = self.sim.get_param_info()

        if hasattr(sim_desc, 'dt_obs'):
            
            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars, sim_desc.dt_obs)

        else:

            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)

        ssm = sim_setup['ssm']
        props = sim_setup['props']
        inputs = sim_setup['inputs']
        
        if sample_gt:

            key, subkey = jr.split(key)
            true_ps = sample_prior(key, props)[0]
            true_cps = to_train_array(true_ps, props)
            true_ps.from_unconstrained(props)
            states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs)

        else:
            
            true_ps, observations = self.sim.get_ground_truth()

        if plot_sims: 

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(states, label='States')
            ax.plot(observations, label='Observations')
            ax.legend()
            ax.set_title('Observations')
            plt.savefig(os.path.join(exp_dir, 'observations.png'))
            plt.close(fig)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            if isinstance(inf_desc, ed.SMC_ABC_Descriptor):

                abc_runner = abc.SMC(props, ssm)

                results = abc_runner.run(
                    key,
                    observations,
                    num_particles=inf_desc.n_samples,
                    qmax=inf_desc.qmax,
                    sigma=inf_desc.sigma,
                    logger=logger
                )

            else:

                raise TypeError('unknown ABC algorithm')
            
            util.io.save(([true_ps, true_cps], observations), os.path.join(exp_dir, 'gt'))
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save(abc_runner.time_all_rounds, os.path.join(exp_dir, 'time_all_rounds'))
            util.io.save_txt(str(seed), os.path.join(exp_dir, 'seed.txt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))
            util.io.save_txt(inspect.getsource(param_info), os.path.join(exp_dir, 'param_info.txt'))

            del results
            abc_runner = None
            jax.clear_backends()
            gc.collect()


    def _run_prt_mcmc(self, exp_dir, sample_gt, plot_sims, n_post_samples, key, seed):
        """
        Runs the ABC experiments.
        """

        import inference.mcmc as mcmc

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        param_info = self.sim.get_param_info()

        if hasattr(sim_desc, 'dt_obs'):
            
            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars, sim_desc.dt_obs)

        else:

            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)

        ssm = sim_setup['ssm']
        props = sim_setup['props']
        inputs = sim_setup['inputs']
        tin = time.time()

        if sample_gt:

            key, subkey = jr.split(key)
            true_ps = sample_prior(key, props)[0]
            true_cps = to_train_array(true_ps, props)
            true_ps.from_unconstrained(props)
            states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs) 

        else:
            
            true_ps, observations = self.sim.get_ground_truth()

        if plot_sims:
    
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(states, label='States')
            ax.plot(observations, label='Observations')
            ax.legend()
            ax.set_title('Observations')
            plt.savefig(os.path.join(exp_dir, 'observations.png'))
            plt.close(fig)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            if isinstance(inf_desc, ed.PRT_MCMC_Descriptor):

                mcmc_runner = mcmc.BPF_MCMC(props, ssm)

                results = mcmc_runner.run(
                    key,
                    observations,
                    num_prt = inf_desc.num_prt,
                    num_posterior_samples=n_post_samples,
                    mcmc_steps = inf_desc.mcmc_steps,
                    num_iters = inf_desc.num_iters,
                    logger=logger
                    )
                
            else:

                raise TypeError('unknown PRT_MCMC algorithm')
            
            tout = time.time()
            logger.write('Total time: {0}'.format(tout - tin))

            util.io.save(([true_ps, true_cps], observations), os.path.join(exp_dir, 'gt'))
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save_txt(str(mcmc_runner.time), os.path.join(exp_dir, 'time.txt'))
            util.io.save_txt(str(seed), os.path.join(exp_dir, 'seed.txt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))
            util.io.save_txt(inspect.getsource(param_info), os.path.join(exp_dir, 'param_info.txt'))

            del results
            mcmc_runner = None
            jax.clear_backends()
            gc.collect()


    def _run_snl(self, exp_dir, sample_gt, plot_sims, n_post_samples, key, seed):
        """
        Runs the likelihood learner with MCMC.
        """

        import inference.nde as nde

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        param_info = self.sim.get_param_info()


        if hasattr(sim_desc, 'dt_obs'):
            
            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars, sim_desc.dt_obs)

        else:

            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)

        ssm = sim_setup['ssm']
        props = sim_setup['props']
        inputs = sim_setup['inputs']
        is_target = sim_setup['exp_info']['is_target']
        param_names = sim_setup['exp_info']['param_names']
        is_constrained_tree = sim_setup['exp_info']['constrainers']
        key, subkey = jr.split(key)
        xparam = sample_prior(subkey, props)[0]

        if sample_gt:

            key, subkey = jr.split(key)
            true_ps = sample_prior(key, props)[0]
            true_cps = to_train_array(true_ps, props)
            true_ps.from_unconstrained(props)
            states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs) 

        else:

            xparam = sample_prior(key, props)[0]
            true_cps = self.sim.get_ground_truth()
            unravel_fn = get_unravel_fn(xparam, props)
            unravel = unravel_fn(true_cps)
            tree = tree_from_params(xparam)
            new_tree = join_trees(unravel, tree, props)
            is_constrained_tree = xparam._is_constrained_tree()
            true_ps = params_from_tree(new_tree, param_names, is_constrained_tree)
            states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs) 

        if plot_sims:

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(states, label='States')
            ax.plot(observations, label='Observations')
            ax.legend()
            ax.set_title('Observations')
            plt.savefig(os.path.join(exp_dir, 'observations.png'))
            plt.close(fig)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            n_inputs = sim_desc.emission_dim * sim_desc.num_timesteps
            n_cond = to_train_array(xparam, props).shape[0]
            key_model, key_learner, key = jr.split(key, 3)
            model = self._create_model(n_inputs, n_cond, key_model)
            learner = nde.SequentialNeuralLikelihood(props, ssm, lag=-1)
            model_desc = self.exp_desc.inf.model

            model, (posterior_sample, posterior_cond_sample) = learner.learn_likelihood(
                key=key_learner,
                observations=observations,
                model=model,
                num_rounds=inf_desc.n_rounds,
                num_timesteps=sim_desc.num_timesteps,
                num_samples=inf_desc.n_samples,
                num_posterior_samples=n_post_samples,
                train_on=inf_desc.train_on,
                mcmc_steps=inf_desc.mcmc_steps,
                num_epochs=model_desc.nepochs,
                learning_rate=model_desc.lr,
                logger=logger

            )

            key, subkey = jr.split(key)
            mll = marg_loglik(subkey, props, observations, model, 100, -1)

            util.io.save(mll, os.path.join(exp_dir, 'mll'))
            util.io.save(nnx.split(model), os.path.join(exp_dir, 'model'))
            util.io.save(([true_ps, true_cps], observations), os.path.join(exp_dir, 'gt'))
            util.io.save(learner.all_params, os.path.join(exp_dir, 'all_params'))
            util.io.save(learner.all_emissions, os.path.join(exp_dir, 'all_emissions'))
            util.io.save(learner.all_cond_params, os.path.join(exp_dir, 'all_cond_params'))
            util.io.save((posterior_sample, posterior_cond_sample), os.path.join(exp_dir, 'posterior'))
            util.io.save(learner.time_all_rounds, os.path.join(exp_dir, 'time_all_rounds'))
            util.io.save(learner.all_dists.reshape(inf_desc.n_rounds, inf_desc.n_samples), os.path.join(exp_dir, 'all_dists'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))
            util.io.save_txt(str(seed), os.path.join(exp_dir, 'seed.txt'))
            util.io.save_txt(inspect.getsource(param_info), os.path.join(exp_dir, 'param_info.txt'))

            del posterior_sample, posterior_cond_sample
            learner = None
            model = None
            jax.clear_backends()
            gc.collect()

    def _run_tsnl(self, exp_dir, sample_gt, plot_sims, n_post_samples, key, seed):
        """
        Runs the likelihood learner with MCMC.
        """

        import inference.nde as nde

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        param_info = self.sim.get_param_info()

        if hasattr(sim_desc, 'dt_obs'):
            
            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars, sim_desc.dt_obs)

        else:

            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)        

        ssm = sim_setup['ssm']
        props = sim_setup['props']
        inputs = sim_setup['inputs']
        param_names = sim_setup['exp_info']['param_names']
        is_constrained_tree = sim_setup['exp_info']['constrainers']
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
            states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs) 

        else:

            xparam = sample_prior(key, props)[0]
            true_cps = self.sim.get_ground_truth()
            unravel_fn = get_unravel_fn(xparam, props)
            unravel = unravel_fn(true_cps)
            tree = tree_from_params(xparam)
            new_tree = join_trees(unravel, tree, props)
            is_constrained_tree = xparam._is_constrained_tree()
            true_ps = params_from_tree(new_tree, param_names, is_constrained_tree)
            states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs) 

        if plot_sims:

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(states, label='States')
            ax.plot(observations, label='Observations')
            ax.legend()
            ax.set_title('Observations')
            plt.savefig(os.path.join(exp_dir, 'observations.png'))
            plt.close(fig)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            n_inputs = sim_desc.emission_dim
            n_params = to_train_array(xparam, props).shape[0]
            n_cond = inf_desc.lag * sim_desc.emission_dim + n_params
            key_model, key_learner = jr.split(key)
            model = self._create_model(n_inputs, n_cond, key_model)
            learner = nde.SequentialNeuralLikelihood(props, ssm, lag=inf_desc.lag)
            model_desc = self.exp_desc.inf.model

            _, (posterior_sample, posterior_cond_sample) = learner.learn_likelihood(
                key=key_learner,
                observations=observations,
                model=model,
                num_rounds=inf_desc.n_rounds,
                num_timesteps=sim_desc.num_timesteps,
                num_samples=inf_desc.n_samples,
                num_posterior_samples=n_post_samples,
                train_on=inf_desc.train_on,
                mcmc_steps=inf_desc.mcmc_steps,
                logger=logger,
                num_tiles = num_tiles,
                num_epochs = model_desc.nepochs,
                learning_rate = model_desc.lr,
                subsample = subsample
            )

            key, subkey = jr.split(key)
            mll = marg_loglik(subkey, props, observations, model, 100, inf_desc.lag)

            util.io.save(mll, os.path.join(exp_dir, 'mll'))
            util.io.save(nnx.split(model), os.path.join(exp_dir, 'model'))
            util.io.save(([true_ps, true_cps], observations), os.path.join(exp_dir, 'gt'))
            util.io.save(learner.all_params, os.path.join(exp_dir, 'all_params'))
            util.io.save(learner.all_emissions, os.path.join(exp_dir, 'all_emissions'))
            util.io.save(learner.all_cond_params, os.path.join(exp_dir, 'all_cond_params'))
            util.io.save(learner.time_all_rounds, os.path.join(exp_dir, 'time_all_rounds'))
            util.io.save(learner.all_dists.reshape(inf_desc.n_rounds, inf_desc.n_samples), os.path.join(exp_dir, 'all_dists'))
            util.io.save((posterior_sample, posterior_cond_sample), os.path.join(exp_dir, 'posterior'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))
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
