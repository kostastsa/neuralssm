import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
import jax # type: ignore
from jax import numpy as jnp # type: ignore
from jax import random as jr # type: ignore
import blackjax # type: ignore
import jax.scipy.stats as jss # type: ignore
from jax.scipy.special import logsumexp as lse # type: ignore
from dynamax.utils.bijectors import RealToPSDBijector # type: ignore
import numpy as onp # type: ignore

from density_models import MAF
from ssm import SV

from flax import nnx # type: ignore
from matplotlib import pyplot as plt # type: ignore
from parameters import params_from_tree, sample_ssm_params, initialize, to_train_array, log_prior, get_unravel_fn, join_trees, tree_from_params
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
import tensorflow_probability.substrates.jax.bijectors as tfb # type: ignore
from simulation_inference import sequential_posterior_sampling, inference_loop
from filters import bpf

import csv
import time

import scienceplots # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib_inline # type: ignore
import os
plt.style.use(['science', 'ieee'])
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
from output_utils import true_loglik


# Initialize model and simulate dataset

## Define experiment parameters
num_timesteps = 100 # Number of emissions timesteps
num_mcmc_steps = 1000 # Number of MCMC steps
num_reps = 10 # number of experiment reps

lag = 10 # Number of emissions to condition on
num_samples = 100 # Number of samples for TAF/SNL training
num_rounds = 1 # Number of rounds of TAF/SNL training

num_particles = 100 # Number of particles for BPF
num_iters = 1 # Number of iterations for BPF likelihood estimator

rw_sigma = 0.1 # Random walk proposal sigma (don't tune)

## Define SVSSM parameters
state_dim = 1
emission_dim = state_dim
input_dim = 0

initial_mean = jnp.zeros(state_dim)
initial_covariance = jnp.eye(state_dim) * 1.0

dynamics_weights  = 0.91 * jnp.eye(state_dim)
dynamics_bias = jnp.zeros(state_dim)
dynamics_input_weights = jnp.zeros((state_dim, input_dim))
dynamics_covariance = jnp.eye(state_dim) * 1.0

emission_bias = jnp.zeros(emission_dim)
emission_covariance = jnp.eye(emission_dim) * 1.0
emission_beta = jnp.array([0.3])
emission_sigma = jnp.array([[2.0]]) 

# Initialize params and props
m = state_dim * (state_dim + 1) // 2
dynamics_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))
ell = emission_dim * (emission_dim + 1) // 2
emission_sigma_dist = tfd.MultivariateNormalDiag(loc=0.3*jnp.ones(ell), scale_diag=0.1*jnp.ones(ell))

param_names = [['mean', 'cov'],
               ['weights', 'bias', 'input_weights', 'cov'],
               ['bias', 'cov', 'beta', 'sigma']]

prior_fields = [[initial_mean, initial_covariance],
                [dynamics_weights, dynamics_bias, dynamics_input_weights, dynamics_covariance],
                [emission_bias, emission_covariance, emission_beta, emission_sigma_dist]]

is_constrained_tree = [[True, True], 
                       [True, True, True, True], 
                       [True, True, True, False]]

constrainers  = [[None, RealToPSDBijector],
                [None, None, None, RealToPSDBijector],
                [None, RealToPSDBijector, None, RealToPSDBijector]]

props, prior = initialize(prior_fields, param_names, constrainers)

seed = 121241378123  
key = jr.PRNGKey(seed)  
taf_error_array = jnp.zeros(num_reps)
snl_error_array = jnp.zeros(num_reps)
bpf_error_array = jnp.zeros(num_reps)
for i in range(num_reps):
    print(f"------------------ Repetition {i} ------------------")
    # Sample ***true*** params and emissions
    key, subkey = jr.split(key)
    svmodel = SV(state_dim, emission_dim)
    [true_param, example_param] = sample_ssm_params(key, prior, 2)
    true_param_vec = to_train_array(true_param, props)
    true_param.from_unconstrained(props)
    _, emissions = svmodel.simulate(subkey, true_param, num_timesteps)

    # Train TAF model
    reverse = True
    random_order = False
    batch_norm = False
    dropout = False
    nmades = 5
    dhidden = 32
    nhidden = 5

    print(f"------------------ TAF ")
    ## Initialize model
    din = emission_dim
    n_params = to_train_array(example_param, props).shape[0]
    dcond = lag * emission_dim + n_params
    key, subkey = jr.split(key)
    taf = MAF(din, nmades, dhidden, nhidden, dcond, nnx.Rngs(subkey), random_order, reverse, batch_norm, dropout)
    test_sv_model = SV(state_dim, emission_dim)
    ## Sample parameters for first round
    key, subkey = jr.split(key)
    taf_postmodel, taf_samples = sequential_posterior_sampling(key = key,
                model = taf,
                ssmodel=test_sv_model,
                lag=lag,
                num_rounds=num_rounds,
                num_timesteps=num_timesteps,
                num_samples=num_samples,
                num_mcmc_steps=num_mcmc_steps,
                emissions=emissions,
                prior=prior,
                props=props,
                example_param=example_param,
                param_names=param_names,
                is_constrained_tree=is_constrained_tree,
                rw_sigma=rw_sigma
)

    print(f"------------------ SNL ") 
    # Initialize model
    din = emission_dim * num_timesteps
    n_params = to_train_array(example_param, props).shape[0]
    dcond = n_params
    key, subkey = jr.split(key)
    snl = MAF(din, nmades, dhidden, nhidden, dcond, nnx.Rngs(subkey), random_order, reverse, batch_norm, dropout)
    snl_postmodel, snl_samples = sequential_posterior_sampling(key = key,
                    model = snl,
                    ssmodel=test_sv_model,
                    lag=0,
                    num_rounds=num_rounds,
                    num_timesteps=num_timesteps,
                    num_samples=num_samples,
                    num_mcmc_steps=num_mcmc_steps,
                    emissions=emissions,
                    prior=prior,
                    props=props,
                    example_param=example_param,
                    param_names=param_names,
                    is_constrained_tree=is_constrained_tree,
                    rw_sigma=rw_sigma
                    )

    print("* * * Sampling posteriors")
    key, subkey = jax.random.split(key)

    ### Define the log-density functions
    def bpf_logdensity_fn(cond_params):
        global subkey
        unravel_fn = get_unravel_fn(example_param, props)
        unravel = unravel_fn(cond_params)
        tree = tree_from_params(example_param)
        new_tree = join_trees(unravel, tree, props)
        params = params_from_tree(new_tree, param_names, is_constrained_tree)
        params.from_unconstrained(props)
        lps = []
        key = subkey
        for _ in range(num_iters):
            key, subkey = jr.split(key)
            _, lp = bpf(params, test_sv_model, emissions, num_particles, subkey)
            lp += log_prior(cond_params, prior)
            lps.append(lp)
        return jnp.mean(jnp.array(lps))

    ### Initialize MCMC chain and kernel
    key, subkey = jr.split(key)
    initial_cond_params = to_train_array(sample_ssm_params(subkey, prior, 1)[0], props)

    # taf_random_walk = blackjax.additive_step_random_walk(taf_logdensity_fn, blackjax.mcmc.random_walk.normal(rw_sigma))
    bpf_random_walk = blackjax.additive_step_random_walk(bpf_logdensity_fn, blackjax.mcmc.random_walk.normal(rw_sigma))

    # taf_initial_state = taf_random_walk.init(initial_cond_params)
    # taf_kernel = jax.jit(taf_random_walk.step)

    bpf_initial_state = bpf_random_walk.init(initial_cond_params)
    bpf_kernel = jax.jit(bpf_random_walk.step)

    ### Run inference loop
    key, subkey1, subkey2 = jax.random.split(key, 3)
    # taf_mcmc_states = inference_loop(subkey1, taf_kernel, taf_initial_state, num_mcmc_steps)
    bpf_mcmc_states = inference_loop(subkey2, bpf_kernel, bpf_initial_state, num_mcmc_steps)

    ## Output
    ### Kernel density estimation and errors
    taf_kernel_points = taf_samples.T
    taf_kde = jss.gaussian_kde(taf_kernel_points)
    taf_num_simulations = num_rounds * num_samples * num_timesteps
    taf_error = -jnp.log(taf_kde.evaluate(true_param_vec))
    taf_error_array = taf_error_array.at[i].set(taf_error[0])

    snl_kernel_points = snl_samples.T
    snl_kde = jss.gaussian_kde(snl_kernel_points)
    snl_num_simulations = num_rounds * num_samples * num_timesteps
    snl_error = -jnp.log(snl_kde.evaluate(true_param_vec))
    snl_error_array = snl_error_array.at[i].set(snl_error[0])

    bpf_kernel_points = bpf_mcmc_states.position.T
    bpf_kde = jss.gaussian_kde(bpf_kernel_points)
    bpf_num_simulations = num_particles * num_timesteps * num_mcmc_steps * num_iters
    bpf_error = -jnp.log(bpf_kde.evaluate(true_param_vec))
    bpf_error_array = bpf_error_array.at[i].set(bpf_error[0])

    print(f"TAF error: {taf_error[0]}, SNL error, :{snl_error[0]}, BPF error: {bpf_error[0]}")

del taf, snl

### Write outputs to file
bpf_success_pct = (num_reps - (jnp.isnan(bpf_error_array).sum() + jnp.isinf(bpf_error_array).sum())) / num_reps
bpf_error_array = bpf_error_array[~jnp.isnan(bpf_error_array)]
bpf_error_array = bpf_error_array[~jnp.isinf(bpf_error_array)]

taf_success_pct = (num_reps - (jnp.isnan(taf_error_array).sum() + jnp.isinf(taf_error_array).sum())) / num_reps
taf_error_array = taf_error_array[~jnp.isnan(taf_error_array)]
taf_error_array = taf_error_array[~jnp.isinf(taf_error_array)]

snl_success_pct = (num_reps - (jnp.isnan(snl_error_array).sum() + jnp.isinf(snl_error_array).sum())) / num_reps
snl_error_array = snl_error_array[~jnp.isnan(snl_error_array)]
snl_error_array = snl_error_array[~jnp.isinf(snl_error_array)]

# Define the data to be written
header_row = ["BPF error", 
              "BPF num_sims", 
              "TAF error", 
              "TAF num_sims", 
              "Baseline error"]

new_row = [jnp.mean(bpf_error_array), 
           jnp.log10(bpf_num_simulations), 
           bpf_success_pct,
           jnp.mean(taf_error_array), 
           jnp.log10(taf_num_simulations),
           taf_success_pct,
           jnp.mean(snl_error_array),
           jnp.log10(snl_num_simulations),
           snl_success_pct,
           num_timesteps,
           num_mcmc_steps,
           num_reps,
           lag,
           num_samples,
           num_rounds,
           num_particles,
           num_iters]

# Specify the file name
file_name = os.path.join('/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm/output', f"output_sv.csv")
# Write the data to the CSV file
with open(file_name, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(new_row)
