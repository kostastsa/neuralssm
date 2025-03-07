import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
from jax import numpy as jnp # type: ignore
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
from util.distributions import OscPrior
import numpyro.distributions as dist # type: ignore

log_rates = jnp.array([jnp.log(0.1), jnp.log(3.0)])

def _init_vals(emission_dim, pop_size):
    
    state_dim = 3
    initial_mean = 1.0 * jnp.array([pop_size-1, 1, 0]) 
    initial_covariance = jnp.eye(state_dim) * .01

    pre = jnp.array([
        [1, 1, 0],
        [0, 1, 0]
    ])

    post = jnp.array([
        [0, 2, 0],
        [0, 0, 1]
    ])

    emission_covariance = jnp.eye(emission_dim) * 0.1

    init_vals = [[initial_mean, initial_covariance],
                [pre, post, log_rates],
                [emission_covariance]]

    return init_vals


def _param_dists(emission_dim, num_reactions, pop_size):

    # Initial
    state_dim = 3
    initial_mean = 1.0 * jnp.array([pop_size-1, 1, 0])
    initial_covariance = jnp.eye(state_dim) * 1.0
    initial_mean_dist = tfd.MultivariateNormalFullCovariance(loc=initial_mean, covariance_matrix=initial_covariance)

    m = state_dim * (state_dim + 1) // 2
    initial_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))

    # Dynamics
    pre_dist = dist.Dirichlet(jnp.ones(num_reactions))
    post_dist = dist.Dirichlet(jnp.ones(num_reactions))

    # Log-rates
    ## uniform
    uniform_base_dist = tfd.Uniform(low=-5.0*jnp.ones(num_reactions), high=2.0*jnp.ones(num_reactions))
    uniform_dist = tfd.Independent(uniform_base_dist, reinterpreted_batch_ndims=1)

    ## gaussian
    gaussian_dist = tfd.MultivariateNormalDiag(loc=log_rates, scale_diag=0.05*jnp.ones(num_reactions))
    osc_dist = OscPrior(uniform_low=[-5.0]*num_reactions, 
                              uniform_high=[2.0]*num_reactions, 
                              gaussian_loc=log_rates, 
                              gaussian_scale=[0.5]*4)
    
    log_rates_dist = gaussian_dist

    # Emissions
    l = emission_dim * (emission_dim + 1) // 2
    emission_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(l), scale_diag=0.1*jnp.ones(l))

    param_dists = [[initial_mean_dist, initial_covariance_dist],
                    [pre_dist, post_dist, log_rates_dist],
                    [emission_covariance_dist]]
    
    return param_dists

