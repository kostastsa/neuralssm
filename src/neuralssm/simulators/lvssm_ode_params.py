import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
from jax import numpy as jnp # type: ignore
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
from util.distributions import OscPrior
import tensorflow_probability.substrates.jax.bijectors as tfb
import numpyro.distributions as dist # type: ignore


def _init_vals(emission_dim):
    
    state_dim = 2
    initial_mean = jnp.array([50.0, 25.0]) 
    initial_covariance = jnp.eye(state_dim) * 0.1
    rates = jnp.array([10.0, 0.4, 0.1, 0.4])
    sigma = jnp.array([0.01])
    emission_alpha = jnp.array([0.1])

    init_vals = [[initial_mean, initial_covariance],
                [rates, sigma],
                [emission_alpha]]

    return init_vals

def _param_dists(emission_dim):

    # Initial
    state_dim = 2
    initial_mean = jnp.array([50.0, 25.0])
    initial_mean_dist = tfd.MultivariateNormalFullCovariance(loc=initial_mean, covariance_matrix= jnp.eye(state_dim) * 0.1)

    m = state_dim * (state_dim + 1) // 2
    initial_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))

    # Rates
    mean_rates = jnp.array([10.0, 0.4, 0.1, 0.4])
    rates_dist = tfd.MultivariateNormalDiag(loc=mean_rates, scale_diag=0.1 * jnp.ones(mean_rates.shape[0]))

    # Sigma
    mean_log_sigma = jnp.array([jnp.log(0.1)])
    sigma_dist = tfd.MultivariateNormalDiag(loc=mean_log_sigma, scale_diag=0.1 * jnp.ones(1))

    # Emissions
    #
    mean_emission_alpha = jnp.array([0.1])
    emission_alpha_dist = tfd.MultivariateNormalDiag(loc=mean_emission_alpha, scale_diag = 0.1 * jnp.ones(1))

    param_dists = [[initial_mean_dist, initial_covariance_dist],
                   [rates_dist, sigma_dist],
                   [emission_alpha_dist]]
    
    return param_dists


