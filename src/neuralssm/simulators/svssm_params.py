import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
from jax import numpy as jnp # type: ignore
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
import numpyro.distributions as dist # type: ignore


def _init_vals(state_dim, emission_dim, input_dim):
        
    initial_mean = jnp.zeros(state_dim)
    initial_covariance = jnp.eye(state_dim) * 1.0

    dynamics_weights  = 0.91 * jnp.eye(state_dim)
    dynamics_bias = jnp.zeros(state_dim)
    dynamics_input_weights = jnp.zeros((state_dim, input_dim))
    dynamics_covariance = jnp.eye(state_dim) * 1.0

    emission_bias = jnp.zeros(emission_dim)
    emission_corchol = jnp.eye(emission_dim) * 1.0

    init_vals = [[initial_mean, initial_covariance],
                    [dynamics_weights, dynamics_bias, dynamics_input_weights, dynamics_covariance],
                    [emission_bias, emission_corchol]]

    return init_vals

def _param_dists(state_dim, emission_dim, input_dim):

    # initial_mean 
    initial_mean_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag=jnp.ones(state_dim))

    # initial_covariance
    m = state_dim * (state_dim + 1) // 2
    initial_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))

    # dynamics_weights
    mu = 0.8 * jnp.eye(state_dim)
    col_cov = jnp.eye(state_dim) * 0.01
    row_cov = jnp.eye(state_dim) * 0.01
    scale_column = jnp.linalg.cholesky(col_cov)
    scale_row = jnp.linalg.cholesky(row_cov)

    dynamics_weights_dist = dist.MatrixNormal(loc=mu,
                                              scale_tril_row=scale_row,
                                              scale_tril_column=scale_column)

    # dynamics_bias
    dynamics_bias_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag= 1.0 * jnp.ones(state_dim))

    # dynamics_input_weights
    mu = 0.0 * jnp.eye(state_dim, input_dim)
    col_cov = jnp.eye(input_dim) * 0.001
    row_cov = jnp.eye(state_dim) * 0.001
    scale_column = jnp.linalg.cholesky(col_cov)
    scale_row = jnp.linalg.cholesky(row_cov)

    dynamics_input_weights_dist = dist.MatrixNormal(loc=mu,
                                                    scale_tril_row=scale_row,
                                                    scale_tril_column=scale_column)

    # dynamics_covariance
    dynamics_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=0.1*jnp.ones(m))

    # emission_bias
    emission_bias_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(emission_dim), scale_diag= 1.0 * jnp.ones(emission_dim))

    # emission_chol
    m = emission_dim * (emission_dim - 1) // 2
    emission_corchol_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))
    
    param_dists = [[initial_mean_dist, initial_covariance_dist],
                    [dynamics_weights_dist, dynamics_bias_dist, dynamics_input_weights_dist, dynamics_covariance_dist],
                    [emission_bias_dist, emission_corchol_dist]]
    
    return param_dists

