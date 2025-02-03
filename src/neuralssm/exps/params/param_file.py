import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
from jax import numpy as jnp # type: ignore
from dynamax.utils.bijectors import RealToPSDBijector # type: ignore
from parameters import initialize
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
from typing import NamedTuple
from parameters import ParamMeta

state_dim = 1
emission_dim = 1
input_dim = 0

initial_mean = jnp.zeros(state_dim)
initial_covariance = jnp.eye(state_dim) * 0.1

dynamics_weights  = 0.9 * jnp.eye(state_dim)
dynamics_bias = jnp.zeros(state_dim)
dynamics_input_weights = jnp.zeros((state_dim, input_dim))
dynamics_covariance = jnp.eye(state_dim) * 0.1

emission_weights = jnp.eye(emission_dim, state_dim)
emission_bias = jnp.zeros(emission_dim)
emission_input_weights = jnp.zeros((emission_dim, input_dim))
emission_covariance = jnp.eye(emission_dim) * 0.1

# Initialize props and prior
m = state_dim * (state_dim + 1) // 2
dynamics_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=0.1*jnp.ones(m))

param_names = [['mean', 'cov'],
               ['weights', 'bias', 'input_weights', 'cov'],
               ['weights', 'bias', 'input_weights', 'cov']]

prior_fields = [[initial_mean, initial_covariance],
                [dynamics_weights, dynamics_bias, dynamics_input_weights, dynamics_covariance_dist],
                [emission_weights, emission_bias, emission_input_weights, emission_covariance]]

is_constrained_tree = [[True, True], 
                       [True, True, True, False], 
                       [True, True, True, True]]

constrainers  = [[None, RealToPSDBijector],
                [None, None, None, RealToPSDBijector],
                [None, None, None, RealToPSDBijector]]

props, prior = initialize(prior_fields, param_names, constrainers)

# parameter metadata
pmeta = ParamMeta(props, prior)