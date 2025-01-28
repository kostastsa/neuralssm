import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap, jit
from jax.scipy.special import factorial as fac
from parameters import get_unravel_fn, tree_from_params, join_trees, params_from_tree

def reshape_emissions(emissions, lag):
    '''
    Takes the emission array and returns an array of stacked emissions with lag L.
    Each element of the new array has L+1 emissions stacked (L for conditioning and 1 for predicting).
    '''
    num_timesteps, emission_dim = emissions.shape
    emissions = jnp.concatenate([jnp.zeros((lag, emission_dim)), emissions])
    lagged_emissions = []
    for t in range(num_timesteps):
        lagged_emissions.append(emissions[t:t+lag+1].flatten())
    return jnp.stack(lagged_emissions)

def map_sims(key, cond_param, props, example_params, ssmodel, num_timesteps):
    ''' Takes parameters in conditional form and example params for static, and
    returns emissions from ssmodel. ''
    '''
    unravel_fn = get_unravel_fn(example_params, props)
    unravel = unravel_fn(cond_param)
    tree = tree_from_params(example_params)
    new_tree = join_trees(unravel, tree, props)
    param = params_from_tree(new_tree, example_params._get_names(), example_params._get_is_constrained())
    param.from_unconstrained(props)
    _, emissions = ssmodel.simulate(key, param, num_timesteps)
    return emissions

def resample(weights, particles, key):                                                                  
    keys = jr.split(key, 2)
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(keys[0], jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    resampled_particles = jnp.take(particles, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    next_key = keys[1]
    return weights, resampled_particles, next_key

def resample2(weights, particles, key):                                                                  
    keys = jr.split(key, 2)
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(keys[0], jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    resampled_particles = jnp.take(particles, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    next_key = keys[1]
    return weights, resampled_particles, next_key, resampled_idx

bin_coeff = jit(lambda x, n: jnp.nan_to_num(fac(x) / (fac(n) * fac(x - n)), nan=0.0))