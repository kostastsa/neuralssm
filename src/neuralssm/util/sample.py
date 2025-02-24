import jax.random as jr
import jax.numpy as jnp
from jax import jit, lax
import blackjax
from util.param import get_unravel_fn, tree_from_params, join_trees, params_from_tree, sample_prior, to_train_array


def map_sims(key, cond_param, props, ssmodel, num_timesteps):
    ''' Takes parameters in conditional form and
    returns emissions from ssmodel. ''
    '''
    xp = sample_prior(key, props)[0]
    unravel_fn = get_unravel_fn(xp, props)
    unravel = unravel_fn(cond_param)
    tree = tree_from_params(xp)
    new_tree = join_trees(unravel, tree, props)
    param = params_from_tree(new_tree, xp._get_names(), xp._is_constrained_tree())
    param.from_unconstrained(props)
    _, emissions = ssmodel.simulate(key, param, num_timesteps)
    return emissions


def mcmc_inference_loop(rng_key, kernel, initial_state, num_samples):
    '''
    Runs the MCMC kernel for num_samples steps and returns the states.
    '''
    @jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jr.split(rng_key, num_samples)
    _, states = lax.scan(one_step, initial_state, keys)

    return states


def sample_logpdf(key, learner, logdensity_fn, num_samples, num_mcmc_steps, rw_sigma):

    assert(num_mcmc_steps > num_samples), 'Number of MCMC steps must be greater than number of samples'

    ## Initialize MCMC chain and kernel
    key, subkey = jr.split(key)
    initial_cond_params = to_train_array(sample_prior(subkey, learner.props, 1)[0], learner.props)
    taf_random_walk = blackjax.additive_step_random_walk(logdensity_fn, blackjax.mcmc.random_walk.normal(rw_sigma))
    taf_initial_state = taf_random_walk.init(initial_cond_params)
    taf_kernel = jit(taf_random_walk.step)

    ## Run MCMC inference loop
    key, subkey = jr.split(key)
    taf_mcmc_states = mcmc_inference_loop(subkey, taf_kernel, taf_initial_state, num_mcmc_steps)
    ps = taf_mcmc_states.position[-num_samples:]

    # Setup params for next round
    params_sample = []
    param_names = learner.xparam._get_names()
    is_constrained_tree = learner.xparam._is_constrained_tree()
    for cond_param in ps:
        unravel_fn = get_unravel_fn(learner.xparam, learner.props)
        unravel = unravel_fn(cond_param)
        tree = tree_from_params(learner.xparam)
        new_tree = join_trees(unravel, tree, learner.props)
        param = params_from_tree(new_tree, param_names, is_constrained_tree)
        params_sample.append(param)

    return params_sample, ps


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