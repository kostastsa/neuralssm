import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap, debug
from jax.scipy.special import logsumexp as lse 
from utils import map_sims
from parameters import log_prior
from parameters import sample_ssm_params, to_train_array
from parameters import sample_ssm_params, to_train_array, log_prior

from jax import numpy as jnp
from jax import random as jr
import tensorflow_probability.substrates.jax.distributions as tfd


def smc_abc(key,
            observations,
            ssm, 
            example_param,
            props,
            prior, 
            num_particles: int=100, 
            num_rounds: int=10, 
            tol_init: float=3.0 
            ):
    
    num_timesteps, emission_dim = observations.shape

    def _step(carry, t):
        particles, weights, cov, key = carry
        tol = tol_init - 0.1 * t
        # debug.print('{x}', x=tol)
        chol = jnp.linalg.cholesky(cov)

        
        def _cond_fn(val):
            dist, _, _, _ = val
            return dist > tol

        def _while_step(_):
            _, _, count, key = _
            # Propose new particles (resample + move)
            key, subkey = jr.split(key)
            sampled_idx = jr.choice(subkey, jnp.arange(weights.shape[0]), shape=(1,), p=weights)
            sampled_prt = jnp.take(particles, sampled_idx, axis=0)
            key, subkey = jr.split(key)
            unit_gauss_vec = tfd.MultivariateNormalDiag(jnp.zeros(prt_dim), jnp.ones(prt_dim)).sample(1, subkey)
            new_particle = sampled_prt + chol @ unit_gauss_vec
            new_particle = new_particle.flatten()
            
            # Simulate emissions
            key, subkey = jr.split(key)
            sim_emissions = map_sims(subkey, new_particle, props, example_param, ssm, num_timesteps)
            # Compute distance
            dist = jnp.linalg.norm(observations - sim_emissions) / jnp.sqrt(num_timesteps * emission_dim)
            count += 1
            return (dist, new_particle, count, key)
        
        key, subkey = jr.split(key)
        _, new_particles, sim_accept, _ = vmap(lambda key: lax.while_loop(_cond_fn, _while_step, (jnp.inf, particles[0], 0, key)))(jr.split(subkey, num_particles))

        # Compute weights
        log_prior_values = vmap(log_prior, in_axes=(0, None))(new_particles, prior)
        log_kernels = tfd.MultivariateNormalFullCovariance(particles, cov).log_prob(new_particles)
        log_new_weights = log_prior_values - lse(jnp.log(weights) + log_kernels)
        log_new_weights -= jnp.max(log_new_weights)
        new_weights = jnp.exp(log_new_weights)
        new_weights /= jnp.sum(new_weights)
        # Adapt covariance
        cov = jnp.cov(new_particles.T, aweights=new_weights).reshape((prt_dim, prt_dim))
        
        carry = (new_particles, new_weights, cov, key)
        outputs = (new_particles, new_weights, sim_accept)
        return carry, outputs

    key, subkey = jr.split(key)
    init_params = sample_ssm_params(subkey, prior, num_particles)
    init_particles = jnp.array(list(map(to_train_array, init_params, [props]*num_particles)))
    prt_dim = init_particles.shape[1]
    init_weights = jnp.ones(num_particles) / num_particles
    init_cov = jnp.cov(init_particles.T, aweights=init_weights).reshape((prt_dim, prt_dim))
    carry = (init_particles, init_weights, init_cov, key)
    _, (particles, weights, sim_accept) = lax.scan(_step, carry, jnp.arange(num_rounds))

    return particles, weights, sim_accept
