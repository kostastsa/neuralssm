import jax.random as jr
from jax import lax, vmap, debug, jit
import logging
import blackjax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import Optional
from util.sample import resample
from jaxtyping import Array, Float, Int
from parameters import ParamSSM
from simulators.ssm import SPN
from util.param import from_conditional, log_prior, sample_prior, to_train_array
from util.misc import swap_axes_on_values
from util.sample import mcmc_inference_loop
from functools import partial
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)


class BPF_MCMC:
    """
    Implements bpf-mcmc.
    """

    def __init__(self, props, ssm):

        self.props = props
        self.ssm = ssm
        self.xparam = None
        self.time = None

    def bpf(self,
            params: ParamSSM,
            observations: Float[Array, "ntime emission dim"],
            num_particles: Int,
            key: jr.PRNGKey = jr.PRNGKey(0),
            inputs: Optional[Float[Array, "ntime input dim"]] = None, 
            ess_threshold: float = 0.5
           ):
        r"""
        Bootstrap particle filter for the nonlinear state space model.
        Args:
            params: Parameters of the nonlinear state space model.
            emissions: Emissions.
            num_particles: Number of particles.
            rng_key: Random number generator key.
            inputs: Inputs. 

        Returns:
            Posterior particle filtered.
        """

        num_timesteps = observations.shape[0]

        # Dynamics and emission functions
        inputs = inputs if inputs is not None else jnp.zeros((num_timesteps, self.ssm.input_dim))
        
        def _step(carry, t):
            weights, ll, prev_states, key = carry

            # Get parameters and inputs for time index t
            u = inputs[t]
            y = observations[t]
            
            # Sample new particles 
            keys = jr.split(key, num_particles+1)
            next_key = keys[0]
            get_new_states = vmap(self.ssm.dynamics_simulator, in_axes=(0,None,0,None))
            new_states = get_new_states(keys[1:], params, prev_states, u)

            # Compute weights 
            get_lps = vmap(self.ssm.emission_log_prob, in_axes=(None,None,0,None))
            lls = get_lps(params, y, new_states, u)
            lls -= jnp.max(lls)
            ls = jnp.exp(lls)

            weights = jnp.multiply(ls, weights)

            ll += jnp.log(jnp.sum(weights))
            new_weights = weights / jnp.sum(weights)

            # Resample if necessary
            resample_cond = 1.0 / jnp.sum(jnp.square(new_weights)) < ess_threshold * num_particles
            weights, new_states, next_key = lax.cond(resample_cond, resample, lambda *args: args, new_weights, new_states, next_key)

            outputs = {
                'weights':weights,
                'particles':new_states
            }

            carry = (weights, ll, new_states, next_key)

            return carry, outputs
        
        keys = jr.split(key, num_particles+1)
        next_key = keys[0]
        weights = jnp.ones(num_particles) / num_particles
        map_sample = vmap(MVN(loc=params.initial.mean.value, covariance_matrix=params.initial.cov.value).sample, in_axes=(None,0))
        particles = map_sample((), keys[1:])

        if isinstance(self.ssm, SPN):

            states = jnp.pad(particles, ((0, 0), (1, 0)), mode='constant', constant_values=0)

        else:

            states = particles
        
        carry = (weights, 0.0, states, next_key)

        out_carry, outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
        ll = out_carry[1]
        outputs = swap_axes_on_values(outputs)
    
        return outputs, ll
    
    def logdensity_fn(self,
                      cond_params, 
                      key, 
                      emissions, 
                      num_prt, 
                      num_iters):
        
        params = from_conditional(self.xparam, self.props, cond_params)
        lps = []

        for _ in range(num_iters):

            key, subkey = jr.split(key)
            _, lp = self.bpf(params, emissions, num_prt, subkey)
            lp += log_prior(cond_params, self.props)
            lps.append(lp)

        return jnp.mean(jnp.array(lps))
    
    def run(self,
            key,
            observations,
            num_prt: int,
            mcmc_steps: int,
            num_posterior_samples: int = 1000,
            num_iters: int = 1,
            logger = None,
            rw_sigma : float = 0.1,
            ):

        logger.write('-----------------------\n')
        logger.write('Sampling BPF-MCMC chain\n')
        logger.write('-----------------------\n')
        tin = time.time()

        key, subkey = jr.split(key)
        initial_param = sample_prior(subkey, self.props, 1)[0]
        initial_cond_params = to_train_array(initial_param, self.props)
        self.xparam = initial_param


        key, subkey = jr.split(key)
        logdensity_fn = partial(self.logdensity_fn, key=subkey, emissions=observations, num_prt=num_prt, num_iters=num_iters)
        bpf_random_walk = blackjax.additive_step_random_walk(logdensity_fn, blackjax.mcmc.random_walk.normal(rw_sigma))
        bpf_initial_state = bpf_random_walk.init(initial_cond_params)
        bpf_kernel = jit(bpf_random_walk.step)

        ### Run inference loop
        mcmc_samples, logpdfs = mcmc_inference_loop(key, bpf_kernel, bpf_initial_state, mcmc_steps)

        tout = time.time()
        self.time = tout-tin
        
        return mcmc_samples[-num_posterior_samples:], logpdfs[-num_posterior_samples:]