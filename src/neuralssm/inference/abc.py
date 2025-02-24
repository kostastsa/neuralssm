import jax.numpy as jnp
import jax.random as jr
import sys, os
from jax import lax, vmap, debug
from jax.scipy.special import logsumexp as lse 
from util.sample import map_sims
from util.param import sample_prior, to_train_array, log_prior
import tensorflow_probability.substrates.jax.distributions as tfd
import logging
import time

logging.basicConfig(level=logging.INFO)

class SMC:
    """
    Implements sequential monte carlo for abc.
    """

    def __init__(self, props, ssm):

        self.props = props
        self.ssm = ssm
        self.time_all_rounds = []

    def run(self,
            key,
            observations,
            num_particles: int=100, 
            eps_init: float=10.0,
            eps_last: float=0.1,
            eps_decay: float=0.9,
            ess_min: float=0.5,
            logger=sys.stdout
            ):
        
        logger.write('---------------\n')
        logger.write('Running SMC-ABC\n')
        logger.write('---------------\n')
    
        num_timesteps, emission_dim = observations.shape
        logger = open(os.devnull, 'w') if logger is None else logger

        def _step(carry, eps):
            tin = time.time()
            particles, weights, cov, key, log_ess, count_sims, acc_dist = carry
            chol = jnp.linalg.cholesky(cov)
            log_ess_min = jnp.log(ess_min)
            log_n_particles = jnp.log(num_particles)
            # eps = acc_dist

            def _cond_fn_step(val):
                dist, _, _, _ = val
                return dist > eps

            def _while_step(_):
                'until accept'
                _, _, count, key = _
                # Propose new particles (resample + move)
                key, subkey = jr.split(key)
                sampled_idx = jr.choice(subkey, jnp.arange(weights.shape[0]), shape=(1,), p=weights)
                prt = jnp.take(particles, sampled_idx, axis=0).squeeze()

                key, subkey = jr.split(key)
                unit_gauss_vec = tfd.MultivariateNormalDiag(jnp.zeros(prt_dim), jnp.ones(prt_dim)).sample(1, subkey)
                new_particle = prt + chol @ unit_gauss_vec.reshape((prt_dim,))

                # Simulate emissions
                key, subkey = jr.split(key)
                sim_emissions = map_sims(subkey, new_particle, self.props, self.ssm, num_timesteps)

                # Compute distance
                dist = jnp.linalg.norm(observations - sim_emissions) / jnp.sqrt(num_timesteps * emission_dim)
                count += 1
                return (dist, new_particle, count, key)
            
            key, subkey = jr.split(key)
            acc_dists, new_particles, counts, _ = vmap(lambda prt, key: lax.while_loop(_cond_fn_step, _while_step, (jnp.inf, prt , 0, key)))(particles, jr.split(subkey, num_particles))

            # Compute weights
            log_prior_values = vmap(log_prior, in_axes=(0, None))(new_particles, self.props)
            log_kernels = tfd.MultivariateNormalFullCovariance(particles, cov).log_prob(new_particles)
            log_new_weights = log_prior_values - lse(jnp.log(weights) + log_kernels)
            log_new_weights -= jnp.max(log_new_weights)
            new_weights = jnp.exp(log_new_weights)
            new_weights /= jnp.sum(new_weights)

            # Compute log ess and do resampling if necessary
            log_ess = -lse(2.0 * jnp.log(new_weights)) - log_n_particles
            key, subkey = jr.split(key)
            sampled_idx = jr.choice(subkey, jnp.arange(weights.shape[0]), shape=(num_particles,), p=new_weights)
            resampled_prt = jnp.take(new_particles, sampled_idx, axis=0)
            new_particles = jnp.where(log_ess < log_ess_min, resampled_prt, new_particles)

            # Adapt covariance
            cov = 2.0 *jnp.cov(new_particles.T, aweights=new_weights).reshape((prt_dim, prt_dim))
            count_sims += jnp.sum(counts)
            acc_dist = 1.0 * jnp.max(acc_dists)
            carry = (new_particles, new_weights, cov, key, log_ess, count_sims, acc_dist)
            tout = time.time()
            self.time_all_rounds.append(tout-tin)
            logger.write('--------- time: {:.2f}\n'.format(tout-tin))
            return carry, (new_particles, new_weights, log_ess, count_sims, acc_dist)

        key, subkey = jr.split(key)

        init_params = sample_prior(subkey, self.props, num_particles)
        init_particles = jnp.array(list(map(to_train_array, init_params, [self.props]*num_particles)))
        prt_dim = init_particles.shape[1]
        init_weights = jnp.ones(num_particles) / num_particles
        init_cov = jnp.cov(init_particles.T, aweights=init_weights).reshape((prt_dim, prt_dim))
        carry = (init_particles, init_weights, init_cov, key, 0.0, 0, 10.0)
        all_eps = jnp.array([eps_decay**i * eps_init for i in range(100)])      
        all_eps = all_eps[all_eps > eps_last]  
        _, (all_particles, all_weights, all_log_ess, all_counts, avg_acc_dists)  = lax.scan(_step, carry, all_eps)
        is_nan = jnp.isnan(all_particles).any()

        return all_particles, all_weights, all_eps, all_log_ess, all_counts, avg_acc_dists, is_nan