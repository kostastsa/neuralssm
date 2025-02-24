import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import Optional
from util.sample import resample
from jaxtyping import Array, Float, Int
from parameters import ParamSSM
from ssm import SSM

def swap_axes_on_values(outputs, axis1=0, axis2=1):
    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))

def bpf(
    params: ParamSSM,
    model: SSM, 
    emissions: Float[Array, "ntime emission dim"],
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

    num_timesteps = len(emissions)

    # Dynamics and emission functions
    inputs = inputs if inputs is not None else jnp.zeros((num_timesteps, model.input_dim))

    
    def _step(carry, t):
        weights, ll, particles, key = carry
        
        # Get parameters and inputs for time index t
        u = inputs[t]
        y = emissions[t]

        # Sample new particles 
        keys = jr.split(key, num_particles+1)
        next_key = keys[0]
        map_sample_particles = vmap(model.dynamics_simulator, in_axes=(0,None,0,None))
        new_particles = map_sample_particles(keys[1:], params, particles, u)

        # Compute weights 
        map_log_prob = vmap(model.emission_log_prob, in_axes=(None,None,0,None))
        lls = map_log_prob(params,y,new_particles,u)
        lls -= jnp.max(lls)
        ls = jnp.exp(lls)
        weights = jnp.multiply(ls, weights)
        ll += jnp.log(jnp.sum(weights))
        new_weights = weights / jnp.sum(weights)

        # Resample if necessary
        resample_cond = 1.0 / jnp.sum(jnp.square(new_weights)) < ess_threshold * num_particles
        weights, new_particles, next_key = lax.cond(resample_cond, resample, lambda *args: args, new_weights, new_particles, next_key)

        outputs = {
            'weights':weights,
            'particles':new_particles
        }

        carry = (weights, ll, new_particles, next_key)

        return carry, outputs
    
    # Initialize carry
    keys = jr.split(key, num_particles+1)
    next_key = keys[0]
    weights = jnp.ones(num_particles) / num_particles
    map_sample = vmap(MVN(loc=params.initial.mean.value, covariance_matrix=params.initial.cov.value).sample, in_axes=(None,0))
    particles = map_sample((), keys[1:])
    carry = (weights, 0.0, particles, next_key)

    # scan
    out_carry, outputs =  lax.scan(_step, carry, jnp.arange(num_timesteps))
    ll = out_carry[1]
    outputs = swap_axes_on_values(outputs)
 
    return outputs, ll

# def apf(
#     params: ParamsBPF,
#     emissions: Float[Array, "ntime emission dim"],
#     num_particles: Int,
#     key: jr.PRNGKey = jr.PRNGKey(0),
#     inputs: Optional[Float[Array, "ntime input dim"]] = None
# ):
#     r"""
#     Bootstrap particle filter for the nonlinear state space model.
#     Args:
#         params: Parameters of the nonlinear state space model.
#         emissions: Emissions.
#         num_particles: Number of particles.
#         rng_key: Random number generator key.
#         inputs: Inputs. 

#     Returns:
#         Posterior particle filtered.
#     """

#     num_timesteps = len(emissions)

#     # Dynamics and emission functions
#     inputs = _process_input(inputs, num_timesteps)

    
#     def _step(carry, t):
#         weights, particles, key = carry
        
#         # Get parameters and inputs for time index t
#         q0 = _get_params(params.dynamics_noise_bias, 2, t)
#         u = inputs[t]
#         y = emissions[t]
#         keys = jr.split(key, num_particles+2)
#         next_key = keys[0]

#         # Compute means of propagation kernels
#         map_means = vmap(params.dynamics_function, in_axes=(0,None,None))(particles, q0, u)

#         # Compute normalized pre-weights
#         map_log_prob = vmap(params.emission_distribution_log_prob, in_axes=(0,None,None))(map_means, y, u)
#         logweights = map_log_prob + jnp.log(weights)
#         logweights -= jnp.max(logweights)
#         weights = jnp.exp(logweights)
#         weights /= jnp.sum(weights)

#         # Do resampling
#         new_weights, resampled_particles, next_key, resampled_idx =  resample2(weights, particles, keys[1])

#         # Sample new particles 

#         map_sample_particles = vmap(params.sample_dynamics_distribution, in_axes=(0,0,None))
#         new_particles = map_sample_particles(keys[2:], resampled_particles, u)

#         map_log_prob = vmap(params.emission_distribution_log_prob, in_axes=(0,None,None))(new_particles, y, u)
#         map_log_prob_denom = vmap(params.emission_distribution_log_prob, in_axes=(0,None,None))(map_means[resampled_idx], y, u)
#         new_lls = map_log_prob - map_log_prob_denom
#         new_lls -= jnp.max(new_lls)
#         new_weights = jnp.exp(new_lls)
#         new_weights /= jnp.sum(new_weights)

        
#         outputs = {
#             'weights': new_weights,
#             'particles':new_particles
#         }

#         carry = (new_weights, new_particles, next_key)

#         return carry, outputs
    
#     # Initialize carry
#     keys = jr.split(key, num_particles+1)
#     next_key = keys[0]
#     weights = jnp.ones(num_particles) / num_particles
#     map_sample = vmap(MVN(loc=params.initial_mean, covariance_matrix=params.initial_covariance).sample, in_axes=(None,0))
#     particles = map_sample((), keys[1:])
#     carry = (weights, particles, next_key)

#     # scan
#     _, outputs =  lax.scan(_step, carry, jnp.arange(num_timesteps))
#     outputs = swap_axes_on_values(outputs)
 
#     return outputs
