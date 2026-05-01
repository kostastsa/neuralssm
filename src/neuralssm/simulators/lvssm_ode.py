import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
import jax # type: ignore
from jax import numpy as jnp # type: ignore
from jax import debug
import os
from simulators.ssm import PolynomialSDE
from util.bijectors import RealToPSDBijector # type: ignore
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
from util.misc import get_prior_fields, get_bool_tree
from util.param import initialize
from simulators.lvssm_ode_params import _init_vals, _param_dists
import util.io

def emission_dist(params, state):
    '''
    Binomial distribution
    '''

    loc = jnp.log(state)
    scale = jnp.array(params.emissions.alpha.value * state.shape[0])

    return tfd.LogNormal(loc=loc, scale=scale)

def lotka_volterra_drift(x: jnp.ndarray,
                        params) -> jnp.ndarray:

    alpha, beta, delta, gamma  = params.dynamics.rates.value

    dx0 = alpha * x[0] - beta * x[0] * x[1]
    dx1 = delta * x[0] * x[1] - gamma * x[1]

    return jnp.array([dx0, dx1])

def lotka_volterra_diffusion(x: jnp.ndarray,
                            params) -> jnp.ndarray:

    sigma = params.dynamics.sigma.value

    return sigma * jnp.eye(2)  # 2x2 identity scaled

def setup(state_dim, emission_dim, input_dim, target_vars, dt_obs=0.1):

    # Initialize model and simulate dataset
    param_names = [['mean', 'cov'],
                   ['rates', 'sigma'],
                   ['alpha']]

    constrainers  = [[None, RealToPSDBijector],
                     [None, tfb.Exp()],
                     [None]]

    init_vals = _init_vals(emission_dim)
    param_dists = _param_dists(emission_dim)

    is_target = get_bool_tree(target_vars, param_names)  
    prior_fields = get_prior_fields(init_vals, param_dists, is_target)

    props = initialize(prior_fields, param_names, constrainers)
    out = {}

    out['ssm'] = PolynomialSDE(
                    drift_fn=lotka_volterra_drift,
                    diffusion_fn=lotka_volterra_diffusion,
                    emission_dim=emission_dim,
                    emission_dist=emission_dist,
                    dt=1e-2)

    out['inputs'] = None
    out['props'] = props
    out['exp_info'] = {
        'sim' : 'lvssm_ode',
        'state_dim': 2,
        'emission_dim': emission_dim,
        'input_dim': 0,
        'is_target': is_target,
        'param_names': param_names,
        'constrainers': constrainers,
        'prior_fields': prior_fields        
    }
    
    return out


def get_root():
    """
    Returns the root folder.
    """

    return 'data/simulators/lgssm'


def get_ground_truth(state_dim, target_vars=None):
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    true_cps = jnp.array([10.0, 0.4, 0.1, 0.4])

    return true_cps


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    return [-5.0, 2.0]


def get_param_info():

    import simulators.lvssm_params as param_info

    return param_info