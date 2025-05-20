import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
import jax # type: ignore
from jax import numpy as jnp # type: ignore
import os
from simulators.ssm import SPN
from util.bijectors import RealToPSDBijector # type: ignore
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
from util.misc import get_prior_fields, get_bool_tree
from util.param import initialize
from simulators.lvssm_params import _init_vals, _param_dists
import util.io

def emission_dist(params, state):
    return tfd.MultivariateNormalFullCovariance(loc=state[1], covariance_matrix=params.emissions.cov.value)

def setup(state_dim, emission_dim, input_dim, target_vars, dt_obs=0.1):

    name = 'lvssm'
    # Initialize model and simulate dataset
    param_names = [['mean', 'cov'],
                   ['pre', 'post', 'log_rates'],
                   ['cov']]

    constrainers  = [[None, None],
                     [None, None, None],
                     [RealToPSDBijector]] # bijectors here defined unconstrained -> constrained

    num_reactions = 4
    init_vals = _init_vals(emission_dim)
    param_dists = _param_dists(emission_dim, num_reactions)

    is_target = get_bool_tree(target_vars, param_names)   
    prior_fields = get_prior_fields(init_vals, param_dists, is_target)

    props = initialize(prior_fields, param_names, constrainers)
    out = {}
    out['ssm'] = SPN(2, num_reactions, emission_dim, emission_dist, 0, dt_obs)
    out['inputs'] = None
    out['props'] = props
    out['exp_info'] = {
        'sim' : 'lvssm',
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


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    true_cps = jnp.array([jnp.log(0.01), jnp.log(0.5), jnp.log(1), jnp.log(0.01)])

    return true_cps


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    return [-5.0, 2.0]


def get_param_info():

    import simulators.lvssm_params as param_info

    return param_info