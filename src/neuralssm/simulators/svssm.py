import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
import jax # type: ignore
from jax import numpy as jnp # type: ignore
import os
from simulators.ssm import SV
from dynamax.utils.bijectors import RealToPSDBijector # type: ignore
from simulators.svssm_params import _init_vals, _param_dists
from util.misc import get_prior_fields, get_bool_tree
from util.param import initialize
import util.io



def setup(state_dim, emission_dim, input_dim, target_vars):

    name = 'svssm'
    param_names = [['mean', 'cov'],
                ['weights', 'bias', 'input_weights', 'cov'],
                ['bias', 'cov', 'beta', 'sigma']]
    
    constrainers  = [[None, RealToPSDBijector],
                    [None, None, None, RealToPSDBijector],
                    [None, RealToPSDBijector, None, RealToPSDBijector]]

    emission_dim = state_dim
    init_vals = _init_vals(state_dim, emission_dim, input_dim)
    param_dists = _param_dists(state_dim, emission_dim, input_dim)
    is_target = get_bool_tree(target_vars, param_names)   
    prior_fields = get_prior_fields(init_vals, param_dists, is_target)
    props = initialize(prior_fields, param_names, constrainers)
    out = {}
    out['ssm'] = SV(state_dim, emission_dim, input_dim)
    out['inputs'] = None
    out['props'] = props
    out['exp_info'] = {
        'sim' : 'svssm',
        'state_dim': state_dim,
        'emission_dim': emission_dim,
        'input_dim': input_dim,
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

    return 'data/simulators/svssm'


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    true_ps = jnp.log([0.01, 0.5, 1.0, 0.01])
    obs_xs = util.io.load(os.path.join(get_root(), 'obs_stats'))

    return true_ps, obs_xs


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    return [-2.5, 2.5]


def get_param_info():

    import simulators.svssm_params as param_info

    return param_info