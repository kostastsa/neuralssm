import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
import jax # type: ignore
from jax import numpy as jnp # type: ignore
import os
from simulators.ssm import LGSSM
from simulators.lgssm_params import _init_vals, _param_dists
from util.misc import get_prior_fields, get_bool_tree
from util.param import initialize
from util.bijectors import PSDToRealBijector, RealToPSDBijector, RealVecToStableMat # type: ignore
import util.io



def setup(state_dim, emission_dim, input_dim, target_vars):

    param_names = [['mean', 'cov'],
                ['weights', 'bias', 'input_weights', 'cov'],
                ['weights', 'bias', 'input_weights', 'cov']]
    
    constrainers  = [[None, RealToPSDBijector],
                    [RealVecToStableMat(state_dim), None, None, RealToPSDBijector],
                    [None, None, None, RealToPSDBijector]]
    
    init_vals = _init_vals(state_dim, emission_dim, input_dim)
    param_dists = _param_dists(state_dim, emission_dim, input_dim)
    is_target = get_bool_tree(target_vars, param_names)   
    prior_fields = get_prior_fields(init_vals, param_dists, is_target)
    props = initialize(prior_fields, param_names, constrainers)
    out = {}
    out['ssm'] = LGSSM(state_dim, emission_dim, input_dim)
    out['inputs'] = None
    out['props'] = props
    out['exp_info'] = {
        'sim' : 'lgssm',
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

    return 'data/simulators/lgssm'


def get_ground_truth(state_dim, target_vars=None):
    """
    Returns ground truth parameters and corresponding observed statistics.
    """
    true_cps_dict = {}
    true_cps_dict['d1'] = (0.05268028 * jnp.eye(state_dim)).reshape((state_dim ** 2, ))
    true_cps_dict['d4'] = PSDToRealBijector()(0.1 * jnp.eye(state_dim))
    
    true_cps = []

    for vars in target_vars:

        true_cps.append(true_cps_dict[vars]) 

    true_cps = jnp.concatenate(true_cps, axis=0)

    return true_cps


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    return [-2.5, 2.5]


def get_param_info():
    import simulators.lgssm_params as param_info
    return param_info