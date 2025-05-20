import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
import jax # type: ignore
from jax import numpy as jnp # type: ignore
import os
from simulators.ssm import SV
from util.bijectors import RealToPSDBijector, VecToCorrMat # type: ignore
from simulators.svssm_params import _init_vals, _param_dists
from util.misc import get_prior_fields, get_bool_tree
from util.param import initialize
import util.io

def setup(state_dim, emission_dim, input_dim, target_vars):

    name = 'svssm'
    param_names = [['mean', 'cov'],
                ['weights', 'bias', 'input_weights', 'cov'],
                ['bias', 'corchol']]
    
    constrainers  = [[None, RealToPSDBijector],
                    [None, None, None, RealToPSDBijector],
                    [None, VecToCorrMat(state_dim)]]

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


def get_ground_truth(state_dim, target_vars=None):
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    if target_vars[0] == 'e2':

        if state_dim == 2:
            
            true_cps = jnp.array([0.57550704])

        if state_dim == 3:

            true_cps = jnp.array([-1.6745086, -1.0824523, 0.34946716])

            # Array([[ 0.99999994, -0.9354245 , -0.8141905 ],
            #        [-0.93542427,  1.        ,  0.68096   ],
            #        [-0.81419045,  0.68096   ,  1.        ]], dtype=float32)

        if state_dim == 4:

            true_cps = jnp.array([1.3273504, 0.80504405, -0.41642225, 0.15039808, -1.1492977, 1.044987])

            # Array([[ 1.        ,  0.65202636,  0.38396442, -0.05389529],
            #        [ 0.6520263 ,  1.        , -0.32844603, -0.71607155],
            #        [ 0.3839644 , -0.32844612,  0.9999999 ,  0.76289207],
            #        [-0.0538953 , -0.7160716 ,  0.76289207,  1.        ]], dtype=float32)
            
    else: 

        raise ValueError('Unknown target variable')

    return true_cps


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    return [-2.5, 2.5]


def get_param_info():

    import simulators.svssm_params as param_info

    return param_info