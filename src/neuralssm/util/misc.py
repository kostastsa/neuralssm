import re
import numpy as np
import jax.numpy as jnp
from jax import lax, vmap, random as jr
from jax.tree_util import tree_map
import util.io
import os
import experiment_descriptor as ed


def remove_whitespace(str):
    """
    Returns the string str with all whitespace removed.
    """

    p = re.compile(r'\s+')
    return p.sub('', str)


def prepare_cond_input(xy, dtype):
    """
    Prepares the conditional input for model evaluation.
    :param xy: tuple (x, y) for evaluating p(y|x)
    :param dtype: data type
    :return: prepared x, y and flag whether single datapoint input
    """

    x, y = xy
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:

        if y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])

    else:

        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], 'wrong sizes'

    return x, y, one_datapoint


def look_up(name):

    lu_field = {'i': '0', 'd': '1', 'e': '2'}
    field_cd = lu_field[name[0]]
    param_cd = str(int(name[1]) - 1)
    
    return field_cd + param_cd


def get_bool_tree(target_vars, param_names):
    r"""Returns a tree of booleans with the same structure as target_vars.
    """
    is_target = []
    target_vars = list(map(lambda var: look_up(var), target_vars))
    for i, field in enumerate(param_names):
        sub_is_target = []
        for j, _ in enumerate(field):
            if str(i) + str(j) in target_vars:
                sub_is_target.append(True)
            else:
                sub_is_target.append(False)
        is_target.append(sub_is_target)
    return is_target


def get_prior_fields(init_vals, param_dists, is_target):    
    r"""Sample parameters from the prior distribution. When prior field is tfd.Distribution,
        sample num_samples values from it, and set the corresponding values of is_constrained to False
        (we consider that parameters are sampled always in unconstrained form). Otherwise set the value 
        equal to the provided value and is_constrained to True.
    """
    return tree_map(lambda b, l1, l2: l2 if b else l1, is_target, init_vals, param_dists)


def swap_axes_on_values(outputs, axis1=0, axis2=1):
    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))


def get_exp_dir(inf,
            sim,
            state_dim,
            emission_dim,
            num_timesteps,
            vars,
            num_samples=None,
            num_rounds=None,
            train_on=None,
            subsample=None,
            num_prt=None,
            qmax=None,
            sigma=None,
            num_iters=None,
            mcmc_steps=None,
            lag=None,
            dhidden=None,
            dt_obs = 0.1):
    
    data_root = '/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm/data/experiments/'

    if inf == 'smc_abc':

        inf_dir = f'abc/smcabc_samples_{num_prt}_qmax_{qmax}_sigma_{sigma}'

    elif inf == 'bpf_mcmc':

        inf_dir = f'mcmc/bpf_numprt_{num_prt}_numiters_{num_iters}_mcmcsteps_{mcmc_steps}'
    
    elif inf == 'snl':

        inf_dir = f'nde/snl/samples_{num_samples}_rounds_{num_rounds}_train_on_{train_on}_mcmc_steps_{mcmc_steps}/maf_nmades_5_dhidden_32_nhiddens_5'

    elif inf == 'tsnl':

        inf_dir = f'nde/tsnl/samples_{num_samples}_rounds_{num_rounds}_lag_{lag}_subsample_{subsample}_train_on_{train_on}_mcmc_steps_{mcmc_steps}/maf_nmades_5_dhidden_{dhidden}_nhiddens_5'

    if sim == 'lgssm':

        sim_dir = data_root + f'{sim}/state-dim_{state_dim}_emission-dim_{emission_dim}_num-timesteps_{num_timesteps}_target-vars_{vars}/'
        
    elif sim == 'svssm':

        sim_dir = data_root + f'{sim}/state-dim_{state_dim}_emission-dim_{emission_dim}_num-timesteps_{num_timesteps}_target-vars_{vars}/'

    elif sim == 'lvssm':

        sim_dir = data_root + f'{sim}/emission-dim_{emission_dim}_num-timesteps_{num_timesteps}_dt_obs_{dt_obs}_target-vars_{vars}/'

    exp_dir = sim_dir + inf_dir

    return exp_dir


def get_exp_data(exp_dir, start_trial, end_trial):

    exp_data = {
        'error': [],
        'rmse': [],
        'results': [],
        'all_dists': [],
        'mll': [],
        'mmd': [],
        'all_emissions': [],
        'gt': [],
        'posterior': [],
        'model': [],
        'props': []
    }
    
    for trial in range(start_trial, end_trial+1):

        print(f'Loading trial {trial}...')

        for entry in exp_data.keys():
            
            try:

                exp_data[entry].append(util.io.load(os.path.join(exp_dir + f'/{trial}', entry)))

            except FileNotFoundError:

                print(f'Entry {entry} not in file.')

                continue

    return exp_data


def str_to_bool(str):

    """
    Converts a string to a boolean value.
    :param str: string to convert
    :return: boolean value
    """

    if str.lower() == 'true':

        return True

    elif str.lower() == 'false':

        return False

    else:
        raise ValueError(f"Invalid boolean string: {str}")