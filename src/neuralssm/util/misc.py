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
            sample_gt,
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
            sampler=None,
            mcmc_steps=None,
            lag=None,
            nmades=None,
            dhidden=None,
            nhiddens=None,
            dt_obs = 0.1):
    
    data_root = '/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm/data/experiments/'

    if inf == 'smc_abc':

        inf_dir = f'abc/smcabc_samples_{num_prt}_qmax_{qmax}_sigma_{sigma}'

    elif inf == 'bpf_mcmc':

        inf_dir = f'mcmc/bpf_numprt_{num_prt}_numiters_{num_iters}_mcmcsteps_{mcmc_steps}'
    
    elif inf == 'snl':

        if sampler is None:

            inf_dir = f'nde/snl/samples_{num_samples}_rounds_{num_rounds}_train_on_{train_on}_mcmc_steps_{mcmc_steps}/maf_nmades_5_dhidden_32_nhiddens_5'

        else:

            inf_dir = f'nde/snl/samples_{num_samples}_rounds_{num_rounds}_train_on_{train_on}_sampler_{sampler}_mcmc_steps_{mcmc_steps}/maf_nmades_5_dhidden_32_nhiddens_5'


    elif inf == 'tsnl':

        if sampler is None:

            inf_dir = f'nde/tsnl/samples_{num_samples}_rounds_{num_rounds}_lag_{lag}_subsample_{subsample}_train_on_{train_on}_mcmc_steps_{mcmc_steps}/maf_nmades_{nmades}_dhidden_{dhidden}_nhiddens_{nhiddens}'

        else:

            inf_dir = f'nde/tsnl/samples_{num_samples}_rounds_{num_rounds}_lag_{lag}_subsample_{subsample}_train_on_{train_on}_sampler_{sampler}_mcmc_steps_{mcmc_steps}/maf_nmades_{nmades}_dhidden_{dhidden}_nhiddens_{nhiddens}'

    if sim == 'lgssm':

        sim_dir = data_root + f'{sim}/state-dim_{state_dim}_emission-dim_{emission_dim}_num-timesteps_{num_timesteps}_target-vars_{vars}/'
        
    elif sim == 'svssm':

        sim_dir = data_root + f'{sim}/state-dim_{state_dim}_emission-dim_{emission_dim}_num-timesteps_{num_timesteps}_target-vars_{vars}/'

    elif sim == 'lvssm':

        sim_dir = data_root + f'{sim}/emission-dim_{emission_dim}_num-timesteps_{num_timesteps}_dt_obs_{dt_obs}_target-vars_{vars}/'

    exp_dir = sim_dir + inf_dir
    exp_dir = os.path.join(exp_dir, f'sample_gt_{sample_gt}')

    return exp_dir


def get_exp_data(exp_dir, start_trial, end_trial):

    exp_data = {
        'kde_error': [],
        'min_error': [],
        'bias': [],
        'num_sims': [],
        'sdev': [],
        'results': [],
        'all_dists': [],
        'mll': [],
        'mmd': [],
        'all_emissions': [],
        'gt': [],
        'posterior': [],
        'losses' : [],
        'gen_sample': [],
        'sample_dists': []
        }
    
    for trial in range(start_trial, end_trial+1):

        for entry in exp_data.keys():
            
            try:

                exp_data[entry].append(util.io.load(os.path.join(exp_dir + f'/{trial}', entry)))

            except FileNotFoundError:

                continue

    return exp_data

def get_data_for(alg, sim_args, inf_args):

    ssm = sim_args['ssm']
    sample_gt = sim_args['sample_gt']
    state_dim = sim_args['state_dim']
    emission_dim = sim_args['emission_dim']
    num_timesteps = sim_args['num_timesteps']
    target_vars = sim_args['target_vars']
    dt_obs = sim_args['dt_obs']
    num_trials = sim_args['num_trials']
    
    if alg == 'smc_abc':

        qmax = inf_args['qmax']
        sigma = inf_args['sigma']
        num_prt = inf_args['num_prt']

        abc_exp_dir = get_exp_dir('smc_abc', 
                                ssm, 
                                sample_gt=sample_gt, 
                                state_dim=state_dim, 
                                emission_dim=emission_dim, 
                                num_timesteps=num_timesteps, 
                                vars=target_vars, 
                                num_prt = num_prt, 
                                qmax= qmax, 
                                sigma=sigma, 
                                dt_obs=dt_obs)
        
        out = get_exp_data(abc_exp_dir, 1, num_trials)

    if alg == 'bpf_mcmc':

        num_prt = inf_args['num_prt']
        n_iters = inf_args['n_iters']
        mcmc_steps = inf_args['mcmc_steps']

        abc_exp_dir = get_exp_dir('bpf_mcmc', 
                                ssm, 
                                sample_gt=sample_gt, 
                                state_dim=state_dim, 
                                emission_dim=emission_dim, 
                                num_timesteps=num_timesteps, 
                                vars=target_vars, 
                                num_prt=num_prt, 
                                num_iters=n_iters, 
                                mcmc_steps=mcmc_steps, 
                                dt_obs=dt_obs)

        out = get_exp_data(abc_exp_dir, 1, num_trials)

    elif alg == 'snl':

        num_samples = inf_args['num_samples']
        sampler = inf_args['sampler']
        mcmc_steps = inf_args['mcmc_steps']
        nmades = inf_args['nmades']
        dhidden = inf_args['dhidden']
        nhiddens = inf_args['nhiddens']
        n_rounds = inf_args['n_rounds']
        train_on = inf_args['train_on']

        snl_exp_dir = get_exp_dir('snl', ssm, 
                                sample_gt=sample_gt, 
                                state_dim=state_dim, 
                                emission_dim=emission_dim, 
                                num_timesteps=num_timesteps, 
                                vars=target_vars, 
                                num_samples=num_samples, 
                                num_rounds=n_rounds, 
                                train_on=train_on, 
                                subsample=None, 
                                sampler=sampler,
                                mcmc_steps=mcmc_steps, 
                                lag=-1, 
                                nmades=nmades, 
                                dhidden=dhidden, 
                                nhiddens=nhiddens, 
                                dt_obs=dt_obs)

        out = get_exp_data(snl_exp_dir, 1, num_trials)

    elif alg == 'tsnl':

        num_samples = inf_args['num_samples']
        sampler = inf_args['sampler']
        mcmc_steps = inf_args['mcmc_steps']
        nmades = inf_args['nmades']
        dhidden = inf_args['dhidden']
        nhiddens = inf_args['nhiddens']
        n_rounds = inf_args['n_rounds']
        train_on = inf_args['train_on']
        lag = inf_args['lag']

        tsnl_exp_dir = get_exp_dir('tsnl', 
                                    ssm, 
                                    sample_gt=sample_gt, 
                                    state_dim=state_dim, 
                                    emission_dim=emission_dim, 
                                    num_timesteps=num_timesteps, 
                                    vars=target_vars, 
                                    num_samples=num_samples, 
                                    num_rounds=n_rounds, 
                                    train_on=train_on, 
                                    subsample=1.0, 
                                    sampler=sampler,
                                    mcmc_steps=mcmc_steps, 
                                    lag=lag, nmades=nmades, 
                                    dhidden=dhidden, 
                                    nhiddens=nhiddens, 
                                    dt_obs=dt_obs)

        out = get_exp_data(tsnl_exp_dir, 1, num_trials)

    return out


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
    

def print_table(rows):

    # Find max width for each column
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    row_names = ['ABC', 'MCMC', 'SNL', 'TSNL']
    
    for row_name, row in zip(row_names, rows):
        print(row_name.ljust(5) + " | " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))