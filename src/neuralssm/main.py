# This module is taken/adapted from the repository ([https://github.com/gpapamak/snl.git])
# Originally authored by George Papamakarios, under the MIT License
import argparse
import os
import jax
import jax.numpy as jnp

from flax import nnx

import gc
import util.io
from util.plot import get_acf_plot
from functools import partial

import os
import time
import subprocess

from experiment_runner import ExperimentRunner
import experiment_descriptor as ed
import experiment_results as er
import jax.random as jr
import misc

import util.numerics
from util.sample import sim_emissions, generate_emissions
from util.train import find_mle, loglik_fn
from inference.diagnostics.two_sample import sq_maximum_mean_discrepancy as mmd
import matplotlib.pyplot as plt
import matplotlib_inline # type: ignore
import scienceplots # type: ignore

plt.style.use(['science', 'ieee'])
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

fig_dir = '/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm/figures'

def parse_args():
    """
    Returns an object describing the command line.
    """

    def int_or_str(value):
        """Try converting to int, otherwise return as string."""
        try:
            return int(value)
        except ValueError:
            return value

    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments.')
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run', help='run experiments')
    parser_run.add_argument('seed', type=int_or_str, help='seed for PRNG')
    parser_run.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_run.set_defaults(func=run_experiment)

    parser_trials = subparsers.add_parser('trials', help='run multiple experiment trials')
    parser_trials.add_argument('seed', type=int_or_str, help='seed for PRNG')
    parser_trials.add_argument('start', type=int, help='# of first trial')
    parser_trials.add_argument('end', type=int, help='# of last trial')
    parser_trials.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_trials.set_defaults(func=run_trials)

    parser_view = subparsers.add_parser('view', help='view results')
    parser_view.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_view.add_argument('-b', '--block', action='store_true', help='block execution after viewing each experiment')
    parser_view.add_argument('-o', '--overwrite', type=str)
    parser_view.add_argument('-t', '--trial', type=int, default=0, help='trial to view (default is 0)')
    parser_view.set_defaults(func=view_results)

    parser_ensemble = subparsers.add_parser('ensemble', help='view ensemble results')
    parser_ensemble.add_argument('start', type=int, help='# of first trial')
    parser_ensemble.add_argument('end', type=int, help='# of last trial')
    parser_ensemble.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_ensemble.set_defaults(func=view_ensemble)

    parser_grouped = subparsers.add_parser('grouped', help='view grouped results - error vs num_sims')
    parser_grouped.add_argument('group_by', type=str, help='variable to group by')
    parser_grouped.add_argument('start', type=int, help='# of first trial')
    parser_grouped.add_argument('end', type=int, help='# of last trial')
    parser_grouped.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_grouped.set_defaults(func=view_grouped)

    parser_errors = subparsers.add_parser('errors', help='plot error by variable and inference method')
    parser_errors.add_argument('start', type=int, help='# of first trial')
    parser_errors.add_argument('end', type=int, help='# of last trial')
    parser_errors.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_errors.set_defaults(func=eval_and_plot_errors)

    parser_mmd = subparsers.add_parser('mmd', help='plot error by variable and inference method')
    parser_mmd.add_argument('start', type=int, help='# of first trial')
    parser_mmd.add_argument('end', type=int, help='# of last trial')
    parser_mmd.add_argument('num_samples', type=int, help='# number of samples to estimate MMD')
    parser_mmd.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_mmd.set_defaults(func=plot_mmd)

    parser_plot_by_var = subparsers.add_parser('plotvar', help='plot error by variable and inference method')
    parser_plot_by_var.add_argument('varX', type=str, help='variable to plot by')
    parser_plot_by_var.add_argument('varY', type=str, help='variable to plot')
    parser_plot_by_var.add_argument('start', type=int, help='# of first trial')
    parser_plot_by_var.add_argument('end', type=int, help='# of last trial')
    parser_plot_by_var.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_plot_by_var.set_defaults(func=plot_by_var)

    parser_plot_by_var = subparsers.add_parser('modsel', help='plot error by variable and inference method')
    parser_plot_by_var.add_argument('varX', type=str, help='variable to plot by')
    parser_plot_by_var.add_argument('start', type=int, help='# of first trial')
    parser_plot_by_var.add_argument('end', type=int, help='# of last trial')
    parser_plot_by_var.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_plot_by_var.set_defaults(func=model_selection)

    parser_plot_dist = subparsers.add_parser('plot_distance', help='plot distance between true and simulated datasets, per round')
    parser_plot_dist.add_argument('start', type=int, help='# of first trial')
    parser_plot_dist.add_argument('end', type=int, help='# of last trial')
    parser_plot_dist.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_plot_dist.set_defaults(func=plot_dist)

    parser_log = subparsers.add_parser('log', help='print experiment logs')
    parser_log.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_log.set_defaults(func=print_log)

    return parser.parse_args()


def run_experiment(args):
    """
    Runs experiments.
    """

    from experiment_runner import ExperimentRunner
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    seed = int(time.time() * 1000) if args.seed=='r' else int(args.seed)  # Milliseconds for more granularity
    key = jr.PRNGKey(seed)

    for exp_desc in exp_descs:

        try:
        
            ExperimentRunner(exp_desc).run(trial=0, sample_gt=True, plot_sims=False, key=key)

        except misc.AlreadyExistingExperiment:
            print('EXPERIMENT ALREADY EXISTS')

    print('ALL DONE')


def run_trials(args):
    """
    Runs experiments for multiple trials with random ground truth.
    """

    if args.start < 1:
        raise ValueError('trial # must be a positive integer')

    if args.end < args.start:
        raise ValueError('end trial can''t be less than start trial')

    for file in args.files:

        file_path = os.getcwd() + '/' + file
        subprocess.run(["open", file_path])
    
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    seed = int(time.time() * 1000) if args.seed=='r' else args.seed  
    key = jr.PRNGKey(seed)
    
    for exp_desc in exp_descs:

        for trial in range(args.start, args.end + 1):

            exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
            trial_dir = os.path.join(exp_dir, str(trial))

            if os.path.exists(trial_dir + '/exp_desc.txt') | os.path.exists(trial_dir + '/info.txt'):

                print('EXPERIMENT ALREADY EXISTS')
                jax.clear_backends()
                gc.collect()
                
                continue

            runner = ExperimentRunner(exp_desc)

            try:

                if os.path.exists(trial_dir):
                    
                    for root, _, files in os.walk(trial_dir, topdown=False):

                        for name in files:

                            os.remove(os.path.join(root, name))

                    os.rmdir(trial_dir)

                key, subkey = jr.split(key)
                out = runner.run(trial=trial, sample_gt=True, plot_sims=False, key=subkey, seed=seed)

            except misc.AlreadyExistingExperiment:

                print('RUNNER FAILED')

            out = None
            jax.clear_backends()
            gc.collect()

    print('ALL DONE')


def view_results(args):
    """
    Views experiments.
    """

    from experiment_viewer import ExperimentViewer, plt

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    if args.overwrite=='false':

        overwrite = False

    else:

        overwrite = True

    for exp_desc in exp_descs:

        try:

            ExperimentViewer(exp_desc, overwrite).view_results(trial=args.trial, block=args.block)

        except misc.NonExistentExperiment:

            print('EXPERIMENT DOES NOT EXIST')


def view_ensemble(args, show=False):
    """
    Takes a set of experiments and line-plots the errors vs num_sims of each algorithm together.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    abc = er.ABC_Results()
    mcmc = er.MCMC_Results()
    snl = er.SNL_Results()
    tsnl = er.TSNL_Results()

    for exp_desc in exp_descs:

        kderr_array = []
        minerr_array = []
        pmerr_array = []
        rmse_array = []
        mmd_array = []

        get_mmd = False

        if isinstance(exp_desc.inf, ed.SNL_Descriptor) or isinstance(exp_desc.inf, ed.TSNL_Descriptor):

            get_mmd = False

        for trial in range(args.start, args.end + 1):

            try:

                print('ON TRIAL:', trial)
                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir += '/' + str(trial)

                try:

                    print(exp_desc.pprint())

                    state_dim = exp_desc.sim.state_dim  
                    emission_dim = exp_desc.sim.emission_dim

                    num_sims = util.io.load(os.path.join(exp_dir, 'num_sims'))
                    kderr = util.io.load(os.path.join(exp_dir, 'kde_error'))
                    minerr = util.io.load(os.path.join(exp_dir, 'min_error'))
                    pmerr = util.io.load(os.path.join(exp_dir, 'post_mean_error'))
                    rmserr = util.io.load(os.path.join(exp_dir, 'rmse')) 

                    kderr_array.append(kderr)
                    minerr_array.append(minerr)
                    pmerr_array.append(pmerr)
                    rmse_array.append(rmserr)

                    if get_mmd:

                        mmd = util.io.load(os.path.join(exp_dir, 'mmd'))
                        mmd_array.append(mmd)

                except FileNotFoundError:
                    print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')

        kderr_array = jnp.array(kderr_array)
        minerr_array = jnp.array(minerr_array)
        pmerr_array = jnp.array(pmerr_array)
        rmse_array = jnp.array(rmse_array)

        key, subkey = jr.split(key)
        kderr, _ = util.numerics.bootstrap(subkey, kderr_array, 100)
        mean_kderr = jnp.mean(kderr)
        std_kderr = jnp.std(kderr)

        key, subkey = jr.split(key)
        minerr, _ = util.numerics.bootstrap(subkey, minerr_array, 100)
        mean_minerr = jnp.mean(minerr)
        std_minerr = jnp.std(minerr)

        key, subkey = jr.split(key)
        pmerr, _ = util.numerics.bootstrap(subkey, pmerr_array, 100)
        mean_pmerr = jnp.mean(pmerr)
        std_pmerr = jnp.std(pmerr)

        key, subkey = jr.split(key)
        rmserr, _ = util.numerics.bootstrap(subkey, rmse_array, 100)
        mean_rmse = jnp.mean(rmserr)
        std_rmse = jnp.std(rmserr)

        mmd_array = jnp.array(mmd_array)
        nans_infs = jnp.isnan(mmd_array) + jnp.isinf(mmd_array)
        mmd_array = mmd_array[~nans_infs]
        bootstrap_mmd, _ = util.numerics.bootstrap(subkey, mmd_array, 100)
        mean_mmd = jnp.mean(bootstrap_mmd)
        std_mmd = jnp.std(bootstrap_mmd)


        if isinstance(exp_desc.inf, ed.ABC_Descriptor):
            abc.results.append([num_sims, mean_kderr, std_kderr, mean_minerr, std_minerr, mean_pmerr, std_pmerr, mean_rmse, std_rmse])
        
        elif isinstance(exp_desc.inf, ed.BPF_MCMC_Descriptor):
            mcmc.results.append([num_sims, mean_kderr, std_kderr, mean_minerr, std_minerr, mean_pmerr, std_pmerr, mean_rmse, std_rmse])

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            snl.results.append([num_sims, mean_kderr, std_kderr, mean_minerr, std_minerr, mean_pmerr, std_pmerr, mean_rmse, std_rmse])

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):
            tsnl.results.append([num_sims, mean_kderr, std_kderr, mean_minerr, std_minerr, mean_pmerr, std_pmerr, mean_rmse, std_rmse])

    abc.make_jnp()
    mcmc.make_jnp()
    snl.make_jnp()
    tsnl.make_jnp()

    fig, ax = plt.subplots(4 + int(get_mmd), 1, figsize=(5, 5))

    for alg in [abc, mcmc, snl, tsnl]:

        try:

            ax[0].plot(alg.results[:, 0], alg.results[:, 1], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[0].fill_between(alg.results[:, 0], alg.results[:,1]-alg.results[:, 2], alg.results[:, 1] + alg.results[:, 2], color=alg.color, alpha=0.05)
            ax[0].set_ylabel('$\mathcal{E}_{KDE}$')
            ax[0].legend(prop={'size': 5})

            ax[1].plot(alg.results[:, 0], alg.results[:, 3], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[1].fill_between(alg.results[:, 0], alg.results[:, 3]-alg.results[:, 4], alg.results[:, 3] + alg.results[:, 4], color=alg.color, alpha=0.05)
            ax[1].set_xlabel('log Number of simulations')
            ax[1].set_ylabel('$\mathcal{E}_{min}$')
            ax[1].legend(prop={'size': 5})

            ax[2].plot(alg.results[:, 0], alg.results[:, 5], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[2].fill_between(alg.results[:, 0], alg.results[:, 5]-alg.results[:, 6], alg.results[:, 5] + alg.results[:, 6], color=alg.color, alpha=0.05)
            ax[2].set_xlabel('log Number of simulations')
            ax[2].set_ylabel('$\mathcal{E}_{PM}$')
            ax[2].legend(prop={'size': 5})

            ax[3].plot(alg.results[:, 0], alg.results[:, 7], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[3].fill_between(alg.results[:, 0], alg.results[:, 7]-alg.results[:, 8], alg.results[:, 7] + alg.results[:, 8], color=alg.color, alpha=0.05)
            ax[3].set_xlabel('log Number of simulations')
            ax[3].set_ylabel('$\mathcal{E}_{RMSE}$')
            ax[3].legend(prop={'size': 5})

            # if isinstance(alg, er.SNL_Results) or isinstance(alg, er.TSNL_Results):

            #     print(f'{alg.name}')
            #     print(alg.results[:, 5])

            #     ax[2].plot(alg.results[:, 0], alg.results[:, 5], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            #     ax[2].fill_between(alg.results[:, 0], alg.results[:, 5]-alg.results[:, 6], alg.results[:, 5] + alg.results[:, 6], color=alg.color, alpha=0.05)

            #     if variable_names[0] == 'state_dim':

            #         ax[2].set_xlabel('dim')

            #     else:

            #         ax[2].set_xlabel('log Number of simulations')

            #     ax[2].set_ylabel('$\mathcal{E}_{MMD}$')
            #     ax[2].legend(prop={'size': 5})

            # ax[1 + int(get_mmd)].set_xlabel('log Number of simulations')

        except IndexError:

            print(f'No {alg.name} results to plot')

        except TypeError:
            
            print(f'{alg.name} has no results')

    fig.savefig(os.path.join(fig_dir, f'{exp_desc.sim.name}_dim_{state_dim}_scan_num_sims_' + '_'.join(exp_desc.sim.target_vars) + '.pdf'), format="pdf", dpi=800)

    if show:

        plt.show()


def view_grouped(args, show=False):
    """
    Takes a set of experiments and line-plots the errors of each algorithm together.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    group_by = args.group_by
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    groups = {'abc': {'label': 'SMC-ABC'},
              'bpf_mcmc': {'label': 'BPF-MCMC'},
              'snl': {'label': 'SNL'},
              'tsnl': {'label': 'T-SNL'}
              }

    group_by_vals = []

    for exp_desc in exp_descs:

        try:

            group_var_val = exp_desc.inf.__dict__[group_by]

        except KeyError:

            print(f'Variable {group_by} not found in experiment descriptor')
        
        error_trials = []
        num_sims_trials = []
        group_by_vals.append(group_var_val)

        for trial in range(args.start, args.end + 1):

            print(exp_desc.pprint())

            try:

                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir += '/' + str(trial)

                try:

                    error, num_sims = util.io.load(os.path.join(exp_dir, 'error'))
                    error_trials.append(error)
                    num_sims_trials.append(num_sims)

                except FileNotFoundError:
                    print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')

        B = 100
        key, subkey = jr.split(key)
        bootstrap_error, _ = util.numerics.bootstrap(subkey, jnp.array(error_trials), B)
        error_avg = jnp.mean(bootstrap_error)
        error_std = jnp.std(bootstrap_error)
        num_sims_avg = jnp.array(num_sims_trials).mean()
            
        if isinstance(exp_desc.inf, ed.ABC_Descriptor):

            if group_var_val not in groups['abc'].keys():

                groups['abc'][group_var_val] = {'varX': [], 'varY': [], 'error_std': []}

            groups['abc'][group_var_val]['varX'].append(num_sims_avg)
            groups['abc'][group_var_val]['varY'].append(error_avg)
            groups['abc'][group_var_val]['error_std'].append(error_std)
        
        elif isinstance(exp_desc.inf, ed.BPF_MCMC_Descriptor):

            if group_var_val not in groups['bpf_mcmc'].keys():

                groups['bpf_mcmc'][group_var_val] = {'varX': [], 'varY': [], 'error_std': []}

            groups['bpf_mcmc'][group_var_val]['varX'].append(num_sims_avg)
            groups['bpf_mcmc'][group_var_val]['varY'].append(error_avg)
            groups['bpf_mcmc'][group_var_val]['error_std'].append(error_std)

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):

            if group_var_val not in groups['snl'].keys():

                groups['snl'][group_var_val] = {'varX': [], 'varY': [], 'error_std': []}

            groups['snl'][group_var_val]['varX'].append(num_sims_avg)
            groups['snl'][group_var_val]['varY'].append(error_avg)
            groups['snl'][group_var_val]['error_std'].append(error_std)

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):

            if group_var_val not in groups['tsnl'].keys():

                groups['tsnl'][group_var_val] = {'varX': [], 'varY': [], 'error_std': []}

            groups['tsnl'][group_var_val]['varX'].append(num_sims_avg)
            groups['tsnl'][group_var_val]['varY'].append(error_avg)
            groups['tsnl'][group_var_val]['error_std'].append(error_std)

    for group in groups.keys():

        for key, val in groups[group].items():

            if key != 'label':

                val['varX'] = jnp.array(val['varX'])
                val['varY'] = jnp.array(val['varY'])
                val['error_std'] = jnp.array(val['error_std'])

    for group in groups.keys():

        try:

            for group_var_val in set(group_by_vals):

                plt.plot(groups[group][group_var_val]['varX'], groups[group][group_var_val]['varY'], 'o-', markersize=5, label=f'{1}, {group_by}={group_var_val}')
                plt.fill_between(groups[group][group_var_val]['varX'], groups[group][group_var_val]['varY'] - groups[group][group_var_val]['error_std'], groups[group][group_var_val]['varY'] + groups[group][group_var_val]['error_std'], alpha=0.2, label='Confidence Band')

        except IndexError:

            print(f'No results to plot')

        except KeyError:

            print(f'No variable {group_by}')

    plt.xlabel('Number of simulations')
    plt.ylabel('-log pdf at true parameters')
    plt.legend(prop={'size': 5})
    plt.savefig(os.path.join(fig_dir, f'Error-num_sims for {group_by}.png'))

    if show: 

        plt.show()


def eval_and_plot_errors(args, show=False):
    """
    Takes a set of experiments and line-plots the errors of each algorithm together.
    """

    from experiment_viewer import plt
    from util.numerics import kde_error, min_error, post_mean_error, rmse

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        kderr_trials = []
        minerr_trials = []
        pmerr_trials = []
        rmse_trials = []
        inf_desc = exp_desc.inf
        sim_desc = exp_desc.sim

        for trial in range(args.start, args.end + 1):
            
            print(f'Working on trial {trial}')
            print(exp_desc.pprint())

            try:

                exp_root = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir = exp_root + '/' + str(trial)

                try:

                    (_, true_cps), observations = util.io.load(os.path.join(exp_dir, 'gt'))

                    if isinstance(inf_desc, ed.ABC_Descriptor):

                        results = util.io.load(os.path.join(exp_dir, 'results'))
                        samples, weights, counts = results
                        kderr = kde_error(samples, true_cps)
                        minerr = min_error(samples, true_cps)
                        pmerr = post_mean_error(samples, true_cps)
                        rmserr = rmse(samples, true_cps)
                        num_simulations = counts * sim_desc.num_timesteps

                    elif isinstance(inf_desc, ed.BPF_MCMC_Descriptor):

                        mcmc_samples, _ = util.io.load(os.path.join(exp_dir, 'results'))
                        kderr = kde_error(mcmc_samples, true_cps)
                        minerr = min_error(mcmc_samples, true_cps)
                        pmerr = post_mean_error(mcmc_samples, true_cps)
                        rmserr = rmse(mcmc_samples, true_cps)
                        num_simulations = inf_desc.num_prt * sim_desc.num_timesteps * inf_desc.mcmc_steps * inf_desc.num_iters

                    elif isinstance(inf_desc, ed.SNL_Descriptor):

                        _, posterior_cond_sample = util.io.load(os.path.join(exp_dir, 'posterior'))
                        kderr = kde_error(posterior_cond_sample, true_cps)
                        minerr = min_error(posterior_cond_sample, true_cps)
                        pmerr = post_mean_error(posterior_cond_sample, true_cps)
                        rmserr = rmse(posterior_cond_sample, true_cps)
                        num_simulations = inf_desc.n_rounds * inf_desc.n_samples * sim_desc.num_timesteps
                            
                    elif isinstance(inf_desc, ed.TSNL_Descriptor):

                        _, posterior_cond_sample = util.io.load(os.path.join(exp_dir, 'posterior'))
                        kderr = kde_error(posterior_cond_sample, true_cps)
                        minerr = min_error(posterior_cond_sample, true_cps)
                        pmerr = post_mean_error(posterior_cond_sample, true_cps)
                        rmserr = rmse(posterior_cond_sample, true_cps)
                        num_simulations = inf_desc.n_rounds * inf_desc.n_samples * sim_desc.num_timesteps

                    util.io.save(kderr, os.path.join(exp_dir, 'kde_error'))
                    util.io.save(minerr, os.path.join(exp_dir, 'min_error'))
                    util.io.save(pmerr, os.path.join(exp_dir, 'post_mean_error'))
                    util.io.save(rmserr, os.path.join(exp_dir, 'rmse'))
                    util.io.save(jnp.log(num_simulations), os.path.join(exp_dir, 'num_sims'))

                    kderr_trials.append(kderr)
                    minerr_trials.append(minerr)
                    pmerr_trials.append(pmerr)
                    rmse_trials.append(rmserr)

                except FileNotFoundError:
                    print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')
            
        if isinstance(exp_desc.inf, ed.ABC_Descriptor):
            label = 'SMC-ABC'
        
        elif isinstance(exp_desc.inf, ed.BPF_MCMC_Descriptor):
            label = 'BPF-MCMC'

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            label = 'SNL'

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):
            label = 'T-SNL'

        try:

            kderr_trials = jnp.array(kderr_trials)
            minerr_trials = jnp.array(minerr_trials)
            pmerr_trials = jnp.array(pmerr_trials)
            rmse_trials = jnp.array(rmse_trials)

            key, subkey = jr.split(key)
            kderr, _ = util.numerics.bootstrap(subkey, kderr_trials, 100)
            kderr_avg = jnp.mean(kderr)
            kderr_std = jnp.std(kderr)

            key, subkey = jr.split(key)
            minerr, _ = util.numerics.bootstrap(subkey, minerr_trials, 100)
            minerr_avg = jnp.mean(minerr)
            minerr_std = jnp.std(minerr)

            key, subkey = jr.split(key)
            pmerr, _ = util.numerics.bootstrap(subkey, pmerr_trials, 100)
            pmerr_avg = jnp.mean(pmerr)
            pmerr_std = jnp.std(pmerr)

            key, subkey = jr.split(key)
            rmserr, _ = util.numerics.bootstrap(subkey, rmse_trials, 100)
            rmse_avg = jnp.mean(rmserr)
            rmse_std = jnp.std(rmserr)

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            ax[0, 0].hist(kderr_trials, bins=50, alpha=0.5, label=f'{label}', density=True)
            ax[0, 0].axvline(kderr_avg, color='green', linestyle='dashed', linewidth=0.5)
            ax[0, 0].axvline(kderr_avg - kderr_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[0, 0].axvline(kderr_avg + kderr_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[0, 0].set_xlabel('kderr')
            ax[0, 0].set_ylabel('count')
            ax[0, 0].set_title(f'{label} - E_KDE histogram %')
            ax[0, 0].legend(prop={'size': 5})

            ax[0, 1].hist(minerr_trials, bins=50, alpha=0.5, label=f'{label}', density=True)
            ax[0, 1].axvline(minerr_avg, color='green', linestyle='dashed', linewidth=0.5)
            ax[0, 1].axvline(minerr_avg - minerr_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[0, 1].axvline(minerr_avg + minerr_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[0, 1].set_xlabel('minerr')
            ax[0, 1].set_ylabel('count')
            ax[0, 1].set_title(f'{label} - E_min histogram %')
            ax[0, 1].legend(prop={'size': 5})

            ax[1, 0].hist(pmerr_trials, bins=50, alpha=0.5, label=f'{label}', density=True)
            ax[1, 0].axvline(pmerr_avg, color='green', linestyle='dashed', linewidth=0.5)
            ax[1, 0].axvline(pmerr_avg - pmerr_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 0].axvline(pmerr_avg + pmerr_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 0].set_xlabel('post_mean_err')
            ax[1, 0].set_ylabel('count')
            ax[1, 0].set_title(f'{label} - E_PME histogram %')
            ax[1, 0].legend(prop={'size': 5})

            ax[1, 1].hist(rmse_trials, bins=50, alpha=0.5, label=f'{label}', density=True)
            ax[1, 1].axvline(rmse_avg, color='green', linestyle='dashed', linewidth=0.5)
            ax[1, 1].axvline(rmse_avg - rmse_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 1].axvline(rmse_avg + rmse_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 1].set_xlabel('rmse')
            ax[1, 1].set_ylabel('count')
            ax[1, 1].set_title(f'{label} - E_RMSE histogram %')
            ax[1, 1].legend(prop={'size': 5})

            fig.savefig(os.path.join(f'{exp_root}', f'Error histograms.png'))

            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(kderr_trials, label='kderr')
            ax.plot(minerr_trials, label='minerr')
            ax.plot(pmerr_trials, label='pmerr')
            ax.plot(rmse_trials, label='rmse')

            ax.set_xlabel('Trial')
            ax.set_ylabel('Error')
            ax.set_title(f'{label} - Errors vs Trial')
            ax.legend(prop={'size': 5})
            fig.savefig(os.path.join(f'{exp_root}', f'Errors vs Trials.png'))

        except FileNotFoundError:

            print('Experiment Empty')
    
    if show: 

       plt.show()


def plot_by_var(args, show=False):
    """
    Takes a set of experiments and line-plots the errors of each algorithm together.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    varX = args.varX
    varY = args.varY

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    groups = {'abc': {'means': [], 'stds': []}, 
              'bpf_mcmc': {'means': [], 'stds': []}, 
              'snl': {'means': [], 'stds': []}, 
              'tsnl': {'means': [], 'stds': []}}
    valsX = []

    for exp_desc in exp_descs:

        print(exp_desc.pprint())

        try:

            valX = exp_desc.inf.__dict__[varX]

        except KeyError:

            print(f'Variable {varX} not found in experiment descriptor')
        
        valsX.append(valX)
        valsY = []

        for trial in range(args.start, args.end + 1):

            try:

                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir += '/' + str(trial)
                valsY.append(util.io.load(os.path.join(exp_dir, varY)))

            except misc.AlreadyExistingExperiment:

                print('TRIAL DOES NOT EXIST')

            except FileNotFoundError:

                print('ERROR FILE NOT')

                continue

        B = 100
        key, subkey = jr.split(key)
        bootstrap, _ = util.numerics.bootstrap(subkey, jnp.array(valsY), B)
        # mean = - 2 * jnp.mean(bootstrap) +  480  * valX 
        mean = jnp.mean(bootstrap)
        std = jnp.std(bootstrap)

        if isinstance(exp_desc.inf, ed.ABC_Descriptor):
            groups['abc']['means'].append(mean)
            groups['abc']['stds'].append(std)
        
        elif isinstance(exp_desc.inf, ed.BPF_MCMC_Descriptor):
            groups['bpf_mcmc']['means'].append(mean)
            groups['bpf_mcmc']['stds'].append(std)

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            groups['snl']['means'].append(mean)
            groups['snl']['stds'].append(std)

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):
            groups['tsnl']['means'].append(mean)
            groups['tsnl']['stds'].append(std)

    valsX = jnp.array(valsX)

    for group in groups.keys():

        groups[group]['means'] = jnp.array(groups[group]['means'])
        groups[group]['stds'] = jnp.array(groups[group]['stds'])

    for group in groups.keys():

        try: 

            plt.plot(valsX, groups[group]['means'], 'o-', markersize=5, label=f'{group}')
            plt.fill_between(valsX, groups[group]['means']-groups[group]['stds'], groups[group]['means']+groups[group]['stds'], alpha=0.2, label='Confidence Band')

        except IndexError:      
                
            print(f'No {group} results to plot')

        except KeyError:

            print(f'{group} has no variable {varX}')

        except TypeError:

            print(f'{group} has no results')

        except ValueError:

            print(f'{group} has no results')

    plt.xlabel(varX)
    # plt.xticks(valsX)
    plt.ylabel('-log pdf at true parameters')
    plt.legend(prop={'size': 5})
    plt.savefig(os.path.join(fig_dir, f'{varX}-{varY}.png'))
    if show: 
        plt.show()


def plot_dist(args, show=False):
    """
    Plots the distance between the true and sampled emissions, averages over trials.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        n_rounds = exp_desc.inf.n_rounds
        dists_trials = []

        for trial in range(args.start, args.end + 1):

            print(exp_desc.pprint())

            try:

                exp_root = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir = exp_root + '/' + str(trial)

                try:

                    all_dists = util.io.load(os.path.join(exp_dir, 'all_dists'))
                    all_dists = jnp.mean(jnp.array(all_dists), axis=1)
                    dists_trials.append(all_dists)

                except FileNotFoundError:
                    print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')

        dist_trials = jnp.array(dists_trials)
        assert n_rounds == dist_trials.shape[1]

        B = 100
        key, subkey = jr.split(key)
        _, boot_sample = util.numerics.bootstrap(subkey, jnp.array(dist_trials), B)
        boot_sample = jnp.mean(boot_sample, axis=1)
        mean_dist = jnp.mean(boot_sample, axis=0)
        med_dist = jnp.median(boot_sample, axis=0)
        std_dist = jnp.std(boot_sample, axis=0)
        rounds = jnp.arange(n_rounds)

        plt.plot(rounds, mean_dist, 'o-', markersize=5, label='mean')
        plt.fill_between(rounds, mean_dist-std_dist, mean_dist+std_dist, alpha=0.2)
        plt.plot(rounds, med_dist, 'o-', markersize=5, label='median')

        plt.xlabel('round') 
        plt.ylabel('distance')
        plt.legend(prop={'size': 5})
        plt.savefig(os.path.join(f'{exp_root}', 'distance_round.png'))

        if show: 
            plt.show()


def plot_mmd(args, show=False):
    """
    Plots the mmd between the samples from the true model and samples from the learned model.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)
    num_samples = args.num_samples
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        sim = misc.get_simulator(exp_desc.sim)
        sim_desc = exp_desc.sim
        inf_desc = exp_desc.inf
        sim_setup = sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)
        mmd_trials = []


        for trial in range(args.start, args.end + 1):
            
            tin = time.time()
            print(f'Working on trial {trial}')
            print(exp_desc.pprint())

            try:

                exp_root = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir = exp_root + '/' + str(trial)

                try:

                    graphdef, state = util.io.load(os.path.join(exp_dir, 'model'))
                    (true_ps, true_cps), _ = util.io.load(exp_dir + '/gt')

                except FileNotFoundError:
                    print('MODEL FILE NOT FOUND')

                model = nnx.merge(graphdef, state)
                ssm = sim_setup['ssm']

                if isinstance(exp_desc.inf, ed.SNL_Descriptor):

                    key, subkey = jr.split(key)
                    est_sample = generate_emissions(key, sim_desc.emission_dim, model, true_cps, num_samples, -1, sim_desc.num_timesteps)

                elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):

                    key, subkey = jr.split(key)
                    est_sample = generate_emissions(key, sim_desc.emission_dim, model, true_cps, num_samples, inf_desc.lag, sim_desc.num_timesteps)

                key, subkey = jr.split(key)
                keys = jr.split(key, num_samples)
                fn = partial(sim_emissions, param=true_ps, ssm=ssm, num_timesteps=sim_desc.num_timesteps)
                true_sample = jnp.array(list(map(fn, keys)))
                mmd_error = mmd(est_sample.squeeze(), true_sample.squeeze())
                mmd_trials.append(mmd_error)

                nans_infs = jnp.isnan(est_sample) + jnp.isinf(est_sample)
                nans_infs = nans_infs.any(axis=-1).flatten()

                print('sample shape=', est_sample.shape)
                print(f'simulated obs have nans in {jnp.sum(nans_infs)} out of {nans_infs.shape[0]} samples')

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')

            util.io.save(mmd_error, os.path.join(exp_dir, 'mmd'))
            util.io.save(est_sample, os.path.join(exp_dir, 'est_sample'))

            tout = time.time() - tin
            print(f'Trial {trial} took {tout:.2f} seconds')

        mmd_trials = jnp.array(mmd_trials)
        
        B = 100
        key, subkey = jr.split(key)
        _, boot_sample = util.numerics.bootstrap(subkey, mmd_trials, B)
        mean_dist = jnp.mean(boot_sample, axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(mean_dist, 'o-', markersize=5, label='mean')
        ax.set_xlabel('trial') 
        ax.set_ylabel('MMD')
        ax.legend(prop={'size': 5})

        fig.savefig(os.path.join(f'{exp_root}', 'mmd.png'))

        if show: 
            plt.show()


def model_selection(args, get_mle=False, show=False):
    """

    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    sel_var = args.varX

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    groups = {'snl': {'mll': [],
                      'std_mll': [],
                      'mlle': [],
                      'std_mlle': []
                      },

             'tsnl': {'mll': [],
                      'std_mll': [],
                      'mlle': [],
                      'std_mlle': []
                      }
            }

    valsX = []

    for exp_desc in exp_descs:

        print(exp_desc.pprint())

        try:

            valX = exp_desc.inf.__dict__[sel_var]

        except KeyError:

            print(f'Variable {sel_var} not found in experiment descriptor')
        
        valsX.append(valX)
        mll = []
        mll_mle = []

        sim_desc = exp_desc.sim
        sim_mod = misc.get_simulator(sim_desc)

        for trial in range(args.start, args.end + 1):

            try:

                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir += '/' + str(trial)
                mll.append(util.io.load(os.path.join(exp_dir, 'mll')))
                _, observations = util.io.load(os.path.join(exp_dir, 'gt'))

                if get_mle:

                    if hasattr(sim_desc, 'dt_obs'):
                
                        sim_setup = sim_mod.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars, sim_desc.dt_obs)

                    else:

                        sim_setup = sim_mod.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)

                    props = sim_setup['props']

                    if isinstance(exp_desc.inf, ed.SNL_Descriptor) | isinstance(exp_desc.inf, ed.TSNL_Descriptor):

                        graphdef, state = util.io.load(os.path.join(exp_dir, 'model'))
                        model = nnx.merge(graphdef, state)
                        mle = find_mle(key, model, observations, exp_desc.inf.lag, props)
                        mll_mle.append(loglik_fn(mle, observations, model, exp_desc.inf.lag))

            except misc.AlreadyExistingExperiment:

                print('TRIAL DOES NOT EXIST')

            except FileNotFoundError:

                print('ERROR FILE NOT')

                continue

        cmplx = lambda x: jnp.log(100) * 160 * x
    
        B = 100
        key, subkey = jr.split(key)
        bmll, _ = util.numerics.bootstrap(subkey, jnp.array(mll), B)
        
        mll_mean = - 2 * jnp.mean(bmll) + cmplx(valX)
        mll_std = jnp.std(bmll)

        if get_mle: 

            key, subkey = jr.split(key)
            bmll_mle, _ = util.numerics.bootstrap(subkey, jnp.array(mll_mle), B)

            mll_mle_mean = - 2 * jnp.mean(bmll_mle) + cmplx(valX)
            mll_mle_std = jnp.std(bmll_mle)

        if isinstance(exp_desc.inf, ed.SNL_Descriptor):

            groups['snl']['mll'].append(mll_mean)
            groups['snl']['std_mll'].append(mll_std)

            if get_mle:

                groups['snl']['mlle'].append(mll_mle_mean)
                groups['snl']['std_mlle'].append(mll_mle_std)

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):

            groups['tsnl']['mll'].append(mll_mean)
            groups['tsnl']['std_mll'].append(mll_std)

            if get_mle:

                groups['tsnl']['mlle'].append(mll_mle_mean)
                groups['tsnl']['std_mlle'].append(mll_mle_std)

    valsX = jnp.array(valsX)

    for group in groups.keys():

        groups[group]['mll'] = jnp.array(groups[group]['mll'])
        groups[group]['std_mll'] = jnp.array(groups[group]['std_mll'])

        if get_mle:

            groups[group]['mlle'] = jnp.array(groups[group]['mlle'])
            groups[group]['std_mlle'] = jnp.array(groups[group]['std_mlle'])

    key, subkey = jr.split(key)
    from simulators import lgssm as ssm
    fig, ax = get_acf_plot(subkey, ssm, 50, 1, 1, 100, ['d4'])

    for group in groups.keys():

        try: 

            ax[1].plot(valsX, groups[group]['mll'], 'o-', markersize=4)
            ax[1].fill_between(valsX, groups[group]['mll']-groups[group]['std_mll'], groups[group]['mll']+groups[group]['std_mll'], alpha=0.2)

            if get_mle:

                ax[1].plot(valsX, groups[group]['mlle'], 'o-', markersize=4, label=f'mll_mle-{group}')
                ax[1].fill_between(valsX, groups[group]['mlle']-groups[group]['std_mlle'], groups[group]['mlle']+groups[group]['std_mlle'], alpha=0.2, label='Confidence Band')

        except IndexError:      
                
            print(f'No {group} results to plot')

        except KeyError:

            print(f'{group} has no variable {sel_var}')

        except TypeError:

            print(f'{group} has no results')

        except ValueError:

            print(f'{group} has no results')

    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('BIC')
    plt.legend(prop={'size': 5})

    fig.savefig(os.path.join(fig_dir, f'{sim_desc.name}_modsel.png'))

    if show: 

        fig.show()


def print_log(args):
    """
    Prints experiment logs.
    """

    from experiment_viewer import ExperimentViewer

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        try:
            ExperimentViewer(exp_desc).print_log()

        except misc.NonExistentExperiment:
            print('EXPERIMENT DOES NOT EXIST')


def main():

    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()