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
import numpy as onp

from experiment_runner import ExperimentRunner
import experiment_descriptor as ed
import experiment_results as er
import jax.random as jr
import misc

import util.numerics
from util.numerics import compute_all_errors, gmm_em_1d_stable, get_good_ids
from util.param import log_prior
from util.misc import print_table
from util.sample import sim_emissions, generate_emissions
from util.train import find_mle, loglik_fn
from inference.diagnostics.two_sample import sq_maximum_mean_discrepancy as mmd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
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
        
    def str2bool(v):

        if isinstance(v, bool):

            return v

        if v.lower() in ('yes', 'true', 't', '1'):

            return True

        elif v.lower() in ('no', 'false', 'f', '0'):

            return False

        else:

            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments.')
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run', help='run experiments')
    parser_run.add_argument('seed', type=int_or_str, help='seed for PRNG')
    parser_run.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_run.set_defaults(func=run_experiment)

    parser_trials = subparsers.add_parser('trials', help='run multiple experiment trials')
    parser_trials.add_argument('sample_gt', type=str2bool, help="A boolean flag (default: True)")
    parser_trials.add_argument('seed', type=int_or_str, help="seed for PRNG: integer, 'r' for random, or 's' to share each trial seed across experiments")
    parser_trials.add_argument('start', type=int, help='# of first trial')
    parser_trials.add_argument('end', type=int, help='# of last trial')
    parser_trials.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_trials.add_argument('--lp-cutoff', type=float, default=None, help='reject gt samples with -log p(gt) below this value (tail sampling)')
    parser_trials.set_defaults(func=run_trials)

    parser_view = subparsers.add_parser('view', help='view results')
    parser_view.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_view.add_argument('-b', '--block', action='store_true', help='block execution after viewing each experiment')
    parser_view.add_argument('-o', '--overwrite', type=str)
    parser_view.add_argument('-t', '--trial', type=int, default=0, help='trial to view (default is 0)')
    parser_view.set_defaults(func=view_results)

    parser_ensemble = subparsers.add_parser('ensemble', help='view ensemble results')
    parser_ensemble.add_argument('sample_gt', type=str2bool, help="A boolean flag (default: True)")
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
    parser_errors.add_argument('sample_gt', type=str2bool, help="A boolean flag (default: True)")
    parser_errors.add_argument('start', type=int, help='# of first trial')
    parser_errors.add_argument('end', type=int, help='# of last trial')
    parser_errors.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_errors.set_defaults(func=eval_and_plot_errors)

    parser_lp_histogram = subparsers.add_parser('lp_histogram', help='plot prior logprob of sampled ground truth against trial errors')
    parser_lp_histogram.add_argument('start', type=int, help='# of first trial')
    parser_lp_histogram.add_argument('end', type=int, help='# of last trial')
    parser_lp_histogram.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_lp_histogram.add_argument('--lp-bins', type=int, default=5, help='# of -log p(gt) bins for method-wise boxplots')
    parser_lp_histogram.set_defaults(func=lp_histogram)

    parser_gen_emissions = subparsers.add_parser('gen_emissions', help='generate emissions')
    parser_gen_emissions.add_argument('sample_gt', type=str2bool, help="A boolean flag (default: True)")
    parser_gen_emissions.add_argument('start', type=int, help='# of first trial')
    parser_gen_emissions.add_argument('end', type=int, help='# of last trial')
    parser_gen_emissions.add_argument('num_samples', type=int, help='# of samples to generate')
    parser_gen_emissions.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_gen_emissions.set_defaults(func=gen_emissions)

    parser_mmd = subparsers.add_parser('eval_and_plot_mmd', help='plot error by variable and inference method')
    parser_mmd.add_argument('sample_gt', type=str2bool, help="A boolean flag (default: True)")
    parser_mmd.add_argument('start', type=int, help='# of first trial')
    parser_mmd.add_argument('end', type=int, help='# of last trial')
    parser_mmd.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_mmd.set_defaults(func=eval_and_plot_mmd)

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

    parser_status = subparsers.add_parser('status', help='completion status of experiments in file')
    parser_status.add_argument('sample_gt', type=str2bool, help="A boolean flag (default: True)")
    parser_status.add_argument('start', type=int, help='# of first trial')
    parser_status.add_argument('end', type=int, help='# of last trial')
    parser_status.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_status.set_defaults(func=completion_status)

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

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    shared_seed = args.seed == 's' 

    if args.seed == 'r':

        seed = int(time.time() * 1000)

    elif shared_seed:

        seed = None

    else:

        seed = int(args.seed)

    if not shared_seed:

        key = jr.PRNGKey(seed)
    
    for exp_desc in exp_descs:

        for trial in range(args.start, args.end + 1):

            exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
            trial_dir = os.path.join(exp_dir, f'sample_gt_{args.sample_gt}', str(trial))

            if os.path.exists(trial_dir + '/exp_desc.txt') | os.path.exists(trial_dir + '/info.txt'):

                print('EXPERIMENT ALREADY EXISTS')
                jax.clear_caches()
                gc.collect()
                
                continue

            runner = ExperimentRunner(exp_desc)

            try:

                if os.path.exists(trial_dir):
                    
                    for root, _, files in os.walk(trial_dir, topdown=False):

                        for name in files:

                            os.remove(os.path.join(root, name))

                    os.rmdir(trial_dir)

                if shared_seed:

                    trial_seed = trial
                    subkey, _ = jr.split(jr.PRNGKey(trial_seed))

                else:

                    trial_seed = seed
                    key, subkey = jr.split(key)

                runner.run(trial=trial, sample_gt=args.sample_gt, plot_sims=True, key=subkey, seed=trial_seed, lp_cutoff=args.lp_cutoff)

            except misc.AlreadyExistingExperiment:

                print('RUNNER FAILED')

            jax.clear_caches()
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
        bias_array = []
        sdev_array = []
        mmd_array = []
        num_sims_array = []

        get_mmd = False

        if isinstance(exp_desc.inf, ed.SNL_Descriptor) or isinstance(exp_desc.inf, ed.TSNL_Descriptor):

            get_mmd = False

        for trial in range(args.start, args.end + 1):

            try:

                print('ON TRIAL:', trial)

                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir = os.path.join(exp_dir, f'sample_gt_{args.sample_gt}', str(trial))

                try:

                    print(exp_desc.pprint())

                    state_dim = exp_desc.sim.state_dim  
                    emission_dim = exp_desc.sim.emission_dim

                    num_sims = util.io.load(os.path.join(exp_dir, 'num_sims'))
                    kderr = util.io.load(os.path.join(exp_dir, 'kde_error'))
                    minerr = util.io.load(os.path.join(exp_dir, 'min_error'))
                    bias = util.io.load(os.path.join(exp_dir, 'bias'))
                    sdev = util.io.load(os.path.join(exp_dir, 'sdev')) 

                    kderr_array.append(kderr)
                    minerr_array.append(minerr)
                    bias_array.append(bias)
                    sdev_array.append(sdev)
                    num_sims_array.append(num_sims)

                    if get_mmd:

                        mmd = util.io.load(os.path.join(exp_dir, 'mmd'))
                        mmd_array.append(mmd)

                except FileNotFoundError:
                    print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')

        kderr_array = jnp.array(kderr_array)
        kderr_array = kderr_array[~(jnp.isnan(kderr_array) + jnp.isinf(kderr_array))]

        # good_ids = get_good_ids(kderr_array)
        good_ids = jnp.arange(kderr_array.shape[0])
        good_pct = good_ids.shape[0]            
        kderr_array = kderr_array[good_ids]

        minerr_array = jnp.array(minerr_array)
        minerr_array = minerr_array[good_ids]
        minerr_array = minerr_array[~(jnp.isnan(minerr_array) + jnp.isinf(minerr_array))]

        bias_array = jnp.array(bias_array)
        bias_array = bias_array[good_ids]
        bias_array = bias_array[~(jnp.isnan(bias_array) + jnp.isinf(bias_array))]

        sdev_array = jnp.array(sdev_array)
        sdev_array = sdev_array[good_ids]
        sdev_array = sdev_array[~(jnp.isnan(sdev_array) + jnp.isinf(sdev_array))]

        key, subkey = jr.split(key)
        kderr, _ = util.numerics.bootstrap(subkey, kderr_array, 100)
        mean_kderr = jnp.mean(kderr)
        std_kderr = jnp.std(kderr)

        key, subkey = jr.split(key)
        minerr, _ = util.numerics.bootstrap(subkey, minerr_array, 100)
        mean_minerr = jnp.mean(minerr)
        std_minerr = jnp.std(minerr)

        key, subkey = jr.split(key)
        bias, _ = util.numerics.bootstrap(subkey, bias_array, 100)
        mean_bias = jnp.mean(bias)
        std_bias = jnp.std(bias)

        key, subkey = jr.split(key)
        sdev, _ = util.numerics.bootstrap(subkey, sdev_array, 100)
        mean_sdev = jnp.mean(sdev)
        std_sdev = jnp.std(sdev)

        mmd_array = jnp.array(mmd_array)
        mmd_array = mmd_array[~(jnp.isnan(mmd_array) + jnp.isinf(mmd_array))]
        bootstrap_mmd, _ = util.numerics.bootstrap(subkey, mmd_array, 100)
        mean_mmd = jnp.mean(bootstrap_mmd)
        std_mmd = jnp.std(bootstrap_mmd)

        num_sims_array = jnp.array(num_sims_array)
        num_sims_array = num_sims_array[~(jnp.isnan(num_sims_array) + jnp.isinf(num_sims_array))]
        mean_num_sims = jnp.mean(num_sims_array)

        if isinstance(exp_desc.inf, ed.ABC_Descriptor):
            abc.results.append([mean_num_sims, mean_kderr, std_kderr, mean_minerr, std_minerr, mean_bias, std_bias, mean_sdev, std_sdev, good_pct])
        
        elif isinstance(exp_desc.inf, ed.BPF_MCMC_Descriptor):
            mcmc.results.append([mean_num_sims, mean_kderr, std_kderr, mean_minerr, std_minerr, mean_bias, std_bias, mean_sdev, std_sdev, good_pct])

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            snl.results.append([mean_num_sims, mean_kderr, std_kderr, mean_minerr, std_minerr, mean_bias, std_bias, mean_sdev, std_sdev, good_pct])

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):
            tsnl.results.append([mean_num_sims, mean_kderr, std_kderr, mean_minerr, std_minerr, mean_bias, std_bias, mean_sdev, std_sdev, good_pct])

    abc.make_jnp()
    mcmc.make_jnp()
    snl.make_jnp()
    tsnl.make_jnp()

    fig, ax = plt.subplots(4, 1, figsize=(5, 5))
    log = ''

    for alg in [abc, mcmc, snl, tsnl]:

        try:

            ax[0].plot(alg.results[:, 0], alg.results[:, 1], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[0].fill_between(alg.results[:, 0], alg.results[:,1]-alg.results[:, 2], alg.results[:, 1] + alg.results[:, 2], color=alg.color, alpha=0.05)
            ax[0].set_ylabel(r'$\mathcal{E}_{KDE}$', fontsize=12)
            ax[0].set_xticklabels([])  
            # ax[0].legend(prop={'size': 8}, fontsize=12)
            handles, labels = ax[0].get_legend_handles_labels()


            ax[1].plot(alg.results[:, 0], alg.results[:, 3], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[1].fill_between(alg.results[:, 0], alg.results[:, 3]-alg.results[:, 4], alg.results[:, 3] + alg.results[:, 4], color=alg.color, alpha=0.05)
            ax[1].set_ylabel(r'$\mathcal{E}_{min}$', fontsize=12)
            ax[1].set_xticklabels([]) 
            # ax[1].legend(prop={'size': 8}, fontsize=12)

            ax[2].plot(alg.results[:, 0], alg.results[:, 5], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[2].fill_between(alg.results[:, 0], alg.results[:, 5]-alg.results[:, 6], alg.results[:, 5] + alg.results[:, 6], color=alg.color, alpha=0.05)
            ax[2].set_ylabel('bias', fontsize=12)
            ax[2].set_xticklabels([])
            # ax[2].legend(prop={'size': 8}, fontsize=12)

            ax[3].plot(alg.results[:, 0], alg.results[:, 7], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[3].fill_between(alg.results[:, 0], alg.results[:, 7]-alg.results[:, 8], alg.results[:, 7] + alg.results[:, 8], color=alg.color, alpha=0.05)
            ax[3].set_xlabel('log Number of simulations', fontsize=12)
            ax[3].set_ylabel('st. dev.', fontsize=12)
            for label in ax[3].get_xticklabels():
                label.set_fontsize(10)
            # ax[3].legend(prop={'size': 8}, fontsize=12)

            log += f'{alg.name}: pcts = {alg.results[:, -1]} \n'

        except IndexError:

            print(f'No {alg.name} results to plot')

        except TypeError:
            
            print(f'{alg.name} has no results')


    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])         
    fig.savefig(os.path.join(fig_dir, f'{exp_desc.sim.name}_dim_{state_dim}_scan_num_sims_' + '_'.join(exp_desc.sim.target_vars) + f'_{args.sample_gt}' + '.pdf'), format="pdf", dpi=800)
    util.io.save_txt(log, os.path.join(fig_dir, f'{exp_desc.sim.name}_dim_{state_dim}_scan_num_sims_' + '_'.join(exp_desc.sim.target_vars) + f'_{args.sample_gt}' + '.txt'))


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
    from util.numerics import kde_error, min_error, bias, sdev

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        kderr_trials = []
        minerr_trials = []
        bias_trials = []
        ang_bias_trials = []
        sdev_trials = []
        inf_desc = exp_desc.inf
        sim_desc = exp_desc.sim

        for trial in range(args.start, args.end + 1):
            
            print(f'Working on trial {trial}')
            print(exp_desc.pprint())

            try:

                exp_root = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_root = os.path.join(exp_root, f'sample_gt_{args.sample_gt}')
                exp_dir = os.path.join(exp_root, str(trial))

                try:

                    (_, true_cps), observations = util.io.load(os.path.join(exp_dir, 'gt'))

                    if isinstance(inf_desc, ed.ABC_Descriptor):

                        cps, _, counts = util.io.load(os.path.join(exp_dir, 'results'))
                        kderr, minerr, bias, ang_bias, sdev, (mds, amd), (ads, apd), (maxds, amxd) = compute_all_errors(cps, true_cps)
                        num_simulations = counts * sim_desc.num_timesteps

                    elif isinstance(inf_desc, ed.BPF_MCMC_Descriptor):

                        cps, _ = util.io.load(os.path.join(exp_dir, 'results'))
                        kderr, minerr, bias, ang_bias, sdev, (mds, amd), (ads, apd), (maxds, amxd) = compute_all_errors(cps, true_cps)
                        num_simulations = inf_desc.num_prt * sim_desc.num_timesteps * inf_desc.mcmc_steps * inf_desc.num_iters

                    elif isinstance(inf_desc, ed.SNL_Descriptor):

                        _, cps = util.io.load(os.path.join(exp_dir, 'posterior'))
                        kderr, minerr, bias, ang_bias, sdev, (mds, amd), (ads, apd), (maxds, amxd) = compute_all_errors(cps, true_cps)
                        num_simulations = inf_desc.n_rounds * inf_desc.n_samples * sim_desc.num_timesteps
                            
                    elif isinstance(inf_desc, ed.TSNL_Descriptor):

                        _, cps = util.io.load(os.path.join(exp_dir, 'posterior'))
                        kderr, minerr, bias, ang_bias, sdev, (mds, amd), (ads, apd), (maxds, amxd) = compute_all_errors(cps, true_cps)
                        num_simulations = inf_desc.n_rounds * inf_desc.n_samples * sim_desc.num_timesteps

                    util.io.save(kderr, os.path.join(exp_dir, 'kde_error'))
                    util.io.save(minerr, os.path.join(exp_dir, 'min_error'))
                    util.io.save(bias, os.path.join(exp_dir, 'bias'))
                    util.io.save(ang_bias, os.path.join(exp_dir, 'ang_bias'))
                    util.io.save(sdev, os.path.join(exp_dir, 'sdev'))
                    util.io.save({'min_dist':(mds, amd), 
                                  'pair_dist':(ads, apd), 
                                  'max_dist': (maxds, amxd)}, os.path.join(exp_dir, 'sample_dists'))
                    util.io.save(jnp.log(num_simulations), os.path.join(exp_dir, 'num_sims'))

                    kderr_trials.append(kderr)
                    minerr_trials.append(minerr)
                    bias_trials.append(bias)
                    ang_bias_trials.append(ang_bias)
                    sdev_trials.append(sdev)

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
            bias_trials = jnp.array(bias_trials)
            ang_bias_trials = jnp.array(ang_bias_trials)
            sdev_trials = jnp.array(sdev_trials)

            #remove NaNs 
            nans_infs = jnp.isnan(kderr_trials) + jnp.isinf(kderr_trials)
            kderr_trials = kderr_trials[~nans_infs]
            nans_infs = jnp.isnan(minerr_trials) + jnp.isinf(minerr_trials)
            minerr_trials = minerr_trials[~nans_infs]
            nans_infs = jnp.isnan(bias_trials) + jnp.isinf(bias_trials)
            bias_trials = bias_trials[~nans_infs]
            nans_infs = jnp.isnan(ang_bias_trials) + jnp.isinf(ang_bias_trials)
            ang_bias_trials = ang_bias_trials[~nans_infs]
            nans_infs = jnp.isnan(sdev_trials) + jnp.isinf(sdev_trials)
            sdev_trials = sdev_trials[~nans_infs]

            key, subkey = jr.split(key)
            kderr, _ = util.numerics.bootstrap(subkey, kderr_trials, 100)
            kderr_avg = jnp.mean(kderr)
            kderr_std = jnp.std(kderr)

            key, subkey = jr.split(key)
            minerr, _ = util.numerics.bootstrap(subkey, minerr_trials, 100)
            minerr_avg = jnp.mean(minerr)
            minerr_std = jnp.std(minerr)

            key, subkey = jr.split(key)
            bias, _ = util.numerics.bootstrap(subkey, bias_trials, 100)
            bias_avg = jnp.mean(bias)
            bias_std = jnp.std(bias)

            key, subkey = jr.split(key)
            ang_bias, _ = util.numerics.bootstrap(subkey, ang_bias_trials, 100)
            ang_bias_avg = jnp.mean(ang_bias)
            ang_bias_std = jnp.std(ang_bias)

            key, subkey = jr.split(key)
            sdev, _ = util.numerics.bootstrap(subkey, sdev_trials, 100)
            sdev_avg = jnp.mean(sdev)
            sdev_std = jnp.std(sdev)

            fig, ax = plt.subplots(2, 3, figsize=(10, 10))

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

            ax[1, 0].hist(bias_trials, bins=50, alpha=0.5, label=f'{label}', density=True)
            ax[1, 0].axvline(bias_avg, color='green', linestyle='dashed', linewidth=0.5)
            ax[1, 0].axvline(bias_avg - bias_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 0].axvline(bias_avg + bias_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 0].set_xlabel('bias')
            ax[1, 0].set_ylabel('count')
            ax[1, 0].set_title(f'{label} - Bias histogram %')
            ax[1, 0].legend(prop={'size': 5})

            ax[1, 1].hist(ang_bias_trials, bins=50, alpha=0.5, label=f'{label}', density=True)
            ax[1, 1].axvline(ang_bias_avg, color='green', linestyle='dashed', linewidth=0.5)
            ax[1, 1].axvline(ang_bias_avg - ang_bias_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 1].axvline(ang_bias_avg + ang_bias_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 1].set_xlabel('ang_bias')
            ax[1, 1].set_ylabel('count')
            ax[1, 1].set_title(f'{label} - Ang Bias histogram %')
            ax[1, 1].legend(prop={'size': 5})

            ax[1, 2].hist(sdev_trials, bins=50, alpha=0.5, label=f'{label}', density=True)
            ax[1, 2].axvline(sdev_avg, color='green', linestyle='dashed', linewidth=0.5)
            ax[1, 2].axvline(sdev_avg - sdev_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 2].axvline(sdev_avg + sdev_std, color='gray', linestyle='dotted', linewidth=0.25)
            ax[1, 2].set_xlabel('sdev')
            ax[1, 2].set_ylabel('count')
            ax[1, 2].set_title(f'{label} - SDev histogram %')
            ax[1, 2].legend(prop={'size': 5})

            fig.savefig(os.path.join(f'{exp_root}', f'Error histograms.png'))

            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(kderr_trials, label='kderr')
            ax.plot(minerr_trials, label='minerr')
            ax.plot(bias_trials, label='bias')
            ax.plot(sdev_trials, label='sdev')

            ax.set_xlabel('Trial')
            ax.set_ylabel('Error')
            ax.set_title(f'{label} - Errors vs Trial')
            ax.legend(prop={'size': 5})

            fig.savefig(os.path.join(f'{exp_root}', f'Errors vs Trials.png'))

        except FileNotFoundError:

            print('Experiment Empty')
    
    if show: 

       plt.show()


def lp_histogram(args, show=False):
    """
    Plots prior log probability of sampled ground-truth parameters against trial
    errors. This command is only meaningful for sample_gt=True experiments.
    """

    error_names = ['kde_error', 'min_error', 'bias', 'sdev']
    error_labels = ['E_KDE', 'E_min', 'Bias', 'SDev']

    def get_inference_label(inf_desc):

        if isinstance(inf_desc, ed.ABC_Descriptor):

            return 'smc-abc'

        elif isinstance(inf_desc, ed.BPF_MCMC_Descriptor):

            return 'mcmc'

        elif isinstance(inf_desc, ed.SNL_Descriptor):

            return 'snl'

        elif isinstance(inf_desc, ed.TSNL_Descriptor):

            return 't-snl'

        else:

            return 'unknown'

    def collect_lp_errors(exp_desc):

        print(exp_desc.pprint())

        inf_desc = exp_desc.inf
        sim_desc = exp_desc.sim
        sim_mod = misc.get_simulator(sim_desc)

        if hasattr(sim_desc, 'dt_obs'):

            sim_setup = sim_mod.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars, sim_desc.dt_obs)

        else:

            sim_setup = sim_mod.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)

        props = sim_setup['props']
        exp_root = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir(), 'sample_gt_True')
        trial_ids = []
        log_probs = []
        errors = {name: [] for name in error_names}

        for trial in range(args.start, args.end + 1):

            print(f'Working on trial {trial}')
            exp_dir = os.path.join(exp_root, str(trial))

            try:

                (_, true_cps), _ = util.io.load(os.path.join(exp_dir, 'gt'))

                if isinstance(inf_desc, ed.ABC_Descriptor):

                    cps, _, _ = util.io.load(os.path.join(exp_dir, 'results'))

                elif isinstance(inf_desc, ed.BPF_MCMC_Descriptor):

                    cps, _ = util.io.load(os.path.join(exp_dir, 'results'))

                elif isinstance(inf_desc, ed.SNL_Descriptor) or isinstance(inf_desc, ed.TSNL_Descriptor):

                    _, cps = util.io.load(os.path.join(exp_dir, 'posterior'))

                else:

                    print('UNKNOWN INFERENCE DESCRIPTOR')
                    continue

                kderr, minerr, bias, _, sdev, _, _, _ = compute_all_errors(cps, true_cps)
                lp = jnp.sum(log_prior(true_cps, props))

                trial_ids.append(trial)
                log_probs.append(float(lp))
                errors['kde_error'].append(float(kderr))
                errors['min_error'].append(float(minerr))
                errors['bias'].append(float(bias))
                errors['sdev'].append(float(sdev))

            except FileNotFoundError:

                print('ERROR FILE NOT FOUND')

        if not log_probs:

            print('Experiment Empty')
            return None

        log_probs = jnp.array(log_probs)
        neg_log_probs = -log_probs
        trial_ids = jnp.array(trial_ids)
        errors = {name: jnp.array(vals) for name, vals in errors.items()}

        return {
            'exp_root': exp_root,
            'label': get_inference_label(inf_desc),
            'trials': trial_ids,
            'log_probs': log_probs,
            'neg_log_probs': neg_log_probs,
            'errors': errors
        }

    def plot_single_result(result):

        util.io.save(
            {
                'trials': result['trials'],
                'log_probs': result['log_probs'],
                'neg_log_probs': result['neg_log_probs'],
                'errors': result['errors']
            },
            os.path.join(result['exp_root'], 'lp_histogram_data')
        )

        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        axes = ax.ravel()

        for i, (name, label) in enumerate(zip(error_names, error_labels)):

            ys = result['errors'][name]
            xs = result['neg_log_probs']
            finite_ids = ~(jnp.isnan(xs) | jnp.isinf(xs) | jnp.isnan(ys) | jnp.isinf(ys))
            axes[i].scatter(xs[finite_ids], ys[finite_ids], s=12, alpha=0.75)
            axes[i].set_xlabel('-log p(gt)')
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label} vs -log p(gt)')

            for x, y, trial in zip(xs[finite_ids], ys[finite_ids], result['trials'][finite_ids]):

                axes[i].annotate(str(int(trial)), (x, y), fontsize=5, alpha=0.6)

        fig.tight_layout()
        fig.savefig(os.path.join(result['exp_root'], 'lp_histogram.png'), dpi=800)
        plt.close(fig)

    def get_finite_xy(result, name):

        xs = onp.asarray(result['neg_log_probs'])
        ys = onp.asarray(result['errors'][name])
        finite_ids = onp.isfinite(xs) & onp.isfinite(ys)

        return xs[finite_ids], ys[finite_ids]

    def get_method_labels(results):

        return list(dict.fromkeys(result['label'] for result in results))

    def plot_combined_results(results, output_dir):

        util.io.save(results, os.path.join(output_dir, 'lp_histogram_data'))

        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        axes = ax.ravel()
        method_labels = get_method_labels(results)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, (name, label) in enumerate(zip(error_names, error_labels)):

            handles = []

            for method_idx, method_label in enumerate(method_labels):

                method_xs = []
                method_ys = []

                for result in results:

                    if result['label'] != method_label:

                        continue

                    xs, ys = get_finite_xy(result, name)
                    method_xs.extend(xs.tolist())
                    method_ys.extend(ys.tolist())

                if not method_xs:

                    continue

                method_xs = onp.asarray(method_xs)
                method_ys = onp.asarray(method_ys)
                mean_x = onp.mean(method_xs)
                mean_y = onp.mean(method_ys)
                width = 2.0 * onp.std(method_xs)
                height = 2.0 * onp.std(method_ys)

                if width == 0.0:

                    width = max(1e-6, abs(mean_x) * 0.02)

                if height == 0.0:

                    height = max(1e-6, abs(mean_y) * 0.02)

                color = colors[method_idx % len(colors)]
                rect = Rectangle(
                    (mean_x - width / 2.0, mean_y - height / 2.0),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.25,
                    linewidth=1.0,
                    label=method_label
                )

                axes[i].add_patch(rect)
                axes[i].scatter(mean_x, mean_y, s=18, color=color, edgecolors='black', linewidths=0.25, zorder=3)
                handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.25, label=method_label))

            axes[i].autoscale_view()
            axes[i].set_xlabel('-log p(gt)')
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label} 2D method boxes')

            if handles:

                axes[i].legend(handles=handles, prop={'size': 5})

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'lp_histogram.png'), dpi=800)
        plt.close(fig)

    def plot_scatter_combined_results(results, output_dir):

        method_style = {
            'smc-abc': {'color': 'blue',   'marker': 'o'},
            'mcmc':    {'color': 'red',    'marker': 'x'},
            'snl':     {'color': 'tomato', 'marker': 's'},
            't-snl':   {'color': 'green',  'marker': 'D'},
        }
        fallback_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        method_labels = get_method_labels(results)

        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        axes = ax.ravel()

        for i, (name, label) in enumerate(zip(error_names, error_labels)):

            handles = []

            for method_idx, method_label in enumerate(method_labels):

                style = method_style.get(method_label, {
                    'color': fallback_colors[method_idx % len(fallback_colors)],
                    'marker': 'o'
                })
                all_xs, all_ys = [], []

                for result in results:

                    if result['label'] != method_label:
                        continue

                    xs, ys = get_finite_xy(result, name)
                    pos = ys > 0
                    all_xs.extend(xs[pos].tolist())
                    all_ys.extend(ys[pos].tolist())

                if not all_xs:
                    continue

                axes[i].scatter(all_xs, all_ys, s=12, alpha=0.6,
                                color=style['color'], marker=style['marker'])
                handles.append(Line2D([0], [0], marker=style['marker'], color='w',
                                      markerfacecolor=style['color'],
                                      markeredgecolor=style['color'], markersize=5,
                                      label=method_label))

            axes[i].set_yscale('log')
            axes[i].set_xlabel('-log p(gt)')
            axes[i].set_ylabel(f'log({label})')
            axes[i].set_title(f'log({label}) vs -log p(gt)')

            if handles:
                axes[i].legend(handles=handles, prop={'size': 5})

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'lp_histogram_scatter.png'), dpi=800)
        plt.close(fig)

    def plot_binned_combined_results(results, output_dir):

        all_xs = []

        for result in results:

            for name in error_names:

                xs, _ = get_finite_xy(result, name)
                all_xs.extend(xs.tolist())

        if not all_xs:

            print('No finite log probabilities found for binned plot')
            return

        num_bins = max(1, int(args.lp_bins))
        min_x = min(all_xs)
        max_x = max(all_xs)

        if min_x == max_x:

            min_x -= 0.5
            max_x += 0.5

        bin_edges = onp.linspace(min_x, max_x, num_bins + 1)
        method_labels = get_method_labels(results)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        group_width = 0.8
        box_width = group_width / max(1, len(method_labels)) * 0.8
        offsets = onp.linspace(
            -group_width / 2 + box_width / 2,
            group_width / 2 - box_width / 2,
            len(method_labels)
        )

        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        axes = ax.ravel()
        bin_centers = onp.arange(num_bins)
        bin_labels = [f'{bin_edges[i]:.2g}-{bin_edges[i + 1]:.2g}' for i in range(num_bins)]

        for i, (name, label) in enumerate(zip(error_names, error_labels)):

            for method_idx, method_label in enumerate(method_labels):

                color = colors[method_idx % len(colors)]

                for bin_idx in range(num_bins):

                    vals = []

                    for result in results:

                        if result['label'] != method_label:

                            continue

                        xs, ys = get_finite_xy(result, name)

                        if bin_idx == num_bins - 1:

                            in_bin = (xs >= bin_edges[bin_idx]) & (xs <= bin_edges[bin_idx + 1])

                        else:

                            in_bin = (xs >= bin_edges[bin_idx]) & (xs < bin_edges[bin_idx + 1])

                        vals.extend(ys[in_bin].tolist())

                    if not vals:

                        continue

                    box = axes[i].boxplot(
                        [vals],
                        positions=[bin_centers[bin_idx] + offsets[method_idx]],
                        widths=box_width,
                        patch_artist=True,
                        showfliers=True
                    )

                    for patch in box['boxes']:

                        patch.set_facecolor(color)
                        patch.set_alpha(0.45)

                    for median in box['medians']:

                        median.set_color('black')

            axes[i].set_xlabel('-log p(gt) bin')
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label} by -log p(gt) bin and method')
            axes[i].set_xticks(bin_centers)
            axes[i].set_xticklabels(bin_labels, rotation=25, ha='right')
            axes[i].legend(
                handles=[
                    Patch(facecolor=colors[j % len(colors)], alpha=0.45, label=method_label)
                    for j, method_label in enumerate(method_labels)
                ],
                prop={'size': 5}
            )

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'lp_histogram_binned.png'), dpi=800)
        plt.close(fig)

    for file in args.files:

        exp_descs = ed.parse(util.io.load_txt(file))
        inf_methods = {get_inference_label(exp_desc.inf) for exp_desc in exp_descs}
        results = []

        for exp_desc in exp_descs:

            result = collect_lp_errors(exp_desc)

            if result is not None:

                results.append(result)

        if len(inf_methods) == 1:

            for result in results:

                plot_single_result(result)

        elif results:

            output_dir = os.path.commonpath([result['exp_root'] for result in results])
            plot_scatter_combined_results(results, output_dir)
            # plot_combined_results(results, output_dir)
            # plot_binned_combined_results(results, output_dir)

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


def gen_emissions(args, show=False):
    """
    Plots the mmd between the samples from the true model and samples from the learned model.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)
    num_samples = args.num_samples
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        sim_desc = exp_desc.sim
        inf_desc = exp_desc.inf

        for trial in range(args.start, args.end + 1):
            
            tin = time.time()
            print(f'Working on trial {trial}')
            print(exp_desc.pprint())

            try:

                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir = os.path.join(exp_dir, f'sample_gt_{args.sample_gt}', str(trial))

                try:

                    graphdef, state = util.io.load(os.path.join(exp_dir, 'model'))
                    (_, true_cps), _ = util.io.load(exp_dir + '/gt')

                except FileNotFoundError:

                    print('MODEL FILE NOT FOUND')

                model = nnx.merge(graphdef, state)

                if isinstance(exp_desc.inf, ed.SNL_Descriptor):

                    key, subkey = jr.split(key)
                    gen_sample = generate_emissions(subkey, sim_desc.emission_dim, model, true_cps, num_samples, -1, sim_desc.num_timesteps)

                elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):

                    key, subkey = jr.split(key)
                    gen_sample = generate_emissions(subkey, sim_desc.emission_dim, model, true_cps, num_samples, inf_desc.lag, sim_desc.num_timesteps)

                util.io.save(gen_sample, os.path.join(exp_dir, 'gen_sample'))

            except misc.AlreadyExistingExperiment:

                print('TRIAL DOES NOT EXIST')


def eval_and_plot_mmd(args, show=False):

    """
    Takes a set of experiments and line-plots the errors vs num_sims of each algorithm together.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    snl = er.SNL_Results()
    tsnl = er.TSNL_Results()

    for exp_desc in exp_descs:

        kde_array = []
        mmd_trials = []
        num_sims_array = []

        sim = misc.get_simulator(exp_desc.sim)
        sim_desc = exp_desc.sim
        inf_desc = exp_desc.inf
        sim_setup = sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)
        ssm = sim_setup['ssm']

        for trial in range(args.start, args.end + 1):

            try:

                print('ON TRIAL:', trial)

                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir = os.path.join(exp_dir, f'sample_gt_{args.sample_gt}', str(trial))

                try:

                    print(exp_desc.pprint())

                    state_dim = exp_desc.sim.state_dim  
                    emission_dim = exp_desc.sim.emission_dim

                    num_sims = util.io.load(os.path.join(exp_dir, 'num_sims'))
                    gen_sample = util.io.load(os.path.join(exp_dir, 'gen_sample'))
                    (true_ps, true_cps), _ = util.io.load(exp_dir + '/gt')
                    
                    kde = util.io.load(os.path.join(exp_dir, 'kde_error'))

                    num_sims_array.append(num_sims)
                    kde_array.append(kde)

                except FileNotFoundError:

                    print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:

                print('TRIAL DOES NOT EXIST')

            num_samples = gen_sample.shape[0]
            key, subkey = jr.split(key)
            keys = jr.split(key, num_samples)
            fn = partial(sim_emissions, param=true_ps, ssm=ssm, num_timesteps=sim_desc.num_timesteps)
            true_sample = jnp.array(list(map(fn, keys)))
            mmd_error = jnp.array([mmd(gen_sample[:, t], true_sample[:, t]) for t in range(sim_desc.num_timesteps)]).mean()
            mmd_trials.append(mmd_error)

            nans_infs = jnp.isnan(gen_sample) + jnp.isinf(gen_sample)
            nans_infs = nans_infs.any(axis=-1).flatten()

            print('sample shape=', gen_sample.shape)
            print(f'simulated obs have nans in {jnp.sum(nans_infs)} out of {nans_infs.shape[0]} samples')

            util.io.save(mmd_error, os.path.join(exp_dir, 'mmd'))

        kde_array = jnp.array(kde_array)
        good_ids = get_good_ids(kde_array)
        good_pct = good_ids.shape[0]            
        kde_array = kde_array[good_ids]
    
        mmd_trials = jnp.array(mmd_trials)
        mmd_trials = mmd_trials[good_ids]

        mmd_trials = jnp.array(mmd_trials)
        mmd_trials = mmd_trials[~(jnp.isnan(mmd_trials) + jnp.isinf(mmd_trials))]
        key, subkey = jr.split(key)
        bootstrap_mmd, _ = util.numerics.bootstrap(subkey, mmd_trials, 100)
        mean_mmd = jnp.mean(bootstrap_mmd)
        std_mmd = jnp.std(bootstrap_mmd)

        num_sims_array = jnp.array(num_sims_array)
        num_sims_array = num_sims_array[~(jnp.isnan(num_sims_array) + jnp.isinf(num_sims_array))]
        mean_num_sims = jnp.mean(num_sims_array)

        if isinstance(exp_desc.inf, ed.SNL_Descriptor):
            snl.results.append([mean_num_sims, mean_mmd, std_mmd, good_pct])

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):
            tsnl.results.append([mean_num_sims, mean_mmd, std_mmd, good_pct])

    snl.make_jnp()
    tsnl.make_jnp()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    log = ''

    for alg in [snl, tsnl]:

        try:

            ax.plot(alg.results[:, 0], alg.results[:, 1], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax.fill_between(alg.results[:, 0], alg.results[:,1]-alg.results[:, 2], alg.results[:, 1] + alg.results[:, 2], color=alg.color, alpha=0.05)
            ax.set_ylabel(r'$\mathcal{E}_{KDE}$')
            ax.legend(prop={'size': 5})

            log += f'{alg.name}: pcts = {alg.results[:, -1]} \n'

        except IndexError:

            print(f'No {alg.name} results to plot')

        except TypeError as e:
            
            print(f'{alg.name} has no results')
        
    fig.savefig(os.path.join(fig_dir, f'{exp_desc.sim.name}_dim_{state_dim}_scan_num_sims_' + '_'.join(exp_desc.sim.target_vars) + f'_{args.sample_gt}_mmd' + '.pdf'), format="pdf", dpi=800)
    util.io.save_txt(log, os.path.join(fig_dir, f'{exp_desc.sim.name}_dim_{state_dim}_scan_num_sims_' + '_'.join(exp_desc.sim.target_vars) + f'_{args.sample_gt}' + '.txt'))


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


def completion_status(args):

    """
    Takes a set of experiments and line-plots the errors vs num_sims of each algorithm together.
    """

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    abc = er.ABC_Results()
    mcmc = er.MCMC_Results()
    snl = er.SNL_Results()
    tsnl = er.TSNL_Results()
    
    for exp_desc in exp_descs:

        count_trials = 0

        for trial in range(args.start, args.end + 1):

            exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
            exp_dir = os.path.join(exp_dir, f'sample_gt_{args.sample_gt}', str(trial))


            if os.path.exists(exp_dir):

                has_results = os.path.exists(os.path.join(exp_dir, 'results.pkl'))
                has_posterior = os.path.exists(os.path.join(exp_dir, 'posterior.pkl'))

                if has_results or has_posterior:

                    count_trials += 1

        if isinstance(exp_desc.inf, ed.ABC_Descriptor):
            abc.results.append(f'{count_trials} / {args.end}')
        
        elif isinstance(exp_desc.inf, ed.BPF_MCMC_Descriptor):
            mcmc.results.append(f'{count_trials} / {args.end}')

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            snl.results.append(f'{count_trials} / {args.end}')

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):
            tsnl.results.append(f'{count_trials} / {args.end}')


    print('ABC', abc.results, '\n',
          'MCMC', mcmc.results, '\n',
          'SNL', snl.results, '\n',
          'T_SNL', tsnl.results)


def main():

    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
