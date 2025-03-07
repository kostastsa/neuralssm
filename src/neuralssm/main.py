# This module is taken/adapted from the repository ([https://github.com/gpapamak/snl.git])
# Originally authored by George Papamakarios, under the MIT License
import argparse
import os
import jax
import jax.numpy as jnp

from flax import nnx

import gc
import util.io
from functools import partial

import experiment_descriptor as ed
import experiment_results as er
import jax.random as jr
import time
import misc
import util.misc
from util.sample import sim_emissions, generate_emissions
from util.param import sample_prior, to_train_array
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
    parser_mmd.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_mmd.set_defaults(func=plot_mmd)

    parser_plot_by_var = subparsers.add_parser('plotvar', help='plot error by variable and inference method')
    parser_plot_by_var.add_argument('varname', type=str, help='variable to plot by')
    parser_plot_by_var.add_argument('start', type=int, help='# of first trial')
    parser_plot_by_var.add_argument('end', type=int, help='# of last trial')
    parser_plot_by_var.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_plot_by_var.set_defaults(func=plot_by_var)

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
        
            ExperimentRunner(exp_desc).run(trial=0, sample_gt=True, key=key)

        except misc.AlreadyExistingExperiment:
            print('EXPERIMENT ALREADY EXISTS')

    print('ALL DONE')


def run_trials(args):
    """
    Runs experiments for multiple trials with random ground truth.
    """

    from experiment_runner import ExperimentRunner

    if args.start < 1:
        raise ValueError('trial # must be a positive integer')

    if args.end < args.start:
        raise ValueError('end trial can''t be less than start trial')
    
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    seed = int(time.time() * 1000) if args.seed=='r' else args.seed  
    key = jr.PRNGKey(seed)
    
    for exp_desc in exp_descs:

        runner = ExperimentRunner(exp_desc)

        for trial in range(args.start, args.end + 1):

            key, subkey = jr.split(key)

            try:
                
                out = runner.run(trial=trial, sample_gt=True, key=subkey, seed=seed)

            except misc.AlreadyExistingExperiment:

                print('EXPERIMENT ALREADY EXISTS')

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


def view_ensemble(args, show=True):
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
        
        error_array = []
        rmse_array = []

        for trial in range(args.start, args.end + 1):

            try:

                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir += '/' + str(trial)

                try:
                    print(exp_desc.pprint())
                    error, num_sims = util.io.load(os.path.join(exp_dir, 'error'))
                    rmse = util.io.load(os.path.join(exp_dir, 'rmse'))
                    error_array.append(error)
                    rmse_array.append(rmse)

                except FileNotFoundError:
                    print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')

        B = 100
        key, subkey = jr.split(key)
        error_array = jnp.array(error_array)
        nans_infs = jnp.isnan(error_array) + jnp.isinf(error_array)
        nfail = jnp.sum(nans_infs)
        error_array = error_array[~nans_infs]

        bootstrap_errors, _ = util.misc.bootstrap(subkey, error_array, B)
        mean = jnp.mean(bootstrap_errors)
        std = jnp.std(bootstrap_errors)
        num_sims = jnp.log(num_sims)

        bootstrap_rmse, _ = util.misc.bootstrap(subkey, jnp.array(rmse_array), B)
        mean_rmse = jnp.mean(bootstrap_rmse)
        std_rmse = jnp.std(bootstrap_rmse)
            
        if isinstance(exp_desc.inf, ed.ABC_Descriptor):
            abc.results.append([mean, std, mean_rmse, std_rmse, num_sims, nfail])
        
        elif isinstance(exp_desc.inf, ed.BPF_MCMC_Descriptor):
            mcmc.results.append([mean, std, mean_rmse, std_rmse, num_sims, nfail])

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            snl.results.append([mean, std, mean_rmse, std_rmse, num_sims, nfail])

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):
            tsnl.results.append([mean, std, mean_rmse, std_rmse, num_sims, nfail])

    abc.make_jnp()
    mcmc.make_jnp()
    snl.make_jnp()
    tsnl.make_jnp()

    fig, ax = plt.subplots(2, 1, figsize=(5, 5))

    for alg in [abc, mcmc, snl, tsnl]:

        try:

            nfail_avg = jnp.mean(alg.results[:, -1])
            ax[0].plot(alg.results[:, 4], alg.results[:, 0], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=5, label=f'{alg.name}, failed={nfail_avg:.0f}/{args.end}')
            ax[0].fill_between(alg.results[:, 4], alg.results[:,0]-alg.results[:, 1], alg.results[:, 0] + alg.results[:, 1], color=alg.color, alpha=0.05, hatch='//')
            ax[0].set_title('Error vs num_sims')
            ax[0].set_xlabel('log Number of simulations')
            ax[0].set_ylabel('Error')
            ax[0].legend()

            ax[1].plot(alg.results[:, 4], alg.results[:, 2], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=5, label=f'{alg.name}')
            ax[1].fill_between(alg.results[:, 4], alg.results[:, 2]-alg.results[:, 3], alg.results[:, 2] + alg.results[:, 3], color=alg.color, alpha=0.05, hatch='x')
            ax[1].set_title('RMSE vs num_sims')
            ax[1].set_xlabel('log Number of simulations')
            ax[1].set_ylabel('RMSE')
            ax[1].legend()

        except IndexError:

            print(f'No {alg.name} results to plot')

        except TypeError:
            
            print(f'{alg.name} has no results')

    fig.savefig(os.path.join(fig_dir, f'Error-num_sims ensemble.pdf'), format="pdf", dpi=800)

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
    groups = {'abc': {}, 'bpf_mcmc': {}, 'snl': {}, 'tsnl': {}}
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
        bootstrap_error, _ = util.misc.bootstrap(subkey, jnp.array(error_trials), B)
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
            val['varX'] = jnp.array(val['varX'])
            val['varY'] = jnp.array(val['varY'])
            val['error_std'] = jnp.array(val['error_std'])

    try:

        for group_var_val in set(group_by_vals):

            plt.plot(groups['abc'][group_var_val]['varX'], groups['abc'][group_var_val]['varY'], 'o-', markersize=5, label=f'SMC-ABC, {group_by}={group_var_val}')
            plt.fill_between(groups['abc'][group_var_val]['varX'], groups['abc'][group_var_val]['varY']-groups['abc'][group_var_val]['error_std'], groups['abc'][group_var_val]['varY']+groups['abc'][group_var_val]['error_std'], color='blue', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No ABC results to plot')

    except KeyError:

        print('ABC has no variable {}'.format(group_by))

    try:

        for group_var_val in set(group_by_vals):

            plt.plot(groups['bpf_mcmc'][group_var_val]['varX'], groups['bpf_mcmc'][group_var_val]['varY'], 's-', markersize=5, label=f'BPF-MCMC, {group_by}={group_var_val}')
            plt.fill_between(groups['bpf_mcmc'][group_var_val]['varX'], groups['bpf_mcmc'][group_var_val]['varY']-groups['bpf_mcmc'][group_var_val]['error_std'], groups['bpf_mcmc'][group_var_val]['varY']+groups['bpf_mcmc'][group_var_val]['error_std'], color='orange', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No BPF-MCMC results to plot')

    except KeyError:

        print('BPF-MCMC has no variable {}'.format(group_by))
    
    try:

        for group_var_val in set(group_by_vals):

            plt.plot(groups['snl'][group_var_val]['varX'], groups['snl'][group_var_val]['varY'], '^-', markersize=5, label=f'SNL, {group_by}={group_var_val}')
            plt.fill_between(groups['snl'][group_var_val]['varX'], groups['snl'][group_var_val]['varY']-groups['snl'][group_var_val]['error_std'], groups['snl'][group_var_val]['varY']+groups['snl'][group_var_val]['error_std'], color='green', alpha=0.2)

    except IndexError:

        print('No SNL results to plot')

    except KeyError:

        print('SNL has no variable {}'.format(group_by))

    try:

        for group_var_val in set(group_by_vals):

            plt.plot(groups['tsnl'][group_var_val]['varX'], groups['tsnl'][group_var_val]['varY'], 'd-', markersize=5, label=f'T-SNL, {group_var_val}')
            plt.fill_between(groups['tsnl'][group_var_val]['varX'], groups['tsnl'][group_var_val]['varY']-groups['tsnl'][group_var_val]['error_std'], groups['tsnl'][group_var_val]['varY']+groups['tsnl'][group_var_val]['error_std'], alpha=0.2, label=f'{group_var_val}')

    except IndexError:

        print('No T-SNL results to plot')

    plt.xlabel('Number of simulations')
    plt.ylabel('-log pdf at true parameters')
    plt.legend(prop={'size': 5})
    plt.savefig(os.path.join(fig_dir, f'Error-num_sims for {group_by}.png'))
    if show: 
        plt.show()


def eval_and_plot_errors(args, show=True):
    """
    Takes a set of experiments and line-plots the errors of each algorithm together.
    """

    from experiment_viewer import plt
    from util.misc import kde_error, rms_error

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        error_trials = []
        rmse_trials = []
        inf_desc = exp_desc.inf
        sim_desc = exp_desc.sim

        for trial in range(args.start, args.end + 1):

            print(exp_desc.pprint())

            try:

                exp_root = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir = exp_root + '/' + str(trial)

                try:

                    (_, true_cps), _ = util.io.load(os.path.join(exp_dir, 'gt'))

                    if isinstance(inf_desc, ed.ABC_Descriptor):
                        results = util.io.load(os.path.join(exp_dir, 'results'))
                        # samples, _, _, _, counts, _, _ = results
                        samples, weights, counts = results
                        error = kde_error(samples, true_cps)
                        rmse = rms_error(samples, true_cps)
                        num_simulations = counts * sim_desc.num_timesteps

                    elif isinstance(inf_desc, ed.BPF_MCMC_Descriptor):
                        mcmc_samples, _ = util.io.load(os.path.join(exp_dir, 'results'))
                        error = kde_error(mcmc_samples, true_cps)
                        rmse = rms_error(mcmc_samples, true_cps)
                        num_simulations = inf_desc.num_prt * sim_desc.num_timesteps * inf_desc.mcmc_steps * inf_desc.num_iters

                    elif isinstance(inf_desc, ed.SNL_Descriptor):
                        _, posterior_cond_sample = util.io.load(os.path.join(exp_dir, 'posterior'))
                        error = kde_error(posterior_cond_sample, true_cps)
                        rmse = rms_error(posterior_cond_sample, true_cps)
                        num_simulations = inf_desc.n_rounds * inf_desc.n_samples * sim_desc.num_timesteps

                    elif isinstance(inf_desc, ed.TSNL_Descriptor):
                        _, posterior_cond_sample = util.io.load(os.path.join(exp_dir, 'posterior'))
                        error = kde_error(posterior_cond_sample, true_cps)
                        rmse = rms_error(posterior_cond_sample, true_cps)
                        num_simulations = inf_desc.n_rounds * inf_desc.n_samples * sim_desc.num_timesteps

                    util.io.save((error, num_simulations), os.path.join(exp_dir, 'error'))
                    util.io.save(rmse, os.path.join(exp_dir, 'rmse'))
                    error_trials.append(error)
                    rmse_trials.append(rmse)

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

            B = 100
            key, subkey = jr.split(key)
            error_trials = jnp.array(error_trials).flatten()
            nfail = jnp.sum(jnp.isnan(error_trials)) + jnp.sum(jnp.isinf(error_trials))
            success_pct = 100 * (1 - nfail / error_trials.size)
            error_trials = error_trials[~jnp.isnan(error_trials)]
            error_trials = error_trials[~jnp.isinf(error_trials)]

            if error_trials.size == 0:

                try:

                    util.io.save_txt(f'nfails={str(nfail)}/{args.end}', os.path.join(exp_root, 'fails.txt'))

                except FileNotFoundError:

                    print('Experiment root folder not found')

            else:

                bootstrap_error, _ = util.misc.bootstrap(subkey, error_trials, B)
                error_avg = jnp.mean(bootstrap_error)
                error_std = jnp.std(bootstrap_error)

                fig, ax = plt.subplots(2, 2, figsize=(20, 10))

                ax[0, 0].hist(error_trials, bins=50, alpha=0.5, label=f'{label}', density=True)
                ax[0, 0].axvline(error_avg, color='green', linestyle='dashed', linewidth=0.5)
                ax[0, 0].axvline(error_avg - error_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[0, 0].axvline(error_avg + error_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[0, 0].set_xlabel('-logprob')
                ax[0, 0].set_ylabel('count')
                ax[0, 0].set_title(f'{label} - Error histogram %')
                ax[0, 0].legend(prop={'size': 5})

                ax[1, 0].plot(error_trials, '-o', markersize=1, label=f'{label}')
                ax[1, 0].axhline(error_avg, color='green', linestyle='dashed', linewidth=0.5, label=f'mean={error_avg:.2f}')
                ax[1, 0].axhline(error_avg - error_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[1, 0].axhline(error_avg + error_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[1, 0].set_xlabel('trial')
                ax[1, 0].set_ylabel('-logprob')
                ax[1, 0].set_title(f'{label} Error plot - Success Rate: {args.end-nfail}/{args.end}')
                ax[1, 0].legend(prop={'size':5})

                bootstrap_rmse, _ = util.misc.bootstrap(subkey, jnp.array(rmse_trials), B)
                rmse_avg = jnp.mean(bootstrap_rmse)
                rmse_std = jnp.std(bootstrap_rmse)

                ax[0, 1].hist(rmse_trials, bins=50, alpha=0.5, label=f'{label}', density=True)
                ax[0, 1].axvline(rmse_avg, color='green', linestyle='dashed', linewidth=0.5)
                ax[0, 1].axvline(rmse_avg - rmse_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[0, 1].axvline(rmse_avg + rmse_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[0, 1].set_xlabel('RMSE')
                ax[0, 1].set_ylabel('Count')
                ax[0, 1].set_title(f'{label} - RMSE histogram %')
                ax[0, 1].legend(prop={'size': 5})

                ax[1, 1].plot(rmse_trials, '-o', markersize=1, label=f'{label}')
                ax[1, 1].axhline(rmse_avg, color='green', linestyle='dashed', linewidth=0.5, label=f'mean={rmse_avg:.2f}')
                ax[1, 1].axhline(rmse_avg - rmse_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[1, 1].axhline(rmse_avg + rmse_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[1, 1].set_xlabel('Trial')
                ax[1, 1].set_ylabel('RMSE')
                ax[1, 1].set_title(f'{label} RMSE plot - Success Rate: {args.end-nfail}/{args.end}')
                ax[1, 1].legend(prop={'size':5})

                fig.savefig(os.path.join(f'{exp_root}', f'Error plots.png'))
                # fig.savefig(os.path.join(f'{exp_root}', f'Error plots.eps'))


        except FileNotFoundError:

            print('Experiment Empty')
    
    if show: 
        plt.show()


def plot_by_var(args, show=True):
    """
    Takes a set of experiments and line-plots the errors of each algorithm together.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    varname = args.varname
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    groups = {'abc': [], 'bpf_mcmc': [], 'snl': [], 'tsnl': []}
    vals = []

    for exp_desc in exp_descs:

        try:

            val = exp_desc.inf.__dict__[varname]

        except KeyError:

            print(f'Variable {varname} not found in experiment descriptor')
        
        error_trials = []
        num_sims_trials = []
        vals.append(val)

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
        bootstrap_error, _ = util.misc.bootstrap(subkey, jnp.array(error_trials), B)
        error_avg = jnp.mean(bootstrap_error)
        error_std = jnp.std(bootstrap_error)
        num_sims_avg = jnp.array(num_sims_trials).mean()
            
        if isinstance(exp_desc.inf, ed.ABC_Descriptor):
            groups['abc'].append([error_avg, error_std, num_sims_avg])
        
        elif isinstance(exp_desc.inf, ed.BPF_MCMC_Descriptor):
            groups['bpf_mcmc'].append([error_avg, error_std, num_sims_avg])

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            groups['snl'].append([error_avg, error_std, num_sims_avg])

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):
            groups['tsnl'].append([error_avg, error_std, num_sims_avg])

    vals = jnp.array(vals)
    for group in groups.keys():
        groups[group] = jnp.array(groups[group])

    try:

        plt.plot(vals, groups['abc'][:, 0], 's-', markersize=5, label=f'SMC-ABC')
        plt.fill_between(vals, groups['abc'][:, 0]-groups['abc'][:, 1], groups['abc'][:, 0]+groups['abc'][:, 1], color='blue', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No ABC results to plot')

    except KeyError:

        print('ABC has no variable {}'.format(varname))

    try:

        plt.plot(vals, groups['bpf_mcmc'][:, 0], 's-', markersize=5, label=f'BPF-MCMC')
        plt.fill_between(vals, groups['bpf_mcmc'][:, 0]-groups['bpf_mcmc'][:, 1], groups['bpf_mcmc'][:, 0]+groups['bpf_mcmc'][:, 1], color='orange', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No BPF-MCMC results to plot')

    except KeyError:

        print('BPF-MCMC has no variable {}'.format(varname))

    try:

        plt.plot(vals, groups['snl'][:, 0], 's-', markersize=5, label=f'SNL')
        plt.fill_between(vals, groups['snl'][:, 0]-groups['snl'][:, 1], groups['snl'][:, 0]+groups['snl'][:, 1], color='green', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No SNL results to plot')

    except KeyError:

        print('SNL has no variable {}'.format(varname))

    try:

        plt.plot(vals, groups['tsnl'][:, 0], 's-', markersize=5, label=f'T-SNL')
        plt.fill_between(vals, groups['tsnl'][:, 0]-groups['tsnl'][:, 1], groups['tsnl'][:, 0]+groups['tsnl'][:, 1], color='red', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No T-SNL results to plot')

    except KeyError:

        print('T-SNL has no variable {}'.format(varname))
    
    plt.xlabel(varname)
    plt.xticks(vals)
    plt.ylabel('-log pdf at true parameters')
    plt.legend(prop={'size': 5})
    plt.savefig(os.path.join(fig_dir, f'Error-{varname}.png'))
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
        _, boot_sample = util.misc.bootstrap(subkey, jnp.array(dist_trials), B)
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

def plot_mmd(args, show=True):
    """
    Plots the mmd between the samples from the true model and samples from the learned model.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)
    num_samples = 10
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        sim = misc.get_simulator(exp_desc.sim)
        sim_desc = exp_desc.sim
        inf_desc = exp_desc.inf
        sim_setup = sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)
        mmd_trials = []
        fig1, ax = plt.subplots(args.end, 1, figsize=(5, args.end*5))

        for trial in range(args.start, args.end + 1):

            print(exp_desc.pprint())

            try:

                exp_root = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir = exp_root + '/' + str(trial)

                try:

                    graphdef, state = util.io.load(os.path.join(exp_dir, 'model'))
                    (true_ps, true_cps), observations = util.io.load(exp_dir + '/gt')

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

                if sim_desc.emission_dim == 1:

                    for e in est_sample:

                        ax[trial-1].plot(e, 'o-', markersize=5, label='samples')

                    ax[trial-1].plot(observations, 'o-', markersize=5, label='observations')
                    ax[trial-1].set_title(f'Trial {trial}')
                    ax[trial-1].legend()

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')

            util.io.save(mmd_error, os.path.join(exp_dir, 'mmd'))

        mmd_trials = jnp.array(mmd_trials)
        
        B = 100
        key, subkey = jr.split(key)
        _, boot_sample = util.misc.bootstrap(subkey, mmd_trials, B)
        mean_dist = jnp.mean(boot_sample, axis=0)

        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        ax2.plot(mean_dist, 'o-', markersize=5, label='mean')
        ax2.set_xlabel('trial') 
        ax2.set_ylabel('MMD')
        ax2.legend(prop={'size': 5})

        fig2.savefig(os.path.join(f'{exp_root}', 'mmd.png'))

        if sim_desc.emission_dim == 1:

            fig1.savefig(os.path.join(f'{exp_root}', 'samples.png'))

        if show: 
            plt.show()

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