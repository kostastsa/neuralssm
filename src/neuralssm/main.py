# This module is taken/adapted from the repository ([https://github.com/gpapamak/snl.git])
# Originally authored by George Papamakarios, under the MIT License
import argparse
import os
import jax
import jax.numpy as jnp

import gc
import util.io

import experiment_descriptor as ed
import jax.random as jr
import time
import misc
import util.misc

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
            gc.collect()
            # jax.clear_backends()

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


def view_ensemble(args):
    """
    Takes a set of experiments and line-plots the errors vs num_sims of each algorithm together.
    """

    from experiment_viewer import plt

    seed = int(time.time() * 1000)
    key = jr.PRNGKey(seed)

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    abc = []
    bpf_mcmc = []
    snl = []
    tsnl = []

    for exp_desc in exp_descs:
        
        error_array = []
        num_simulations_array = []

        for trial in range(args.start, args.end + 1):

            try:

                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir += '/' + str(trial)

                try:
                    print(exp_desc.pprint())
                    error, num_simulations = util.io.load(os.path.join(exp_dir, 'error'))
                    error_array.append(error)
                    num_simulations_array.append(num_simulations)

                except FileNotFoundError:
                    print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')


        B = 100
        key, subkey = jr.split(key)
        bootstrap_errors, _ = util.misc.bootstrap(subkey, jnp.array(error_array), B)
        error = jnp.mean(bootstrap_errors)
        std = jnp.std(bootstrap_errors)
        num_sims = jnp.array(num_simulations_array).mean()
            
        if isinstance(exp_desc.inf, ed.ABC_Descriptor):
            abc.append([error, std, num_sims])
        
        elif isinstance(exp_desc.inf, ed.BPF_MCMC_Descriptor):
            bpf_mcmc.append([error, std, num_sims])

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            snl.append([error, std, num_sims])

        elif isinstance(exp_desc.inf, ed.TSNL_Descriptor):
            tsnl.append([error, std, num_sims])

    abc = jnp.array(abc)
    bpf_mcmc = jnp.array(bpf_mcmc)
    snl = jnp.array(snl)
    tsnl = jnp.array(tsnl)

    try:

        plt.plot(abc[:, -1], abc[:, 0], 'o-', markersize=5, label='SMC-ABC')
        plt.fill_between(abc[:, -1], abc[:, 0]-abc[:, 1], abc[:, 0]+abc[:, 1], color='blue', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No ABC results to plot')

    try:
       
       plt.plot(bpf_mcmc[:, -1], bpf_mcmc[:, 0], 's-', markersize=5, label='BPF-MCMC')
       plt.fill_between(bpf_mcmc[:, -1], bpf_mcmc[:, 0]-bpf_mcmc[:, 1], bpf_mcmc[:, 0]+bpf_mcmc[:, 1], color='orange', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No BPF-MCMC results to plot')

    try:

        plt.plot(snl[:, -1], snl[:, 0], '^-', markersize=5, label='SNL')
        plt.fill_between(snl[:, -1], snl[:, 0]-snl[:, 1], snl[:, 0]+snl[:, 1], color='green', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No SNL results to plot')

    try:
        plt.plot(tsnl[:, 0], 'd-', markersize=5, label='T-SNL')
        plt.fill_between(jnp.arange(tsnl[:,0].shape[0]), tsnl[:, 0]-tsnl[:, 1], tsnl[:, 0]+tsnl[:, 1], color='red', alpha=0.2, label='Confidence Band')

        # plt.plot(tsnl[:, -1], tsnl[:, 0], 'd-', markersize=5, label='T-SNL')
        # plt.fill_between(tsnl[:, -1], tsnl[:, 0]-tsnl[:, 1], tsnl[:, 0]+tsnl[:, 1], color='red', alpha=0.2, label='Confidence Band')

    except IndexError:

        print('No T-SNL results to plot')

    plt.xlabel('Number of simulations')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, f'{exp_desc}.png'))


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
        plt.savefig(os.path.join(f'{exp_root}', 'error-distance.png'))
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