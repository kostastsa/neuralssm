import experiment_descriptor as ed
import sys
import os
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt

INF_ALGS = ['smc_abc', 'bpf_mcmc', 'snl', 'tsnl']


def create_descs(
    ssm,
    alg,
    target_vars,
    num_timesteps,
    dt_obs,
    state_dim,
    emission_dim,
    different_inference_methods=False,
    ):

    exp_descs = ''

    sim = ed.SimulatorDescriptor().get_descriptor(ssm)

    scan_abc = [1000]
    scan_mcmc = [1000]
    scan_snl = [500]
    scan_tsnl = [500]

    # scan_timesteps = [100 * i for i in range(1, 11)]
    # scan_abc = scan_timesteps
    # scan_mcmc = scan_timesteps
    # scan_snl = scan_timesteps
    # scan_tsnl = scan_timesteps


    algs = alg if isinstance(alg, (list, tuple)) else [alg]

    if not different_inference_methods:
        algs = [algs[0]]

    scan_inits = []
    scan_fins = []

    for alg in algs:

        inf = ed.InferenceDescriptor().get_descriptor(alg)

        if alg == 'smc_abc':

            scan_init = scan_abc[0]
            scan_fin  = scan_abc[-1]
            scan_inits.append(scan_init)
            scan_fins.append(scan_fin)

            q_max = 0.9
            sigma = 1.0
            # num_prt = 1000

            for val in scan_abc:

                num_prt = val
                sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, dt_obs, target_vars)
                inf_desc = inf.create_desc(num_prt, q_max, sigma)

                exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc)
                exp_descs += '\n'

        elif alg == 'bpf_mcmc':

            scan_init = scan_mcmc[0]
            scan_fin  = scan_mcmc[-1]
            scan_inits.append(scan_init)
            scan_fins.append(scan_fin)
            # num_prt = 1000

            for val in scan_mcmc:

                num_prt = val
                mcmc_steps = 1000
                num_iters = 1
                sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, dt_obs, target_vars)
                inf_desc = inf.create_desc(num_prt, num_iters, mcmc_steps)

                exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc)
                exp_descs += '\n'

        elif alg == 'snl':

            nmades, dhidden, nhidden = 5, 32, 5
            act_fun, random_order, reverse = 'tanh', False, True
            batch_norm, dropout = False, False
            nepochs, lr = 20, 1e-6

            model_args = (nmades, dhidden, nhidden, act_fun, random_order, reverse, batch_norm, dropout, nepochs, lr)
            mcmc_steps = 1000
            n_rounds = 5
            mcmc_steps = 1000
            train_on = 'all'
            # n_samples = 100

            scan_init = scan_snl[0]
            scan_fin  = scan_snl[-1]
            scan_inits.append(scan_init)
            scan_fins.append(scan_fin)

            for val in scan_snl:
                
                sampler = 'rwm'
                mcmc_steps = 1000
                train_on = 'best'
                n_samples = val

                if n_samples < 100:

                    train_on = 'all'

                if mcmc_steps <= n_samples:

                    mcmc_steps = 5 * mcmc_steps

                sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, dt_obs, target_vars)
                inf_desc = inf.create_desc(model_args, n_samples, n_rounds, train_on, sampler, mcmc_steps)

                exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc)
                exp_descs += '\n'

        elif alg == 'tsnl':

            lag = 15
            nmades, dhidden, nhidden = 5, 32, 5
            act_fun, random_order, reverse = 'relu', False, True
            batch_norm, dropout = False, False
            nepochs, lr = 20, 1e-4

            model_args = (nmades, dhidden, nhidden, act_fun, random_order, reverse, batch_norm, dropout, nepochs, lr)
            n_rounds = 5
            mcmc_steps = 1000
            subsample = 1.0
            # n_samples = 100

            scan_init = scan_tsnl[0]
            scan_fin  = scan_tsnl[-1]
            scan_inits.append(scan_init)
            scan_fins.append(scan_fin)

            for val in scan_tsnl:

                n_samples = val
                train_on = 'best'
                sampler = 'rwm'
                mcmc_steps = 1000

                if n_samples < 100:

                    train_on = 'all'

                sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, dt_obs, target_vars)
                inf_desc = inf.create_desc(model_args, n_samples, n_rounds, lag, subsample, train_on, sampler, mcmc_steps)
            
                exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc)
                exp_descs += '\n'
    
    return exp_descs, (min(scan_inits), max(scan_fins))

if __name__ == '__main__':

    exp_root = '/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm/exps'

    target_vars = 'd4'
    ssm = 'lgssm'
    scan = 'num_sims'

    state_dim = 1
    emission_dim = 1

    different_inference_methods = True
    algs = INF_ALGS
    exp_dir = exp_root + f'/{ssm}/{target_vars}/state_dim_{state_dim}'

    os.makedirs(exp_dir, exist_ok=True)

    if different_inference_methods:

        exp_descs, (scan_init, scan_fin) = create_descs(
            ssm = ssm,
            alg = algs,
            target_vars = target_vars,
            num_timesteps = 100,
            dt_obs = 1.0,
            state_dim = state_dim,
            emission_dim = emission_dim,
            different_inference_methods = different_inference_methods,
            )

        alg_label = 'all_inf_methods'
        with open(exp_dir + f'/{ssm}_{target_vars}_{alg_label}.txt', 'w') as f:

            f.write(exp_descs)

    else:

        for alg in algs:

            exp_descs, (scan_init, scan_fin) = create_descs(
                ssm = ssm,
                alg = alg,
                target_vars = target_vars,
                num_timesteps = 100,
                dt_obs = 1.0,
                state_dim = state_dim,
                emission_dim = emission_dim,
                different_inference_methods = different_inference_methods,
                )

            with open(exp_dir + f'/{ssm}_{target_vars}_{alg}_scan_{scan}_{scan_init}_{scan_fin}.txt', 'w') as f:

                f.write(exp_descs)
