import experiment_descriptor as ed
import sys
import os
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt

def create_descs(
    ssm,
    alg,
    target_vars,
    num_timesteps,
    dt_obs,
    state_dim,
    emission_dim,
    ):

    exp_descs = ''

    sim = ed.SimulatorDescriptor().get_descriptor(ssm)
    inf = ed.InferenceDescriptor().get_descriptor(alg)

    scan_dims = [1, 2, 3, 4, 5, 6 ,7, 8, 9, 10]
    scan_abc = [10, 50, 100, 1000]
    scan_mcmc = [100, 200, 500, 1000]
    scan_snl = [50, 100, 500, 1000]
    scan_tsnl = [1, 10, 100, 500]

        
    if alg == 'smc_abc':

        scan_init = scan_abc[0]
        scan_fin  = scan_abc[-1]

        q_max = 0.7
        sigma = 1.0

        for val in scan_abc:

            num_prt = val
            sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, dt_obs, target_vars)
            inf_desc = inf.create_desc(num_prt, q_max, sigma)

            exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc)
            exp_descs += '\n'

    elif alg == 'bpf_mcmc':

        scan_init = scan_mcmc[0]
        scan_fin  = scan_mcmc[-1]

        for val in scan_mcmc:

            num_prt = val
            mcmc_steps = 10000
            num_iters = 1
            sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, dt_obs, target_vars)
            inf_desc = inf.create_desc(num_prt, num_iters, mcmc_steps)

            exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc)
            exp_descs += '\n'

    elif alg == 'snl':

        model_args = (5, 32, 5, 'relu', False, True, False, False, 20, 1e-4)
        mcmc_steps = 1000
        n_rounds = 5
        mcmc_steps = 1000
        train_on = 'best'

        scan_init = scan_snl[0]
        scan_fin  = scan_snl[-1]

        for val in scan_snl:
            
            mcmc_steps = 1000
            train_on = 'best'
            n_samples = val

            if n_samples <= 100:

                train_on = 'all'

            if mcmc_steps <= n_samples:

                mcmc_steps = 5 * mcmc_steps

            sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, dt_obs, target_vars)
            inf_desc = inf.create_desc(model_args, n_samples, n_rounds, train_on, mcmc_steps)

            exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc)
            exp_descs += '\n'

    elif alg == 'tsnl':

        lag = 10
        model_args = (5, 32, 5, 'relu', False, True, False, False, 20, 1e-4)
        mcmc_steps = 1000
        n_rounds = 5
        train_on = 'best'
        mcmc_steps = 1000
        subsample = 1.0

        scan_init = scan_tsnl[0]
        scan_fin  = scan_tsnl[-1]

        for val in scan_tsnl:

            n_samples = val
            train_on = 'best'
            mcmc_steps = 1000

            if n_samples < 100:

                train_on = 'all'

            sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, dt_obs, target_vars)
            inf_desc = inf.create_desc(model_args, n_samples, n_rounds, lag, subsample, train_on, mcmc_steps)
        
            exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc)
            exp_descs += '\n'
    
    return exp_descs, (scan_init, scan_fin)

if __name__ == '__main__':

    exp_root = '/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm/exps'

    target_vars = 'd4'
    ssm = 'lgssm'
    scan = 'num_sims'

    state_dim = 2
    emission_dim = 2

    for alg in ['smc_abc', 'bpf_mcmc', 'snl', 'tsnl']:

        exp_descs, (scan_init, scan_fin) = create_descs(
            ssm = ssm,
            alg = alg,
            target_vars = 'd4',
            num_timesteps = 100,
            dt_obs = None,
            state_dim = state_dim,
            emission_dim = emission_dim
            )

        os.makedirs(exp_root + f'/{ssm}/{target_vars}/state_dim_{state_dim}', exist_ok=True)
        with open(exp_root + f'/{ssm}/{target_vars}/state_dim_{state_dim}/{ssm}_{target_vars}_{alg}_scan_{scan}_{scan_init}_{scan_fin}.txt', 'w') as f:

            f.write(exp_descs)