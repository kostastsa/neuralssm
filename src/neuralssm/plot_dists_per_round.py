import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import misc

import inference.mcmc as mcmc
import inference.diagnostics.two_sample as two_sample
import simulators.gaussian as sim
import experiment_descriptor as ed

import util.io
import util.math
import util.plot

root = misc.get_root()
rng = np.random.RandomState(42)

prior = sim.Prior()
model = sim.Model()
true_ps, obs_xs = sim.get_ground_truth()

# for mcmc
thin = 10
n_mcmc_samples = 5000
burnin = 100





def plot_results(args):

    # NDE
    all_mmd_nde = []
    all_n_sims_nde = []
    for exp_desc in ed.parse(args.files):
        all_mmd_nde.append(get_mmd_nde(exp_desc))
        all_n_sims_nde.append(exp_desc.inf.n_samples)

    all_mmd_ppr = None
    all_n_sims_ppr = None

    all_mmd_snp = None
    all_n_sims_snp = None

    all_mmd_snl = None
    all_n_sims_snl = None

    for exp_desc in ed.parse(util.io.load_txt('exps/gauss_seq.txt')):

        # Post Prop
        if isinstance(exp_desc.inf, ed.PostProp_Descriptor):
            all_prop_mmd, post_mmd = get_mmd_postprop(exp_desc)
            all_mmd_ppr = all_prop_mmd + [post_mmd]
            all_n_sims_ppr = [(i + 1) * exp_desc.inf.n_samples_p for i in range(len(all_prop_mmd))]
            all_n_sims_ppr.append(all_n_sims_ppr[-1] + exp_desc.inf.n_samples_f)

        # SNPE
        if isinstance(exp_desc.inf, ed.SNPE_MDN_Descriptor):
            all_mmd_snp = get_mmd_snpe(exp_desc)
            all_n_sims_snp = [(i + 1) * exp_desc.inf.n_samples for i in range(exp_desc.inf.n_rounds)]

        # SNL
        if isinstance(exp_desc.inf, ed.SNL_Descriptor):
            all_mmd_snl = get_mmd_snl(exp_desc)
            all_n_sims_snl = [(i + 1) * exp_desc.inf.n_samples for i in range(exp_desc.inf.n_rounds)]

    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', size=16)

    all_n_sims = np.concatenate([all_n_sims_slk, all_n_sims_smc, all_n_sims_ppr, all_n_sims_snp, all_n_sims_nde, all_n_sims_snl])
    min_n_sims = np.min(all_n_sims)
    max_n_sims = np.max(all_n_sims)

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(all_n_sims_smc, np.sqrt(all_mmd_smc), 'v:', color='y', label='SMC ABC')
    ax.semilogx(all_n_sims_slk, np.sqrt(all_mmd_slk), 'D:', color='maroon', label='SL')
    ax.semilogx(all_n_sims_ppr, np.sqrt(all_mmd_ppr), '>:', color='c', label='SNPE-A')
    ax.semilogx(all_n_sims_snp, np.sqrt(all_mmd_snp), 'p:', color='g', label='SNPE-B')
    ax.semilogx(all_n_sims_nde, np.sqrt(all_mmd_nde), 's:', color='b', label='NL')
    ax.semilogx(all_n_sims_snl, np.sqrt(all_mmd_snl), 'o:', color='r', label='SNL')
    ax.set_xlabel('Number of simulations (log scale)')
    ax.set_ylabel('Maximum Mean Discrepancy')
    ax.set_xlim([min_n_sims * 10 ** (-0.2), max_n_sims * 10 ** 0.2])
    ax.set_ylim([0.0, ax.get_ylim()[1]])
    ax.legend(fontsize=14)

    plt.show()


def main():

    parser = argparse.ArgumentParser(description='Plotting the results for the MMD experiment.')
    parser.add_argument('sim', type=str, choices=['gauss'], help='simulator')

    plot_results()


if __name__ == '__main__':
    main()
