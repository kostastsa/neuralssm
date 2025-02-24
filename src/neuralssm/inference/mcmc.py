import jax.random as jr
from jax import lax, vmap, debug, jit
import logging
import blackjax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import Optional
from util.sample import resample
from jaxtyping import Array, Float, Int
from parameters import ParamSSM
from util.param import from_conditional, log_prior, sample_prior, to_train_array
from util.misc import swap_axes_on_values
from util.sample import mcmc_inference_loop
from functools import partial
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)


class BPF_MCMC:
    """
    Implements bpf-mcmc.
    """

    def __init__(self, props, ssm):

        self.props = props
        self.ssm = ssm
        self.xparam = None
        self.time = None

    def bpf(self,
            params: ParamSSM,
            observations: Float[Array, "ntime emission dim"],
            num_particles: Int,
            key: jr.PRNGKey = jr.PRNGKey(0),
            inputs: Optional[Float[Array, "ntime input dim"]] = None, 
            ess_threshold: float = 0.5
           ):
        r"""
        Bootstrap particle filter for the nonlinear state space model.
        Args:
            params: Parameters of the nonlinear state space model.
            emissions: Emissions.
            num_particles: Number of particles.
            rng_key: Random number generator key.
            inputs: Inputs. 

        Returns:
            Posterior particle filtered.
        """

        num_timesteps = len(observations)

        # Dynamics and emission functions
        inputs = inputs if inputs is not None else jnp.zeros((num_timesteps, self.ssm.input_dim))

        
        def _step(carry, t):
            weights, ll, particles, key = carry
            
            # Get parameters and inputs for time index t
            u = inputs[t]
            y = observations[t]

            # Sample new particles 
            keys = jr.split(key, num_particles+1)
            next_key = keys[0]
            map_sample_particles = vmap(self.ssm.dynamics_simulator, in_axes=(0,None,0,None))
            new_particles = map_sample_particles(keys[1:], params, particles, u)

            # Compute weights 
            map_log_prob = vmap(self.ssm.emission_log_prob, in_axes=(None,None,0,None))
            lls = map_log_prob(params,y,new_particles,u)
            lls -= jnp.max(lls)
            ls = jnp.exp(lls)
            weights = jnp.multiply(ls, weights)
            ll += jnp.log(jnp.sum(weights))
            new_weights = weights / jnp.sum(weights)

            # Resample if necessary
            resample_cond = 1.0 / jnp.sum(jnp.square(new_weights)) < ess_threshold * num_particles
            weights, new_particles, next_key = lax.cond(resample_cond, resample, lambda *args: args, new_weights, new_particles, next_key)

            outputs = {
                'weights':weights,
                'particles':new_particles
            }

            carry = (weights, ll, new_particles, next_key)

            return carry, outputs
        
        keys = jr.split(key, num_particles+1)
        next_key = keys[0]
        weights = jnp.ones(num_particles) / num_particles
        map_sample = vmap(MVN(loc=params.initial.mean.value, covariance_matrix=params.initial.cov.value).sample, in_axes=(None,0))
        particles = map_sample((), keys[1:])
        carry = (weights, 0.0, particles, next_key)

        out_carry, outputs =  lax.scan(_step, carry, jnp.arange(num_timesteps))
        ll = out_carry[1]
        outputs = swap_axes_on_values(outputs)
    
        return outputs, ll
    
    def logdensity_fn(self,
                      cond_params, 
                      key, 
                      emissions, 
                      num_prt, 
                      num_iters):
        
        params = from_conditional(self.xparam, self.props, cond_params)
        lps = []
        for _ in range(num_iters):
            key, subkey = jr.split(key)
            _, lp = self.bpf(params, emissions, num_prt, subkey)
            lp += log_prior(cond_params, self.props)
            lps.append(lp)
        return jnp.mean(jnp.array(lps))
    
    def run(self,
            key,
            observations,
            num_prt: int,
            mcmc_steps: int,
            num_posterior_samples: int = 1000,
            num_iters: int = 1,
            logger = None,
            rw_sigma : float = 0.1,
            ):

        logger.write('-----------------------\n')
        logger.write('Sampling BPF-MCMC chain\n')
        logger.write('-----------------------\n')
        tin = time.time()

        key, subkey = jr.split(key)
        initial_param = sample_prior(subkey, self.props, 1)[0]
        self.xparam = initial_param
        initial_cond_params = to_train_array(initial_param, self.props)
        key, subkey = jr.split(key)
        logdensity_fn = partial(self.logdensity_fn, key=subkey, emissions=observations, num_prt=num_prt, num_iters=num_iters)

        bpf_random_walk = blackjax.additive_step_random_walk(logdensity_fn, blackjax.mcmc.random_walk.normal(rw_sigma))
        bpf_initial_state = bpf_random_walk.init(initial_cond_params)
        bpf_kernel = jit(bpf_random_walk.step)

        ### Run inference loop
        mcmc_samples = mcmc_inference_loop(key, bpf_kernel, bpf_initial_state, mcmc_steps)

        tout = time.time()
        self.time = tout-tin
        logger.write(f'MCMC runtime: {tout-tin}\n')
        
        return mcmc_samples[-num_posterior_samples:] 
    

class MCMC_Sampler:
    """
    Superclass for MCMC samplers.
    """

    def __init__(self, x, lp_f, thin):
        """
        :param x: initial state
        :param lp_f: function that returns the log prob
        :param thin: amount of thinning; if None, no thinning
        """

        self.x = np.array(x, dtype=float)
        self.lp_f = lp_f
        self.L = lp_f(self.x)
        self.thin = 1 if thin is None else thin
        self.n_dims = self.x.size if self.x.ndim == 1 else self.x.shape[1] #state dim

    def set_state(self, x):
        """
        Sets the state of the chain to x.
        """

        self.x = np.array(x, dtype=float)
        self.L = self.lp_f(self.x)

    def gen(self, n_samples):
        """
        Generates MCMC samples. Should be implemented in a subclass.
        """
        raise NotImplementedError('Should be implemented as a subclass.')


class GaussianMetropolis(MCMC_Sampler):
    """
    Metropolis algorithm with isotropic gaussian proposal.
    """

    def __init__(self, x, lp_f, step, thin=None):
        """
        :param x: initial state
        :param lp_f: function that returns the log prob
        :param step: std of gaussian proposal
        :param thin: amount of thinning; if None, no thinning
        """

        MCMC_Sampler.__init__(self, x, lp_f, thin)
        self.step = step

    def gen(self, n_samples, logger=sys.stdout, show_info=False, rng=np.random):
        """
        :param n_samples: number of samples
        :param logger: logger for logging messages. If None, no logging takes place
        :param show_info: whether to plot info at the end of sampling
        :param rng: random number generator to use
        :return: numpy array of samples
        """

        assert n_samples >= 0, 'number of samples can''t be negative'

        n_acc = 0
        L_trace = []
        acc_rate_trace = []
        samples = np.empty([n_samples, self.n_dims])
        logger = open(os.devnull, 'w') if logger is None else logger

        for n in range(n_samples):

            for _ in range(self.thin):

                # proposal
                x = self.x + self.step * rng.randn(*self.x.shape)

                # metropolis acceptance step
                L_new = self.lp_f(x)
                if rng.rand() < np.exp(L_new - self.L):
                    self.x = x
                    self.L = L_new
                    n_acc += 1

            samples[n] = self.x

            # acceptance rate
            acc_rate = n_acc / float(self.thin * (n+1))
            logger.write('sample = {0}, acc rate = {1:.2%}, log prob = {2:.2}\n'.format(n+1, acc_rate, self.L))

            # record traces
            if show_info:
                L_trace.append(self.L)
                acc_rate_trace.append(acc_rate)

        # show plot with the traces
        if show_info:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(L_trace)
            ax[0].set_ylabel('log probability')
            ax[1].plot(acc_rate_trace)
            ax[1].set_ylim([0, 1])
            ax[1].set_ylabel('acceptance rate')
            ax[1].set_xlabel('samples')
            plt.show(block=False)

        return samples


class SliceSampler(MCMC_Sampler):
    """
    Slice sampling for multivariate continuous probability distributions.
    It cycles sampling from each conditional using univariate slice sampling.
    """

    def __init__(self, x, lp_f, max_width=float('inf'), thin=None):
        """
        :param x: initial state
        :param lp_f: function that returns the log prob
        :param max_width: maximum bracket width
        :param thin: amount of thinning; if None, no thinning
        """

        MCMC_Sampler.__init__(self, x, lp_f, thin)
        self.max_width = max_width
        self.width = None

    def gen(self, n_samples, logger=sys.stdout, show_info=False, rng=np.random):
        """
        :param n_samples: number of samples
        :param logger: logger for logging messages. If None, no logging takes place
        :param show_info: whether to plot info at the end of sampling
        :param rng: random number generator to use
        :return: numpy array of samples
        """

        assert n_samples >= 0, 'number of samples can''t be negative'

        order = range(self.n_dims)
        L_trace = []
        samples = np.empty([n_samples, self.n_dims])
        logger = open(os.devnull, 'w') if logger is None else logger

        if self.width is None:
            logger.write('tuning bracket width...\n')
            self._tune_bracket_width(rng)

        for n in range(n_samples):

            for _ in range(self.thin):

                rng.shuffle(order)

                for i in order:
                    self.x[i], _ = self._sample_from_conditional(i, self.x[i], rng)

            samples[n] = self.x.copy()

            self.L = self.lp_f(self.x)
            logger.write('sample = {0}, log prob = {1:.2}\n'.format(n+1, self.L))

            if show_info:
                L_trace.append(self.L)

        # show trace plot
        if show_info:
            fig, ax = plt.subplots(1, 1)
            ax.plot(L_trace)
            ax.set_ylabel('log probability')
            ax.set_xlabel('samples')
            plt.show(block=False)

        return samples

    def _tune_bracket_width(self, rng):
        """
        Initial test run for tuning bracket width.
        Note that this is not correct sampling; samples are thrown away.
        :param rng: random number generator to use
        """

        n_samples = 50
        order = range(self.n_dims)
        x = self.x.copy()
        self.width = np.full(self.n_dims, 0.01)

        for n in range(n_samples):

            rng.shuffle(order)

            for i in range(self.n_dims):
                x[i], wi = self._sample_from_conditional(i, x[i], rng)
                self.width[i] += (wi - self.width[i]) / (n + 1)

    def _sample_from_conditional(self, i, cxi, rng):
        """
        Samples uniformly from conditional by constructing a bracket.
        :param i: conditional to sample from
        :param cxi: current state of variable to sample
        :param rng: random number generator to use
        :return: new state, final bracket width
        """

        # conditional log prob
        Li = lambda t: self.lp_f(np.concatenate([self.x[:i], [t], self.x[i+1:]]))
        wi = self.width[i]

        # sample a slice uniformly
        logu = Li(cxi) + np.log(1.0 - rng.rand())

        # position the bracket randomly around the current sample
        lx = cxi - wi * rng.rand()
        ux = lx + wi

        # find lower bracket end
        while Li(lx) >= logu and cxi - lx < self.max_width:
            lx -= wi

        # find upper bracket end
        while Li(ux) >= logu and ux - cxi < self.max_width:
            ux += wi

        # sample uniformly from bracket
        xi = (ux - lx) * rng.rand() + lx

        # if outside slice, reject sample and shrink bracket
        while Li(xi) < logu:
            if xi < cxi:
                lx = xi
            else:
                ux = xi
            xi = (ux - lx) * rng.rand() + lx

        return xi, ux - lx
