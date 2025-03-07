import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm')
from abc import ABC
from abc import abstractmethod
from parameters import ParamSSM
from jaxtyping import Array, Float
from typing import Optional, Tuple, Callable
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jax.tree_util import tree_map
from jax import lax, vmap, debug
from jax.scipy.special import logsumexp as lse, gammaln
import jax.numpy as jnp
import jax.random as jr
from dynamax.types import PRNGKey # type: ignore

class SSM(ABC):
    r"""A base class for state space models. Such models consist of parameters, which
    we may learn, as well as hyperparameters, which specify static properties of the
    model. This base class allows parameters to be indicated a standardized way
    so that they can easily be converted to/from unconstrained form for optimization.

    **Abstract Methods**

    Models that inherit from `SSM` must implement a few key functions and properties:

    * :meth:`initial_distribution` returns the distribution over the initial state given parameters
    * :meth:`dynamics_simulator` simulates the next state given the current state and parameters
    * :meth:`emission_simulator` simulates the emission given the current state and parameters
    * :meth:`log_prior` (optional) returns the log prior probability of the parameters
    * :attr:`emission_shape` returns a tuple specification of the emission shape
    * :attr:`inputs_shape` returns a tuple specification of the input shape, or `None` if there are no inputs.
    """

    @abstractmethod
    def initial_distribution(
        self,
        params: ParamSSM,
        inputs: Optional[Float[Array, "input_dim"]]
    ) -> tfd.Distribution:
        r"""Return an initial distribution over latent states.

        Args:
            params: model parameters $\theta$
            inputs: optional  inputs  $u_t$

        Returns:
            distribution over initial latent state, $p(z_1 \mid \theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def dynamics_simulator(
        self,
        key: PRNGKey,
        params: ParamSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]
    ) -> tfd.Distribution:
        r"""Next latent state given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of next latent state $p(z_{t+1} \mid z_t, u_t, \theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def emission_simulator(
        self,
        key: PRNGKey,
        params: ParamSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        r"""Return a new emission given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            current emission 

        """
        raise NotImplementedError
    
    @abstractmethod
    def emission_log_prob(
        self,
        params: ParamSSM,
        emissions: Float[Array, "emission_dim"],
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None):
        r"""Return a distribution over emissions given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of current emission $p(y_t \mid z_t, u_t, \theta)$

        """
        raise NotImplementedError

    # All SSMs support sampling
    def simulate(
        self,
        key: PRNGKey,
        params: ParamSSM,
        num_timesteps: int,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
              Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            inputs: inputs $u_{1:T}$

        Returns:
            latent states and emissions

        """
        def _step(prev_state, args):
            key, inpt = args
            key1, key2, key = jr.split(key, 3)
            state = self.dynamics_simulator(key1, params, prev_state, inpt)
            emission = self.emission_simulator(key2, params, state, inpt)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_state = self.initial_distribution(params, initial_input).sample(seed=key1)
        initial_emission = self.emission_simulator(key2, params, initial_state, initial_input)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_inputs))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

class LGSSM(SSM):

    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        input_dim: int=0
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim

    def initial_distribution(
        self,
        params: ParamSSM,
        inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean.value, params.initial.cov.value)

    def dynamics_simulator(
        self,
        key: PRNGKey,
        params: ParamSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None):
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.dynamics.weights.value @ state + params.dynamics.input_weights.value @ inputs + params.dynamics.bias.value
        new_state = MVN(mean, params.dynamics.cov.value).sample(1, key)
        return new_state.flatten()

    def emission_simulator(
        self,
        key: PRNGKey,
        params: ParamSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None):
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.emissions.weights.value @ state + params.emissions.input_weights.value @ inputs + params.emissions.bias.value
        new_emission =  MVN(mean, params.emissions.cov.value).sample(1, key)
        return new_emission.flatten()
    
    def emission_log_prob(
        self,
        params: ParamSSM,
        emission: Float[Array, "emission_dim"],
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None):
        mean = params.emissions.weights.value @ state + params.emissions.input_weights.value @ inputs + params.emissions.bias.value
        cov = params.emissions.cov.value
        dist = MVN(mean, cov)
        return dist.log_prob(emission)
    
class SV(SSM):
    
    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        input_dim: int=0
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim

    def initial_distribution(
        self,
        params: ParamSSM,
        inputs: Optional[Float[Array, "input_dim"]]=None) -> tfd.Distribution:
        return MVN(params.initial.mean.value, params.initial.cov.value)

    def dynamics_simulator(
        self,
        key: PRNGKey,
        params: ParamSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None):
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.dynamics.weights.value @ state + params.dynamics.input_weights.value @ inputs + params.dynamics.bias.value
        new_state = MVN(mean, params.dynamics.cov.value).sample(1, key)
        return new_state.flatten()

    def emission_simulator(
        self,
        key: PRNGKey,
        params: ParamSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None):
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        r = MVN(params.emissions.bias.value, params.emissions.cov.value).sample(1, key)
        new_emission = params.emissions.beta.value * jnp.diag(jnp.exp(state / params.emissions.sigma.value[0])) @ r
        return new_emission.flatten()
    
    def emission_log_prob(
        self,
        params: ParamSSM,
        emission: Float[Array, "emission_dim"],
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None):
        V = params.emissions.beta.value * jnp.diag(jnp.exp(state / params.emissions.sigma.value[0]))
        mean = V @ params.emissions.bias.value
        cov = V @ params.emissions.cov.value @ V.T
        dist = MVN(mean, cov)
        return dist.log_prob(emission)

class SPN(SSM):
    '''
    Stochastic petri net SSM
    '''

    def __init__(
        self,
        state_dim: int,
        num_reactions: int,
        emission_dim: int,
        emission_dist: Callable,
        input_dim: int=0,
        dt_obs: float=1.0
    ):
        self.state_dim = state_dim # number of species
        self.num_reactions = num_reactions # number of reactions
        self.emission_dim = emission_dim # number of observables
        self.emission_dist = emission_dist # emission function
        self.input_dim = input_dim
        self.dt_obs = dt_obs

    log_power_hazard_fn = lambda self, state, pre, log_rates: vmap(lambda row, log_rate: log_rate + jnp.sum(vmap(lambda b, e: e * jnp.log(b))(state, row)))(pre, log_rates)
    log_bin =lambda self, N, k: gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
    sum_log_bin = lambda self, x, n: jnp.sum(vmap(self.log_bin)(x, n))
    log_bin_hazard_fn = lambda self, pop_state, pre, log_rates: vmap(lambda row, log_rate: log_rate + self.sum_log_bin(pop_state, row))(pre, log_rates)

    def log_hazard_fn(self, pop_state, pre, log_rates): 
        return self.log_bin_hazard_fn(pop_state, pre, log_rates)

    def initial_distribution(
        self,
        params: ParamSSM,
        inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean.value, params.initial.cov.value)

    def dynamics_simulator(
        self,
        key: PRNGKey, 
        params: ParamSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
        ):
        '''
        Simulate the next state given the current state and parameters. The 
        '''

        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        S = (params.dynamics.post.value - params.dynamics.pre.value)

        def _cond_fn(val):

            state, _, _ = val
            
            return state[0] < self.dt_obs
        
        def _while_step(_):

            state, ireactions, key = _
            sumtime = state[0]
            pop_state = state[1:]

            key, subkey = jr.split(key)
            lh = self.log_hazard_fn(pop_state, params.dynamics.pre.value, params.dynamics.log_rates.value)
            lh0 = lse(lh)
            dt = tfd.Exponential(jnp.exp(lh0)).sample(seed=subkey)
            lh -= jnp.max(lh)
            prob_vec = jnp.exp(lh) / jnp.sum(jnp.exp(lh))
            
            key, subkey = jr.split(key)
            event = tfd.Categorical(probs=prob_vec).sample(seed=subkey)

            pop_state += S[event]
            pop_state = jnp.maximum(pop_state, 0.0)
            sumtime += dt
            ireactions += 1
            new_state = jnp.pad(pop_state, (1, 0), mode='constant', constant_values=sumtime)

            return new_state, ireactions, key

        carry = (state, 0, key)
        next_state, _, _ = lax.while_loop(_cond_fn, _while_step, carry)
        sumtime = next_state[0]
        pop_state = next_state[1:]
        next_state = jnp.pad(pop_state, (1, 0), mode='constant', constant_values=sumtime-self.dt_obs)
        
        return next_state

    def emission_simulator(
        self,
        key: PRNGKey,
        params: ParamSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        new_emission = self.emission_dist(params, state).sample(seed=key)
        return new_emission.flatten()
    
    def emission_log_prob(
        self,
        params: ParamSSM,
        emission: Float[Array, "emission_dim"],
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None):
        state = state[1:]
        return self.emission_dist(params, state).log_prob(emission)

    def simulate(
        self,
        key: PRNGKey,
        params: ParamSSM,
        num_timesteps: int,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
              Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            inputs: inputs $u_{1:T}$

        Returns:
            latent states and emissions

        """

        def _step(prev_state, args):

            key, inpt = args
            key1, key2 = jr.split(key)
            state = self.dynamics_simulator(key1, params, prev_state, inpt)
            emission = self.emission_simulator(key2, params, state, inpt)
            
            return state, (state[1:], emission)

        # Sample the initial state
        key, subkey = jr.split(key)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_state = jnp.pad(self.initial_distribution(params, initial_input).sample(seed=subkey), (1, 0))

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (states, emissions) = lax.scan(_step, initial_state, (next_keys, next_inputs))

        return states, emissions