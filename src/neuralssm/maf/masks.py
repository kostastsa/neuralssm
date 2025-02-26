from jax import vmap, jit
from functools import partial
import jax.numpy as jnp
import jax.random as jr
import jax

@partial(jit, static_argnums=(1,2,3,4,5))
def create_degrees(key, din, dhidden, nhidden, random=False, reverse=False):
    """
    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
    :return: list of degrees
    """

    degrees = []
    keys = jr.split(key, nhidden + 1)

    degrees_0 = jnp.arange(1, din + 1)
    if random:
        degrees_0 = jr.permutation(keys[0], degrees_0, independent=True)
    if reverse:
        degrees_0 = degrees_0[::-1]
    degrees.append(degrees_0)

    for n in range(nhidden):
        min_prev_degree = jnp.min(jnp.array([jnp.min(degrees[-1]), din - 1]))
        degrees_l = jr.randint(keys[n+1], (dhidden, ), min_prev_degree, din)
        degrees.append(degrees_l)

    return degrees

@partial(jit, static_argnums=(1))
def create_masks(degrees, dcond=0):
    """
    Creates the binary masks that make the connectivity autoregressive.
    :param degrees: a list of degrees for every layer
    :return: list of all masks, as theano shared variables
    """

    Ms = []

    for _, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        M = d0[:, jnp.newaxis] <= d1
        Ms.append(M.astype(int))

    Ms[0] = jnp.concatenate([jnp.ones((dcond, len(degrees[1]))), Ms[0]])

    Mmp = degrees[-1][:, jnp.newaxis] < degrees[0]

    return Ms, Mmp.astype(int)

@partial(jit, static_argnums=(1,2,3,4,5,6))
def create_degrees2(key, din, dhidden, nhidden, dcond=0, random=False, reverse=False):
    """
    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
    :return: list of degrees
    """

    degrees = []
    keys = jr.split(key, nhidden + 1)

    degrees_cond = jnp.arange(dcond) + 1
    degrees_input = jnp.arange(dcond + 1, dcond + din + 1)
    if random:
        degrees_input = jr.permutation(keys[0], degrees_input, independent=True)
    if reverse:
        degrees_input = degrees_input[::-1]
    degrees_0 = jnp.concatenate([degrees_cond, degrees_input])
    degrees.append(degrees_0)

    min_prev_degree = jnp.max(jnp.array([dcond, 1]))
    for n in range(nhidden):
        degrees_l = jr.randint(keys[n+1], (dhidden, ), min_prev_degree, dcond + din)
        degrees.append(degrees_l)
        min_prev_degree = jnp.min(jnp.array([jnp.min(degrees[-1]), dcond + din - 1]))

    return degrees

@partial(jit, static_argnums=(1))
def create_masks2(degrees, dcond=0):
    """
    Creates the binary masks that make the connectivity autoregressive.
    :param degrees: a list of degrees for every layer
    :return: list of all masks, as theano shared variables
    """

    Ms = []

    for _, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        M = d0[:, jnp.newaxis] <= d1
        Ms.append(M.astype(int))

    Mmp = degrees[-1][:, jnp.newaxis] < degrees[0][dcond:]

    return Ms, Mmp.astype(int)