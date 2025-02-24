from jax import numpy as jnp
from jax import random as jr

def sim_data(gen_params, sim_model, n_samples=None, rng_key=jr.PRNGKey(0)):
    """
    Simulates a and returns a given number of samples from the simulator.
    Takes care of failed simulations, and guarantees the exact number of requested samples will be returned.
    If number of samples is None, it returns one sample.
    """

    if n_samples is None:
        ps, xs = sim_data(gen_params, sim_model, n_samples=1, rng_key=rng_key)
        return ps[0], xs[0]

    assert n_samples > 0

    ps = None
    xs = None

    while True:

        # simulate parameters and data
        ps = gen_params(n_samples, rng_key=rng_key)
        xs = sim_model(rng_key, ps)

        # filter out simulations that failed
        idx = [x is not None for x in xs]

        if not jnp.any(idx):
            continue

        if not jnp.all(idx):
            ps = jnp.stack(ps[idx])
            xs = jnp.stack(xs[idx])

        break  # we'll break only when we have at least one successful simulation

    n_rem = n_samples - ps.shape[0]
    assert n_rem < n_samples

    if n_rem > 0:
        # request remaining simulations
        ps_rem, xs_rem = sim_data(gen_params, sim_model, n_rem, rng_key)
        ps = jnp.concatenate([ps, ps_rem], axis=0)
        xs = jnp.concatenate([xs, xs_rem], axis=0)

    assert ps.shape[0] == xs.shape[0] == n_samples

    return ps, xs