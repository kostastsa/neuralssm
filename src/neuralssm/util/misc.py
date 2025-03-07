import re
import numpy as np
import jax.numpy as jnp
from jax import lax, vmap, random as jr
import jax.scipy.stats as jss
from jax.tree_util import tree_map


def remove_whitespace(str):
    """
    Returns the string str with all whitespace removed.
    """

    p = re.compile(r'\s+')
    return p.sub('', str)


def prepare_cond_input(xy, dtype):
    """
    Prepares the conditional input for model evaluation.
    :param xy: tuple (x, y) for evaluating p(y|x)
    :param dtype: data type
    :return: prepared x, y and flag whether single datapoint input
    """

    x, y = xy
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:

        if y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])

    else:

        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], 'wrong sizes'

    return x, y, one_datapoint


def look_up(name):

    lu_field = {'i': '0', 'd': '1', 'e': '2'}
    field_cd = lu_field[name[0]]
    param_cd = str(int(name[1]) - 1)
    
    return field_cd + param_cd

def get_bool_tree(target_vars, param_names):
    r"""Returns a tree of booleans with the same structure as target_vars.
    """
    is_target = []
    target_vars = list(map(lambda var: look_up(var), target_vars))
    for i, field in enumerate(param_names):
        sub_is_target = []
        for j, _ in enumerate(field):
            if str(i) + str(j) in target_vars:
                sub_is_target.append(True)
            else:
                sub_is_target.append(False)
        is_target.append(sub_is_target)
    return is_target


def get_prior_fields(init_vals, param_dists, is_target):    
    r"""Sample parameters from the prior distribution. When prior field is tfd.Distribution,
        sample num_samples values from it, and set the corresponding values of is_constrained to False
        (we consider that parameters are sampled always in unconstrained form). Otherwise set the value 
        equal to the provided value and is_constrained to True.
    """
    return tree_map(lambda b, l1, l2: l2 if b else l1, is_target, init_vals, param_dists)


def swap_axes_on_values(outputs, axis1=0, axis2=1):
    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))


def kmeans(key, data, num_clusters, tol=1e-4):
    dx = data.shape[1]

    def cond_fun(carry):
        old_centroids, new_centroids = carry
        any_nans = jnp.isnan(new_centroids).any()
        return jnp.logical_and(jnp.linalg.norm(old_centroids - new_centroids) > tol, any_nans)

    def _step(carry):
        _, centroids = carry
        distances = vmap(lambda x: vmap(lambda x, mu: jnp.linalg.norm(x - mu), in_axes=(None, 0))(x, centroids))(data)
        cluster_assignments = jnp.argmin(distances, axis=1)
        new_centroids = vmap(lambda i: jnp.sum(jnp.where(jnp.tile(cluster_assignments==i, (dx,1)).T, data, 0.0), axis=0) / jnp.sum(cluster_assignments==i))(jnp.arange(num_clusters))
        carry = (centroids, new_centroids)
        return carry

    init_carry = (jnp.zeros((num_clusters, dx)), jr.choice(key, data, (num_clusters,)))
    out = lax.while_loop(cond_fun, _step, init_carry)
    
    return out[1]


def kde_error(positions, true_cps):
        
        kernel_points = positions.T
        kde = jss.gaussian_kde(kernel_points)
        error = -jnp.log(kde.evaluate(true_cps))
        return error


def rms_error(cps, true_cps):

    return jnp.linalg.norm(jnp.mean(cps, axis=0) - true_cps)


def bootstrap(key, rmse_array, B):
    N = rmse_array.shape[0]
    rmse_boot = jnp.zeros((B,))
    boot_samples = []
    for b in range(B):
        key, subkey = jr.split(key)
        ids = jr.randint(subkey, (N,), 0, N)
        boot = rmse_array[ids]
        rmse_boot = rmse_boot.at[b].set(jnp.mean(boot))
        boot_samples.append(boot)
    boot_samples = jnp.stack(boot_samples, axis=0)
    return rmse_boot, boot_samples


def compute_distances(emissions, observations, num_timesteps, emission_dim):
    
    distances = vmap(lambda sim_emissions: jnp.linalg.norm(observations - sim_emissions) / jnp.sqrt(num_timesteps * emission_dim))(emissions)

    return distances


def clear_nans(errors):
    means = errors[:, 0]
    stds = errors[:, 1]
    nsims = errors[:, 2]
    nans_infs = jnp.isnan(means) + jnp.isinf(means)
    means = means[~nans_infs]
    stds = stds[~nans_infs]
    nsims = jnp.log(nsims[~nans_infs])
    
    nfail = jnp.sum(nans_infs)
    success_pct = (1 - nfail / errors.size)
    errors = errors[~jnp.isnan(errors)]
    errors = errors[~jnp.isinf(errors)]

    out = jnp.array([means, stds, nsims]).T

    return out, nfail, success_pct