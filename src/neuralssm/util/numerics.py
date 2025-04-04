import jax.numpy as jnp
from jax import vmap, lax, random as jr
import jax.scipy.stats as jss
from jax.scipy.special import logsumexp as lse


def kde_error(particles, true_cps, sigma=1.0):
        
    num_prt = particles.shape[0]
    dim = particles.shape[1]
    dps = particles - true_cps
    ds = - jnp.linalg.norm(dps, axis=1)**2 / 2 / sigma**2
    logp = lse(ds) #- jnp.log(num_prt) - (dim/2) * jnp.log(2 * jnp.pi * sigma ** 2)

    return -logp

def min_error(particles, true_cps, sigma=1.0):

    num_prt = particles.shape[0]
    dim = particles.shape[1]
    dps = particles - true_cps
    ns = jnp.linalg.norm(dps, axis=1)
    
    return (1/jnp.sqrt(2)/sigma**2) * jnp.min(ns)


def post_mean_error(particles, true_cps):

    return jnp.linalg.norm(jnp.mean(particles, axis=0) - true_cps)


def rmse(particles, true_cps):

    dps = particles - true_cps
    nsq = jnp.linalg.norm(dps, axis=1) ** 2
    mse = jnp.mean(nsq)

    return jnp.sqrt(mse)


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

    # ensure emissions has a leading dimension
    if emissions.ndim == 2:
        emissions = emissions[None, ...]
    
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


def compute_acf(y, max_lag):

    """Computes the autocorrelation function (ACF) for each dimension and the autocorrelation matrix."""
    T, _ = y.shape
    y_mean = jnp.mean(y, axis=0)
    L = jnp.tril(jnp.ones((T, T)))
    y = y - y_mean
    
    if max_lag >= T:
    
        raise ValueError("max_lag should be smaller than the number of time steps T to avoid excessive computation.")
    
    map_acf_frobenius = lambda k: jnp.linalg.norm(jnp.abs(jnp.einsum('i,ij,ik -> jk', L[T-k], y, jnp.roll(y, -k, axis=0)) / jnp.sum(L[T-k])), 'fro')
    
    return vmap(map_acf_frobenius)(jnp.arange(max_lag + 1))


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


def ratio(x, alpha, exp_matrix_gen, sample, sigma):

    exp_values = vmap(exp_matrix_gen, in_axes=(None, 0, None))(x, sample, sigma)
    ratio = alpha @ exp_values

    return ratio, exp_values


def exp_matrix_gen(x, y, sigma): 
    return jnp.exp(-(0.5 / sigma**2) * jnp.linalg.norm(x - y)**2)


def find_quantile(prev_sample, new_sample, sigma=1.0):
    '''Implements the method from the paper Adaptive Approximate Bayesian Computation
    Tolerance Selection by Simola et al. to find the quantile parameter for the
    '''

    n_samples = prev_sample.shape[0]
    
    exp_matrix = vmap(lambda x: vmap(exp_matrix_gen, in_axes=(None, 0, None))(x, prev_sample, sigma))(new_sample)
    exp_matrix_0 = vmap(lambda x: vmap(exp_matrix_gen, in_axes=(None, 0, None))(x, prev_sample, sigma))(prev_sample)
    e0 = jnp.sum(exp_matrix_0, axis=0) / n_samples

    def cond_fn(carry):
        _, err = carry
        return err > .1

    def step(carry):

        prev_alpha, err = carry
        b = 1 / (exp_matrix @ prev_alpha)
        new_alpha = prev_alpha * (exp_matrix.T @ b) / e0 / n_samples
        err = jnp.linalg.norm(new_alpha - prev_alpha)
        carry = new_alpha, err
    
        return carry

    alpha_star, _ = lax.while_loop(cond_fn, step, (jnp.ones(n_samples), 2.0))

    def _cond_fn(carry):
        _, err = carry
        return err > 0.01

    def _step(carry):

        prev_x, err = carry
        r, exp_values = ratio(prev_x, alpha_star, exp_matrix_gen, prev_sample, sigma)
        new_x = jnp.einsum('i,i,ij->j', alpha_star, exp_values, prev_sample) / r
        err = jnp.linalg.norm(new_x - prev_x)
        carry = new_x, err
    
        return carry

    i_star = jnp.argmax(alpha_star)
    x_star, _ = lax.while_loop(_cond_fn, _step, (prev_sample[i_star], 2.0))
    c, _ = ratio(x_star, alpha_star, exp_matrix_gen, prev_sample, sigma)
    q = 1 / c

    return q