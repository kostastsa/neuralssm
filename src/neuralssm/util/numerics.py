import jax.numpy as jnp
from jax import vmap, lax, random as jr
import jax.scipy.stats as jss
from jax.scipy.special import logsumexp as lse
import jax.numpy as jnp
from jax.scipy.stats import norm, mode
import jax


def kde_error(particles, true_cps, sigma=None):
        
    if sigma is None:

        n = particles.shape[0]
        d = particles.shape[1]
        std_dev = jnp.std(particles, axis=0)
        sigma = 2.06 * jnp.min(std_dev) * n ** (-1 / (d + 4))

    n = particles.shape[0]
    d = particles.shape[1]
    dps = particles - true_cps
    ds = - jnp.linalg.norm(dps, axis=1)**2 / 2 / sigma**2
    logp = lse(ds) - jnp.log(n) - (d/2) * jnp.log(2 * jnp.pi * sigma ** 2)

    return -logp


def min_error(particles, true_cps, sigma=1.0):

    num_prt = particles.shape[0]
    dim = particles.shape[1]
    dps = particles - true_cps
    ns = jnp.linalg.norm(dps, axis=1)
    
    return jnp.min(ns)


def bias(particles, true_cps):

    return jnp.linalg.norm(jnp.mean(particles, axis=0) - true_cps)


def angular_bias(particles, true_cps):

    return jnp.dot(jnp.mean(particles, axis=0) - true_cps, true_cps) / (jnp.linalg.norm(jnp.mean(particles, axis=0) - true_cps) * jnp.linalg.norm(true_cps))


def sdev(particles):

    mean = jnp.mean(particles, axis=0)
    cov = jnp.einsum('ij, ik -> jk', particles-mean, particles-mean) / particles.shape[0]
    std = jnp.trace(cov)

    return std


def rmse(particles, true_cps):

    b = bias(particles, true_cps)
    s = sdev(particles)

    return jnp.sqrt(b**2 + s**2)


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


def gmm_em_1d_stable(x, num_iter=100, tol=1e-6, var_floor=1e-6, weight_floor=1e-3):

    n = x.shape[0]

    # Initialize means, variances, and weights
    means = jnp.array([x.min(), x.max()])
    variances = jnp.ones(2)
    weights = jnp.array([0.5, 0.5])

    def e_step(x, means, variances, weights):

        probs = jnp.stack([
            weights[k] * norm.pdf(x, loc=means[k], scale=jnp.sqrt(variances[k]))
            for k in range(2)
        ])
        responsibilities = probs / probs.sum(axis=0)

        return responsibilities

    def m_step(x, responsibilities):

        N_k = responsibilities.sum(axis=1)
        N_k = jnp.maximum(N_k, 1e-3)  # avoid divide-by-zero

        means = (responsibilities @ x) / N_k
        variances = ((responsibilities * (x - means[:, None])**2).sum(axis=1)) / N_k
        variances = jnp.maximum(variances, var_floor)  # apply variance floor

        weights = N_k / x.shape[0]
        weights = jnp.maximum(weights, weight_floor)
        weights /= weights.sum()  # re-normalize

        return means, variances, weights

    def step_fn(carry, _):

        means, variances, weights, _ = carry
        resp = e_step(x, means, variances, weights)
        new_means, new_vars, new_weights = m_step(x, resp)

        return (new_means, new_vars, new_weights, resp), None

    (final_means, final_vars, final_weights, resp), _ = jax.lax.scan(step_fn, (means, variances, weights, jnp.ones((2,n))), None, length=num_iter)
    ids = jnp.argsort(final_means)
    final_means = final_means[ids]
    final_vars = final_vars[ids]
    final_weights = final_weights[ids]

    return final_means, final_vars, final_weights, resp


def get_good_ids(errors, entropy_thresh=0.1):

    gmm_means, gmm_vars, _, resp = gmm_em_1d_stable(errors)

    if jnp.isnan(gmm_means).any() or jnp.isinf(gmm_means).any():

        z = (errors - errors.mean()) / errors.std()
        good_ids = z <= mode(z).mode + 1e-3
        good_ids = jnp.arange(errors.shape[0])[good_ids]

    else:

        # resps: shape (2, N)
        eps = 1e-10  # for numerical stability
        log_resps = jnp.log(resp + eps)
        entropies = -jnp.sum(resp * log_resps, axis=0)  # shape (N,)
        mean_entropy = jnp.mean(entropies)

        if mean_entropy < entropy_thresh:
            # Case A: hard assignments, select cluster 0 only
            good_ids = jnp.where(resp[0] > 1 - 1e-2)[0]
            good_ids = jnp.arange(errors.shape[0])[good_ids]

        else:
            # Case B: soft assignments, keep all
            good_ids = jnp.arange(errors.shape[0])

    return good_ids


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


def average_min_distance(particles):
    '''
    Distance between each particle and its nearest neighbor.
    This is the average of the minimum distances between each particle and its nearest neighbor.
    '''

    particles = jnp.unique(particles, axis=0)
    ds = vmap(lambda y: vmap(lambda x: jnp.linalg.norm(x-y))(particles))(particles)
    mod_ds = ds + jnp.diag(jnp.full((ds.shape[0],), jnp.inf))
    min_ds = jnp.min(mod_ds, axis=1)
    avg_min_distance = jnp.mean(min_ds)

    return min_ds, avg_min_distance


def average_pair_distance(particles):

    particles = jnp.unique(particles, axis=0)
    ds = vmap(lambda y: vmap(lambda x: jnp.linalg.norm(x-y))(particles))(particles)
    mod_ds = jnp.triu(ds)[jnp.triu(ds)>0]
    mod_ds = mod_ds.flatten()

    return mod_ds, jnp.mean(mod_ds)


def average_max_distance(particles):

    particles = jnp.unique(particles, axis=0)
    ds = vmap(lambda y: vmap(lambda x: jnp.linalg.norm(x-y))(particles))(particles)
    max_ds = jnp.max(ds, axis=1)

    return max_ds, jnp.mean(max_ds)


def compute_all_errors(cps, true_cps):

    kderr = kde_error(cps, true_cps)
    minerr = min_error(cps, true_cps)
    b = bias(cps, true_cps)
    ab = angular_bias(cps, true_cps)
    s = sdev(cps)
    
    mds, amd = average_min_distance(cps)
    ads, apd = average_pair_distance(cps)
    maxds, amxd = average_max_distance(cps)

    return kderr, minerr, b, ab, s, (mds, amd), (ads, apd), (maxds, amxd)

