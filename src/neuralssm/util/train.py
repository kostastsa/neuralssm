import jax.numpy as jnp
import jax.random as jr
import numpy as onp
from jax import vmap, lax, jit
from jax.scipy.special import logsumexp as lse
from jax.tree_util import tree_map
from datasets.data_loaders import Data
from flax import nnx
from util.sample import sim_emissions
from util.param import to_train_array, log_prior, sample_prior
from util.numerics import kmeans
from functools import partial
import torch # type: ignore


def reshape_emissions(emissions, lag):
    
    '''
    Takes the emission array and returns an array of stacked emissions with lag L.
    Each element of the new array has L+1 emissions stacked (L for conditioning and 1 for predicting).
    '''
    
    num_timesteps, emission_dim = emissions.shape
    emissions = jnp.concatenate([jnp.zeros((lag, emission_dim)), emissions])
    lagged_emissions = []
    for t in range(num_timesteps):
        lagged_emissions.append(emissions[t:t+lag+1].flatten())
    lagged_emissions = jnp.stack(lagged_emissions)

    return lagged_emissions


def logdensity_fn(cond_params, model, emissions, props, lag):
    
    '''
    Computes the log density of the TSNL (lag>=0) and SNL (lag<0) models.
    '''

    if lag>=0:

        lagged_emissions = reshape_emissions(emissions, lag)
        tile_cond_params = jnp.tile(cond_params, (lagged_emissions.shape[0], 1))
        lp = -model.loss_fn(jnp.concatenate([tile_cond_params, lagged_emissions], axis=1))

    else:

        lp = -model.loss_fn(jnp.concatenate([cond_params[None], emissions.flatten()[None]], axis=1))

    # lp *= 0.5
    lp += log_prior(cond_params, props)

    return lp


def loglik_fn(cond_params, emissions, model, lag):
    
    '''
    Computes the loglik of the T-SNL (lag>=0) and SNL (lag<0) models.
    '''

    # ensure has batch dimension
    if len(cond_params.shape) == 1:

        cond_params = cond_params[None]

    if len(emissions.shape) == 2:

        emissions = emissions[None]

    # make equal size
    if cond_params.shape[0] < emissions.shape[0]:
    
        cond_params = jnp.tile(cond_params, (emissions.shape[0], 1))
    
    elif emissions.shape[0] < cond_params.shape[0]:
         
        emissions = jnp.tile(emissions, (cond_params.shape[0], 1, 1))

    if lag>=0:

        lps = vmap(tsnl_loglik_fn, in_axes=(0, 0, None, None))(cond_params, emissions, model, lag)

    else:
         
        lps = vmap(snl_loglik_fn, in_axes=(0, 0, None))(cond_params, emissions, model)

    return lps


def tsnl_loglik_fn(cond_params, emissions, model, lag):

    lagged_emissions = reshape_emissions(emissions, lag)
    tile_cond_params = jnp.tile(cond_params, (lagged_emissions.shape[0], 1))
    lp = -model.loss_fn(jnp.concatenate([tile_cond_params, lagged_emissions], axis=1))

    return tile_cond_params.shape[0] * lp


def snl_loglik_fn(cond_params, emissions, model):
     
    lp = -model.loss_fn(jnp.concatenate([cond_params[None], emissions.flatten()[None]], axis=1))

    return lp

def marg_loglik(key, props, observations, model, n_samples, lag):

    params = sample_prior(jr.PRNGKey(0), props, n_samples)
    tta = partial(to_train_array, props=props)
    cps = map(tta, params)
    cps = jnp.array(list(cps))

    return -jnp.log(n_samples) + lse(loglik_fn(cps, observations, model, lag))


@nnx.jit  
def train_step(model, optimizer, data):
    
    '''
    Training step for the MAF model.
    '''
    
    loss_fn = lambda model: model.loss_fn(data)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


def get_sds(key, logger, learner, num_samples, params_sample, num_timesteps):
    
    '''
    Returns the training dataset. 
    
    '''
    assert num_samples == len(params_sample), 'Number of samples must match the number of parameters.'
    
    if logger is not None:
        
        logger.write('------------getting emissions\n')
    
    keys = jr.split(key, num_samples)

    cond_params = jnp.array(tree_map(lambda params: to_train_array(params, learner.props), params_sample))

    for p in params_sample:
        
        p.from_unconstrained(learner.props)

    fn = partial(sim_emissions, ssm=learner.ssm, num_timesteps=num_timesteps)
    emissions = jnp.array(list(map(fn, keys, params_sample)))
            
    return cond_params, emissions


def lag_ds(straight_ds, lag, num_tiles):
        
        assert lag >= 0, 'Lag must be non-negative.'
        emissions = straight_ds[1]
        cond_params = straight_ds[0]
        lagged_emissions = vmap(reshape_emissions, in_axes=(0, None))(emissions, lag)
        cond_params_tiled = vmap(jnp.tile, in_axes=(0, None))(cond_params, (lagged_emissions.shape[1], 1))
        dataset = jnp.concatenate([cond_params_tiled, lagged_emissions], axis=2)

        return dataset

def subsample_fn(key, dataset, num_tiles):
        
        keys = jr.split(key, num_tiles+1)
        dataset = jr.choice(keys[-1], dataset.swapaxes(0,1), shape=(num_tiles,), replace=False)
        dataset = dataset.swapaxes(0,1)

        return dataset


def _get_data_loaders(dataset, batch_size):

        # Setup data loaders
        ntrain, nval = int(0.95 * dataset.shape[0]), int(0.05 * dataset.shape[0])
        train_data, val_data, test_data = dataset[:ntrain], dataset[ntrain:ntrain+nval], dataset[ntrain+nval:]
        data = Data(dataset.shape[1], train_data, val_data, test_data)

        train = torch.from_numpy(data.train)
        val = torch.from_numpy(data.val)
        test = torch.from_numpy(data.test)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)

        return train_loader, val_loader

import jax
import optax

def find_mle(key, model, emissions, lag, props):

    def loss_fn(cp):
        return -loglik_fn(cond_params=cp, model=model, emissions=emissions, lag=lag).squeeze()

    # Optimization step function
    @jit
    def step(x, opt_state):

        loss, grads = jax.value_and_grad(loss_fn)(x)
        updates, opt_state = optimizer.update(grads, opt_state, x)
        x = optax.apply_updates(x, updates)
        
        return x, opt_state, loss

    x = sample_prior(key, props, 1)[0]
    x = to_train_array(x, props)
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(x)

    for i in range(1000):

        x, opt_state, loss = step(x, opt_state)

    return x

