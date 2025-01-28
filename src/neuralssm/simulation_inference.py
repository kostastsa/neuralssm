from jax import numpy as jnp
import jax
import numpy as onp
from jax import random as jr
from jax import vmap
from jax.tree_util import tree_map
from utils import reshape_emissions, map_sims
from parameters import  ParamSSM, to_train_array, log_prior, sample_ssm_params
from density_models import MAF
from ssm import SSM
from datasets.data_loaders import Data, get_data_loaders 
from flax import nnx
import optax
import blackjax
from parameters import get_unravel_fn, tree_from_params, join_trees, params_from_tree
from functools import partial

def inference_loop(rng_key, kernel, initial_state, num_samples):
    '''
    Runs the MCMC kernel for num_samples steps and returns the states.
    '''
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

def logdensity_fn(cond_params, model, emissions, prior, lag):
    '''
    Computes the log density of the TAF (lag>0) and SNL (lag=0) models.
    '''
    if lag>0:
        lagged_emissions = reshape_emissions(emissions, lag)
        tile_cond_params = jnp.tile(cond_params, (lagged_emissions.shape[0], 1))
        lp = -model.loss_fn(jnp.concatenate([tile_cond_params, lagged_emissions], axis=1))
    else:
        lp = -model.loss_fn(jnp.concatenate([cond_params[None], emissions.flatten()[None]], axis=1))
    lp += log_prior(cond_params, prior)
    return lp

@nnx.jit  
def train_step(model, optimizer, data):
    '''
    Training step for the MAF model.
    '''
    loss_fn = lambda model: model.loss_fn(data)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # inplace updates
    return loss

def sample_and_train(key,
                     model: MAF,
                     ssmodel: SSM,
                     lag: int,
                     num_timesteps: int, 
                     props: ParamSSM,
                     params_sample: list,
                     example_param,
                     prev_dataset: onp.array = onp.array([]),
                     batch_size: int = 128,
                     num_epochs: int = 20,
                     learning_rate: float = 1 * 1e-4,
                     verbose = False):
    '''
    Takes a list of parameters in params_sample (usually output of sample_ssm_params) and samples a SSM model for each parameter to construct the 
    new dataset. Then it trains the MAF model on the union of prev_dataset and the new dataset. It returns the trained model and the union dataset. 
    If lag=0 it uses the direct SNL model, otherwise for l>0 the TAF (truncated model ).
    '''
    # Sample emissions and create dataset
    if verbose:
        print("----Creating dataset")

    num_samples = len(params_sample)
    keys = jr.split(key, num_samples)

    all_cond_params = jnp.array(tree_map(lambda params: to_train_array(params, props), params_sample))
    all_emissions = vmap(map_sims, in_axes=(0,0,None,None,None,None))(keys, all_cond_params, props, example_param, ssmodel, num_timesteps)
    if lag > 0:
        all_cond_params_tiled = vmap(jnp.tile, in_axes=(0, None))(all_cond_params, (num_timesteps, 1))
        all_lagged_emissions = vmap(reshape_emissions, in_axes=(0, None))(all_emissions, lag)
        dataset = jnp.concatenate([all_cond_params_tiled, all_lagged_emissions], axis=2)
    else:
        all_emissions = all_emissions.reshape(num_samples, -1)
        dataset = jnp.concatenate([all_cond_params, all_emissions], axis=1)

    dataset = onp.array(dataset.reshape(-1, dataset.shape[-1]))
    if prev_dataset.shape == (0,):
        prev_dataset = onp.empty((0, dataset.shape[1]))
    new_dataset = onp.concatenate([prev_dataset, dataset], axis=0)

    # Setup data loaders
    if verbose:
        print("----Setting up data loaders")
    ntrain, nval = int(0.95 * new_dataset.shape[0]), int(0.05 * new_dataset.shape[0])
    train_data, val_data, test_data = new_dataset[:ntrain], new_dataset[ntrain:ntrain+nval], new_dataset[ntrain+nval:]
    data = Data(new_dataset.shape[1], train_data, val_data, test_data)
    train_loader, val_loader, _  = get_data_loaders(data, batch_size)

    # Set up trainer
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=1e-6)) 

    # TRAINING LOOP
    if verbose:
        print("----Start training loop")
    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        model.train()
        train_loss = []
        val_loss = []
        model.train()
        for batch in train_loader:
            batch = jnp.array(batch)
            loss = train_step(model, optimizer, batch)
            train_loss.append(loss)
        model.eval()
        for batch in val_loader:
            batch = jnp.array(batch)
            loss = model.loss_fn(batch)
            val_loss.append(loss)
        train_loss = jnp.mean(jnp.array(train_loss))
        val_loss = jnp.mean(jnp.array(val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if verbose: 
            print(f"--------Epoch {i}, training loss: {train_loss}, validation loss: {val_loss}")

    return model, new_dataset

def sequential_posterior_sampling(
                    key : jr.PRNGKey,
                    model: MAF,
                    ssmodel: SSM,
                    lag: int,
                    num_rounds : int,
                    num_timesteps: int,
                    num_samples: int,
                    num_mcmc_steps: int,
                    emissions,
                    prior,
                    props,
                    example_param,
                    param_names,
                    is_constrained_tree,
                    rw_sigma=0.1
                    ):
    '''
    Sequentially samples parameters from the posterior using the TAF likelihood and MCMC.
    The output is the trained model after num_rounds rounds and the final MCMC samples.
    The sample_and_train and logdensity_fn have to be set to taf or snl to distinguish between the two methods.
    '''
    # Sample initial parameters
    key, subkey = jr.split(key)
    params_sample = sample_ssm_params(subkey, prior, num_samples) # Here, output params are in mixed constrained/unconstrained form
                                                                # The trainable params (given in prior by dist) are unconstrained
                                                                # whereas the not-trainable params (given in prior by arrays) are constrained
                                                                # In the trainer, the cond_params are appended to the dataset and then the params are converted
                                                                # to constrained form before being passed to the model for simulation. 
    dataset = jnp.array([])
    for r in range(num_rounds):
        print(f"-------- Round {r}")
        print("* Training")
        # Sample SSM emissions and train model
        key, subkey = jr.split(key)
        model, dataset = sample_and_train(
            key = subkey,
            model = model,
            ssmodel = ssmodel,
            params_sample = params_sample,
            example_param = example_param, 
            prev_dataset = dataset, 
            lag = lag,
            num_timesteps = num_timesteps, 
            props = props,
            num_epochs = 20,
            learning_rate = 1 * 1e-4,
            verbose=False
        )

        # Sample new parameters using trained likelihood and MCMC
        print("* Sampling new parameters")

        ## Pin logdensity function to the current model and emissions   
        pin_logdensity_fn = partial(logdensity_fn, model=model, emissions=emissions, prior=prior, lag=lag)

        ## Initialize MCMC chain and kernel
        key, subkey = jr.split(key)
        initial_cond_params = to_train_array(sample_ssm_params(subkey, prior, 1)[0], props)
        taf_random_walk = blackjax.additive_step_random_walk(pin_logdensity_fn, blackjax.mcmc.random_walk.normal(rw_sigma))
        taf_initial_state = taf_random_walk.init(initial_cond_params)
        taf_kernel = jax.jit(taf_random_walk.step)

        ## Run MCMC inference loop
        key, subkey = jax.random.split(key)
        taf_mcmc_states = inference_loop(subkey, taf_kernel, taf_initial_state, num_mcmc_steps)
        positions = taf_mcmc_states.position[-num_samples:]
        params_sample = []
        print("* Adding new params")
        for cond_param in positions:
            unravel_fn = get_unravel_fn(example_param, props)
            unravel = unravel_fn(cond_param)
            tree = tree_from_params(example_param)
            new_tree = join_trees(unravel, tree, props)
            param = params_from_tree(new_tree, param_names, is_constrained_tree)
            params_sample.append(param)

    return model, positions