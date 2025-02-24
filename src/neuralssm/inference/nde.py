from jax import numpy as jnp, random as jr, vmap
import numpy as onp
from density_models import MAF
from functools import partial
from util.train import get_sds, subsample_fn, lag_ds, _get_data_loaders, train_step, logdensity_fn
from util.param import  sample_prior
from util.sample import sample_logpdf
from util.misc import compute_distances
from flax import nnx
import optax
import time


class SequentialNeuralLikelihood:
    """
    Trains a likelihood model using posterior MCMC sampling to guide simulations.
    """

    def __init__(self, props, ssm, lag):

        self.props = props
        self.ssm = ssm
        self.lag = lag
        self.xparam = None

    def train_model(self, model, optimizer, loaders, num_epochs):
        
        train_loader, val_loader = loaders
        train_losses = []
        val_losses = []
        for _ in range(num_epochs):
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

        return model


    def learn_likelihood(
                        self,
                        key : jr.PRNGKey,
                        observations,
                        model: MAF,
                        num_rounds : int,
                        num_timesteps: int,
                        num_samples: int,
                        num_posterior_samples: int,
                        train_on: str,
                        mcmc_steps: int,
                        batch_size: int = 128,
                        num_epochs: int = 20,
                        learning_rate: float = 1 * 1e-4,
                        rw_sigma=0.1,
                        logger = None,
                        num_tiles=None,
                        subsample=False,
                        do_kmeans=False
                        ):
        '''
        Sequentially samples parameters from the posterior using the TAF likelihood and MCMC.
        The output is the trained model after num_rounds and the final MCMC samples.
        The sample_and_train and logdensity_fn have to be set to taf or snl to distinguish between the two methods.
        '''
        # Sample initial parameters
        key, subkey = jr.split(key)
        params_sample = sample_prior(subkey, self.props, num_samples)
        self.xparam = params_sample[0]    
        # self.all_emissions = []
        # self.all_cond_params = []
        self.all_params = []
        self.time_all_rounds = []

        for r in range(num_rounds):
            
            tin = time.time()
            logger.write('----------------------------------\n')
            logger.write('Learning MAF likelihood, round {0}\n'.format(r + 1))
            logger.write('----------------------------------\n')
            logger.write('---------seting up datasets\n')

            # Sample emissions and create dataset
            key, subkey = jr.split(key)
            new_sds = get_sds(subkey, self, num_samples, params_sample, num_timesteps)
            cond_params, emissions = new_sds   
            dists = compute_distances(emissions, observations, num_timesteps, self.ssm.emission_dim)

            if r == 0:
                
                sds = new_sds
                all_emissions = emissions
                all_cond_params = cond_params
                all_dists = dists

            else:

                all_emissions = jnp.concatenate([all_emissions, emissions], axis=0)
                all_cond_params = jnp.concatenate([all_cond_params, cond_params], axis=0)
                all_dists = jnp.concatenate([all_dists, dists], axis=0)

                if train_on == 'last':
                    
                    sds = new_sds

                elif train_on == 'all':
                    
                    sds = (all_cond_params, all_emissions)

                elif train_on == 'best':
                    
                    weights = jnp.exp(-all_dists)
                    weights /= jnp.sum(weights)
                    key, subkey = jr.split(key)
                    idx = jr.choice(subkey, jnp.arange(all_emissions.shape[0]), shape=(num_samples,), p=weights, replace=False)
                    # idx = jnp.argsort(weights)[-num_samples:]
                    best_emissions = jnp.take(all_emissions, idx, axis=0)
                    best_cond_params = jnp.take(all_cond_params, idx, axis=0)
                    sds = (best_cond_params, best_emissions)

            if self.lag >= 0:

                dataset = lag_ds(sds, self.lag, num_tiles)

                if subsample: 
                    
                    key, subkey = jr.split(key)
                    dataset = subsample_fn(subkey, dataset, num_tiles)
                
            else:
                cond_params, emissions = sds
                dataset = jnp.concatenate([cond_params, emissions.reshape(num_samples, -1)], axis=1)

            fin_dataset = onp.array(dataset.reshape(-1, dataset.shape[-1]))
            loaders = _get_data_loaders(fin_dataset, batch_size)
            logger.write('---------training model\n')
            optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=1e-6)) 
            model = self.train_model(model, optimizer, loaders, num_epochs)
            logger.write('---------sampling new parameters\n')

            # Sample new parameters using trained likelihood and MCMC
            plogpdf = partial(logdensity_fn, model=model, emissions=observations, props=self.props, lag=self.lag)
            key, subkey = jr.split(key)
            params_sample, _ = sample_logpdf(key=subkey, learner=self, logdensity_fn=plogpdf, num_samples=num_samples, num_mcmc_steps=mcmc_steps, rw_sigma=rw_sigma)
            self.all_params.append(params_sample)

            tout = time.time()
            self.time_all_rounds.append(tout-tin)
            logger.write('---------time: {:.2f}\n'.format(tout-tin))

        self.all_dists = all_dists
        self.all_emissions = all_emissions
        self.all_cond_params = all_cond_params

        # Sample posterior
        plogpdf = partial(logdensity_fn, model=model, emissions=observations, props=self.props, lag=self.lag)
        key, subkey = jr.split(key)
        posterior_sample, posterior_cond_sample = sample_logpdf(key=subkey, learner=self, logdensity_fn=plogpdf, num_samples=num_posterior_samples, num_mcmc_steps=int(2*num_posterior_samples), rw_sigma=rw_sigma)

        return  model, (posterior_sample, posterior_cond_sample)