import jax
from jax import numpy as jnp, random as jr
import numpy as onp
from maf.density_models import MAF
from functools import partial
from util.train import get_sds, subsample_fn, lag_ds, _get_data_loaders, train_step, logdensity_fn
from util.param import  sample_prior
from util.sample import sample_logpdf
from util.numerics import compute_distances
from inference import lag_scoring
from inference import lag_selection as ls
from flax import nnx
import optax
import time
import gc


class SequentialNeuralLikelihood:
    """
    Trains a likelihood model using posterior MCMC sampling to guide simulations.
    """

    def __init__(self, props, ssm, lag, lag_opts=None):
        '''
        lag: integer >= 0 (T-SNL with a fixed lag), -1 (SNL), 'auto' (the lag is selected once from the
        round-0 simulations and fixed, Option A) or 'ada' (the lag is re-selected in every round with a
        network of fixed shape, Option B).
        lag_opts: for 'auto' and 'ada', a dict with 'tol' (TV tolerance eta), 'hp' (hyperparameters of the
        lag-dropout network, see lag_scoring.hp_from_model_desc) and optionally
          'ld_epochs_factor': epochs of the lag-dropout network as a multiple of num_epochs
                              (default lag_scoring.LD_EPOCHS_FACTOR);
        and, for 'ada' only (none of these is part of the descriptor, so none changes the storage path):
          'ld_epochs_factor_warm': the same factor for the rounds r >= 1 (default: the value above);
          'L_max': the cap of the network (default ada_lag_max(L*_0, L_cap), at most L_cap);
          'hold': keep the lag of the previous round if the selection of a round fails (default True).
        After learning, self.lag is the integer lag that was used (for 'ada', the lag of the last round),
        self.lag_selection holds the selection record, and self.lag_scores the held-out scores of every
        selection ({round: array (n, lags)}).
        '''

        self.props = props
        self.ssm = ssm
        self.lag = lag
        self.lag_opts = lag_opts or {}
        self.lag_selection = None
        self.lag_scores = {}
        self.xparam = None

    def train_model(self, model, optimizer, loaders, num_epochs, logger=None, mask=None):

        '''
        mask: None, or a function that turns a batch of full-lag rows into lag-dropout input (Option B,
        lag_scoring.LagDropoutMask). It is applied to the training and the validation batches.
        '''
        
        train_loader, val_loader = loaders
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):

            model.train()
            train_loss = []
            val_loss = []
            model.train()

            for batch in train_loader:

                batch = jnp.array(batch)
                if mask is not None:
                    batch = mask(batch)
                loss = train_step(model, optimizer, batch)
                train_loss.append(loss)

            model.eval()

            for batch in val_loader:

                batch = jnp.array(batch)
                if mask is not None:
                    batch = mask(batch)
                loss = model.loss_fn(batch)
                val_loss.append(loss)

            train_loss = jnp.mean(jnp.array(train_loss))
            val_loss = jnp.mean(jnp.array(val_loss))
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        logger.write(f"-------- training loss: {train_loss}, \n\t\t validation loss: {val_loss} \n")

        return model, (train_loss, val_loss)

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
                        sampler: str,
                        mcmc_steps: int,
                        batch_size: int = 128,
                        num_epochs: int = 20,
                        learning_rate: float = 1 * 1e-4,
                        logger = None,
                        num_tiles=None,
                        subsample=False,
                        do_kmeans=False,
                        model_factory=None
                        ):
        '''
        Sequentially samples parameters from the posterior using the TAF likelihood and MCMC.
        The output is the trained model after num_rounds and the final MCMC samples.
        The sample_and_train and logdensity_fn have to be set to taf or snl to distinguish between the two methods.

        With lag='auto' or lag='ada' the network cannot be built before the lag is known: pass model_factory,
        a function n_cond -> MAF (n_cond is the size of the conditioning vector), and model=None. The lag is
        selected after the round-0 simulations (select_lag_round0). Integer lags follow the original path,
        with the same random-number stream.

        'auto' (Option A): the model is built at the selected lag (n_cond = n_params + lag * d), trained on
        all round-0 trajectories, and the lag stays fixed.

        'ada' (Option B): the model is a lag-dropout network of fixed lag L_max = ada_lag_max(L*_0, L_cap)
        (n_cond = n_params + L_max + L_max * d). It trains in every round with a random lag for each row, and
        it keeps its parameters from round to round. Before a round r >= 1 trains on its new simulations,
        the network of the previous round scores them (they are held out by construction), and
        lag_scoring.reselect_lag gives the lag of round r. The MCMC of round r and the final posterior use
        this lag through a lag_scoring.LagMaskedModel. The model that is returned is such a view at the lag
        of the last round (its .base is the network).
        '''

        mode = self.lag

        if mode in ('auto', 'ada') and model_factory is None:

            raise ValueError(f"lag '{mode}' needs a model_factory (n_cond -> model)")

        ld, mask, rounds = None, None, []
        ld_factor = self.lag_opts.get('ld_epochs_factor', lag_scoring.LD_EPOCHS_FACTOR)
        ld_factor_warm = self.lag_opts.get('ld_epochs_factor_warm') or ld_factor

        # Sample initial parameters
        key, subkey = jr.split(key)
        params_sample = sample_prior(subkey, self.props, num_samples)
        key, subkey = jr.split(key)
        self.xparam = sample_prior(subkey, self.props, num_samples)[0]
        self.all_params = []
        train_losses = []
        val_losses = []

        for r in range(num_rounds):

            logger.write('----------------------------------\n')
            logger.write('Learning MAF likelihood, round {0}\n'.format(r + 1))
            logger.write('----------------------------------\n')
            logger.write('---------setting up datasets\n')

            # Sample emissions and create dataset
            t_round, n_comp = time.perf_counter(), lag_scoring.n_compilations()
            key, subkey = jr.split(key)
            new_sds = get_sds(subkey, logger, self, num_samples, params_sample, num_timesteps)
            cond_params, emissions = new_sds
            dists = compute_distances(emissions, observations, num_timesteps, self.ssm.emission_dim)
            log_round = {'round': r, 'seconds': {'simulate': time.perf_counter() - t_round},
                         'compilations': {'simulate': lag_scoring.n_compilations() - n_comp}}

            if r == 0:

                sds = new_sds
                all_emissions = emissions
                all_cond_params = cond_params
                all_dists = dists

            else:

                ref_data = (all_cond_params, all_emissions)    # the data that the network has seen
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
                    best_emissions = jnp.take(all_emissions, idx, axis=0)
                    best_cond_params = jnp.take(all_cond_params, idx, axis=0)
                    sds = (best_cond_params, best_emissions)

            t_phase, n_comp = time.perf_counter(), lag_scoring.n_compilations()

            if r == 0 and mode in ('auto', 'ada'):

                key, subkey = jr.split(key)
                self.lag_selection, self.lag_scores[0] = lag_scoring.select_lag_round0(
                    subkey, cond_params, emissions, num_timesteps,
                    hp=self.lag_opts['hp'], eta=self.lag_opts.get('tol', 0.1),
                    ld_epochs_factor=ld_factor, return_nll=True,
                    logger=lambda msg: logger.write(msg) if logger is not None else None)
                self.lag = self.lag_selection['L_star']
                n_params, emission_dim = cond_params.shape[1], emissions.shape[-1]

                if mode == 'auto':

                    model = model_factory(n_params + self.lag * emission_dim)

                else:

                    L_max = self.lag_opts.get('L_max') or ls.ada_lag_max(self.lag, self.lag_selection['L_cap'])
                    L_max = int(min(L_max, self.lag_selection['L_cap']))
                    self.lag = min(self.lag, L_max)
                    ld = {'L_max': L_max, 'n_params': n_params, 'd': emission_dim}
                    key, subkey = jr.split(key)
                    mask = lag_scoring.LagDropoutMask(subkey, n_params, L_max, emission_dim)
                    model = model_factory(n_params + L_max + L_max * emission_dim)
                    logger.write(f"lag selection (Option B): L*_0={self.lag}, the network has the lag L_max={L_max}\n")

                jax.clear_caches()
                gc.collect()

            elif r >= 1 and mode == 'ada':

                key, subkey = jr.split(key)
                rd, self.lag_scores[r] = lag_scoring.reselect_lag(
                    subkey, model, cond_params, emissions, ref_data[0], ref_data[1], ld['L_max'],
                    self.lag_selection['L_cap'], num_timesteps, eta=self.lag_opts.get('tol', 0.1),
                    L_prev=self.lag, hold=self.lag_opts.get('hold', True), return_nll=True)
                log_round.update(rd)
                logger.write(f"lag selection (Option B), round {r + 1}: L*_r={rd['L_star']} (L_tol={rd['L_tol']}, "
                             f"L_cv={rd['L_cv']}, fit {rd['fit_status']}), L_max={ld['L_max']}\n")

                if rd['held']:

                    logger.write(f"WARNING lag selection round {r + 1}: the selection failed (fit {rd['fit_status']}, "
                                 f"the held-out scores are not usable). Kept the lag {rd['L_star']} of the previous "
                                 f"round. The routine gave {rd['L_star_raw']}.\n")

                if rd['damaged_warning']:

                    logger.write(f"WARNING lag selection round {r + 1}: {rd['n_damaged']} of {rd['n_scored']} scored "
                                 f"trajectories ({100 * rd['damaged_fraction']:.0f} %) have a damaged score (a lag is "
                                 f"worse than lag 0 by more than {rd['damaged_nats']:g} nat per observation, or not "
                                 f"finite). The network does not describe them. The lag is not changed by this warning.\n")

                self.lag = rd['L_star']

                if rd['resolution_failure']:

                    logger.write(f"WARNING lag selection round {r + 1}: resolution failure. Stage 0 sees "
                                 f"{rd['n_prefix_stage0']} significant lags, stage 1 sees {rd['n_prefix']}. "
                                 f"L*_r is a lower bound.\n")

                if rd['grid_limited']:

                    logger.write(f"WARNING lag selection round {r + 1}: the cap L_max={ld['L_max']} limits the "
                                 f"result. The cap is fixed by 2 * L*_0 and has no setting yet.\n")

            if mode in ('auto', 'ada'):

                log_round['seconds']['select'] = time.perf_counter() - t_phase
                log_round['compilations']['select'] = lag_scoring.n_compilations() - n_comp

            if self.lag >= 0:

                dataset = lag_ds(sds, ld['L_max'] if mode == 'ada' else self.lag, num_tiles)

                if subsample: 
                    
                    key, subkey = jr.split(key)
                    dataset = subsample_fn(subkey, dataset, num_tiles)
                
            else:

                cond_params, emissions = sds
                dataset = jnp.concatenate([cond_params, emissions.reshape(emissions.shape[0], -1)], axis=1)

            logger.write('------------getting data loaders\n')

            fin_dataset = onp.array(dataset)
            fin_dataset = fin_dataset.reshape(-1, dataset.shape[-1])
            loaders = _get_data_loaders(fin_dataset, batch_size)

            logger.write('---------training model\n')

            t_phase, n_comp = time.perf_counter(), lag_scoring.n_compilations()
            optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=1e-4)) 

            if mode == 'ada':

                factor = ld_factor if r == 0 else ld_factor_warm
                model, losses = self.train_model(model, optimizer, loaders, max(1, int(round(factor * num_epochs))), logger, mask)

            else:

                model, losses = self.train_model(model, optimizer, loaders, num_epochs, logger)            

            train_loss, val_loss = losses
            log_round['seconds']['train'] = time.perf_counter() - t_phase
            log_round['compilations']['train'] = lag_scoring.n_compilations() - n_comp
            
            if jnp.isnan(train_loss) or jnp.isnan(val_loss):
                
                assert False, "Forced assertion failure : NaN loss"

            if train_loss > 1e5 or val_loss > 1e5:

                assert False, "Forced assertion failure : Loss too large"

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.write('---------sampling new parameters\n')

            # Sample new parameters using trained likelihood and MCMC
            t_phase, n_comp = time.perf_counter(), lag_scoring.n_compilations()
            eval_model = self._eval_model(model, ld)
            plogpdf = partial(logdensity_fn, model=eval_model, emissions=observations, props=self.props, lag=self.lag)
            key, subkey = jr.split(key)
            params_sample, ps = sample_logpdf(key=subkey, learner=self, logdensity_fn=plogpdf, prev_cps = sds[0], num_samples=num_samples, sampler=sampler, num_mcmc_steps=mcmc_steps)
            ps.block_until_ready()
            del ps
            self.all_params.append(params_sample)
            log_round['seconds']['mcmc'] = time.perf_counter() - t_phase
            log_round['compilations']['mcmc'] = lag_scoring.n_compilations() - n_comp
            log_round['lag'] = self.lag
            rounds.append(log_round)

            jax.clear_caches()
            gc.collect()

        self.all_dists = all_dists
        self.all_emissions = all_emissions
        self.all_cond_params = all_cond_params
        self.losses = (jnp.mean(jnp.array(train_losses)), jnp.mean(jnp.array(val_losses)))
    
        # Sample posterior
        eval_model = self._eval_model(model, ld)
        plogpdf = partial(logdensity_fn, model=eval_model, emissions=observations, props=self.props, lag=self.lag)
        key, subkey = jr.split(key)
        posterior_sample, posterior_cond_sample = sample_logpdf(key=subkey, learner=self, logdensity_fn=plogpdf, prev_cps=sds[0], num_samples=num_posterior_samples, sampler=sampler, num_mcmc_steps=int(2*num_posterior_samples))

        if self.lag_selection is not None:

            self._finish_selection_record(ld, rounds, ld_factor, ld_factor_warm)

        return  eval_model, (posterior_sample, posterior_cond_sample)

    def _eval_model(self, model, ld):

        '''
        The model that evaluates the likelihood at self.lag. For 'ada' this is the lag-dropout network seen
        as a T-SNL model of lag self.lag; otherwise the model itself.
        '''

        if ld is None:

            return model

        return lag_scoring.LagMaskedModel(model, self.lag, ld['L_max'], ld['n_params'], ld['d'])

    def _finish_selection_record(self, ld, rounds, ld_factor, ld_factor_warm):

        '''
        Adds the record of the rounds to self.lag_selection (written to lag_selection.json by the runner).
        For 'auto' the record has the time and the number of compilations of each phase. For 'ada' it also has
        the lag of every round (L_star_r), the cap L_max, and the flag capped.
        '''

        sel = self.lag_selection

        if ld is None:

            sel['rounds'] = rounds
            return

        rounds[0].update({k: sel[k] for k in ('L_star', 'L_tol', 'L_cv', 'L0', 'fit_status', 'n_prefix',
                                              'n_prefix_stage0', 'resolution_failure', 'damaged_fraction',
                                              'n_damaged', 'damaged_nats', 'damaged_warning')})
        rounds[0].update(grid_limited=sel['grid_capped'], at_L_max=bool(sel['L_star'] >= ld['L_max']),
                         L_star_raw=sel['L_star'], held=False, usable=True)
        rounds[0]['L_star'] = rounds[0]['lag']        # the lag in use (the cap L_max can clip L*_0)
        sel.update(option='B', L_max=ld['L_max'], n_params=ld['n_params'], emission_dim=ld['d'],
                   L_star_r=[int(rd['L_star']) for rd in rounds], L_final=int(self.lag),
                   L_star_raw_r=[int(rd['L_star_raw']) for rd in rounds],
                   held_rounds=[rd['round'] for rd in rounds if rd['held']],
                   damaged_rounds=[rd['round'] for rd in rounds if rd['damaged_warning']],
                   L_max_source='lag_opts' if self.lag_opts.get('L_max') else 'default',
                   ld_epochs_factor_warm=ld_factor_warm, hold=bool(self.lag_opts.get('hold', True)),
                   capped=any(rd['grid_limited'] for rd in rounds),
                   capped_rounds=[rd['round'] for rd in rounds if rd['grid_limited']],
                   resolution_failure_rounds=[rd['round'] for rd in rounds if rd['resolution_failure']],
                   ld_epochs_factor=ld_factor, rounds=lag_scoring._jsonable(rounds))