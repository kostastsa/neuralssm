# import jax.numpy as jnp
# from jax import random as jr
# from utils import reshape_emissions
# from parameters import jitter, to_train_array, log_prior, sample_ssm_params
# from parameters import to_train_array, get_unravel_fn, tree_from_params, join_trees, params_from_tree
# from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions, ParamsLGSSM, lgssm_smoother, lgssm_filter # type: ignore
# from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM # type: ignore

# def to_dynamax_params(params):

#     initial = ParamsLGSSMInitial(mean=params.initial.mean.value, 
#                                  cov=params.initial.cov.value)
#     dynamics = ParamsLGSSMDynamics(weights=params.dynamics.weights.value, 
#                                    bias=params.dynamics.bias.value, 
#                                    input_weights=params.dynamics.input_weights.value, 
#                                    cov=params.dynamics.cov.value)
#     emissions = ParamsLGSSMEmissions(weights=params.emissions.weights.value, 
#                                      bias=params.emissions.bias.value, 
#                                      input_weights=params.emissions.input_weights.value, 
#                                      cov=params.emissions.cov.value)

#     return ParamsLGSSM(initial, dynamics, emissions)

# def logdensity_fn(model, prior, cond_params, emissions, lag):

#     lagged_emissions = reshape_emissions(emissions, lag)
#     tile_cond_params = jnp.tile(cond_params, (lagged_emissions.shape[0], 1))
#     lp = -model.loss_fn(jnp.concatenate([tile_cond_params, lagged_emissions], axis=1))
#     lp += log_prior(cond_params, prior)

#     return lp

# def loglik(model, cond_params, emissions, lag):

#     lagged_emissions = reshape_emissions(emissions, lag)
#     tile_cond_params = jnp.tile(cond_params, (lagged_emissions.shape[0], 1))
#     ll = -model.loss_fn(jnp.concatenate([tile_cond_params, lagged_emissions], axis=1))

#     return ll

# def true_loglik(cond_params, emissions, state_dim, emission_dim, props, example_param, name_tree, is_constrained_tree):

#     unravel_fn = get_unravel_fn(example_param, props)
#     unravel = unravel_fn(cond_params)
#     tree = tree_from_params(example_param)
#     new_tree = join_trees(unravel, tree, props)
#     params = params_from_tree(new_tree, name_tree, is_constrained_tree)
#     params.from_unconstrained(props)
#     dyn_params = to_dynamax_params(params)
#     dyn_lgssm = LinearGaussianSSM(state_dim, emission_dim)
#     posterior_filtered = dyn_lgssm.filter(dyn_params, emissions)

#     return posterior_filtered.marginal_loglik

# def smc_sampler(key, model, props, prior, loglik, emissions, lag, num_samples, jittering=0.01):

#     key, subkey = jr.split(key)
#     proposed_params = sample_ssm_params(subkey, prior, num_samples)
#     param_value_list = []
#     log_weights = []
#     for params in proposed_params:
#         param_value = to_train_array(params, props)
#         param_value_list.append(param_value)
#         log_weights.append(loglik(model, param_value, emissions, lag)) 
#     log_weights = jnp.array(log_weights)
#     log_weights -= jnp.max(log_weights)
#     weights = jnp.exp(log_weights)
#     weights /= jnp.sum(weights)

#     # Resample
#     key, subkey = jr.split(key)
#     resampled_idx = jr.choice(subkey, jnp.arange(weights.shape[0]), shape=(len(proposed_params),), p=weights)
#     resampled_params = []
#     for idx in resampled_idx:
#         key, subkey = jr.split(key)
#         # jitter 
#         jitter(subkey, proposed_params[idx], props, jittering)
#         resampled_params.append(proposed_params[idx])    

#     return resampled_params