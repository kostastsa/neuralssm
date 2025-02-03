import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jax import jit
from typing import Union, Tuple, List, NamedTuple
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from functools import reduce, partial
from dynamax.types import PRNGKey # type: ignore
from dynamax.parameters import ParameterProperties # type: ignore

#################################### NEW CLASS ###############################

class ParamNode():
        
        def __init__(self, 
                    value: jnp.ndarray,
                    is_constrained: bool=False):
            
            self.value = value
            self.is_constrained = is_constrained

        def __str__(self):
            return f"value={self.value}, is_constrained={self.is_constrained}"


class ParamField():

    def __init__(self, 
                 values_list: List[jnp.ndarray],
                 param_names: List[str], 
                 is_constrained_list: List[bool]):

        for i, value in enumerate(values_list):
            setattr(self, param_names[i], ParamNode(value, is_constrained_list[i]))

    def __str__(self):
        str = ''
        for key in self.__dict__:
            str += f"{key}={getattr(self, key).__class__.__name__}({self.__dict__[key]}), "
        return str[:-2]
    

class Field():

    def __init__(self, 
                 values_list: List[jnp.ndarray],
                 param_names: List[str]=None):

        for i, value in enumerate(values_list):
            if param_names is not None:
                setattr(self, param_names[i], value)
            else:
                setattr(self, f"param_{i}", value)

    def __str__(self):
        str = ''
        for key in self.__dict__:
            str += f"{key}={getattr(self, key).__class__.__name__}({self.__dict__[key]}), "
        return str[:-2]


class ParamSSM():

    def __init__(self, 
                 initial: ParamField, 
                 dynamics: ParamField, 
                 emissions: ParamField):

        self.initial = initial
        self.dynamics = dynamics
        self.emissions = emissions

    def __str__(self):
        str = f'{self.__class__.__name__}('
        for key in ['initial', 'dynamics', 'emissions']:
            str += f"{key}={getattr(self, key).__class__.__name__}({self.__dict__[key]}), "
        return str[:-2]

    def to_unconstrained(self, props):
        for field_name in ['initial', 'dynamics', 'emissions']:
            field = getattr(self, field_name)
            props_field = getattr(props, field_name)
            for subfield_name in field.__dict__:
                subfield = getattr(field, subfield_name)
                props_subfield = getattr(props_field, subfield_name)
                if subfield.is_constrained:
                    constrainer = props_subfield.constrainer
                    setattr(subfield, 'is_constrained', False)
                    if constrainer is not None:
                        setattr(subfield, 'value', constrainer().inverse(subfield.value))

    def from_unconstrained(self, props):
        for field_name in ['initial', 'dynamics', 'emissions']:
            field = getattr(self, field_name)
            props_field = getattr(props, field_name)
            for subfield_name in field.__dict__:
                subfield = getattr(field, subfield_name)
                props_subfield = getattr(props_field, subfield_name)
                if not subfield.is_constrained:
                    constrainer = props_subfield.constrainer
                    setattr(subfield, 'is_constrained', True)
                    if constrainer is not None:
                        setattr(subfield, 'value', constrainer()(subfield.value))

    def _get_names(self):
        names_tree = []
        for field_name in ['initial', 'dynamics', 'emissions']:
            field = getattr(self, field_name)
            names_list = []
            for subfield_name in field.__dict__:
                names_list.append(subfield_name)
            names_tree.append(names_list)
        return names_tree
    
    def _get_is_constrained(self):
        is_constrained_tree = []
        for field_name in ['initial', 'dynamics', 'emissions']:
            field = getattr(self, field_name)
            is_constrained_list = []
            for subfield_name in field.__dict__:
                is_constrained_list.append(getattr(getattr(field, subfield_name), 'is_constrained'))
            is_constrained_tree.append(is_constrained_list)
        return is_constrained_tree


class ParamMeta(NamedTuple):
    props: ParamSSM
    prior: ParamSSM


def get_unravel_fn(params, props):
    list_trainable_params = []
    for field_name in ['initial', 'dynamics', 'emissions']:
        field = getattr(params, field_name)
        props_field = getattr(props, field_name)
        sublist_trainable_params = []
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            props_subfield = getattr(props_field, subfield_name)
            if props_subfield.trainable:
                sublist_trainable_params.append(subfield.value)
            else:
                sublist_trainable_params.append(jnp.array([]))
        list_trainable_params.append(sublist_trainable_params)
    _, unravel_fn = ravel_pytree(list_trainable_params)
    return unravel_fn

def to_train_array(params, props):
    '''
    Convert the parameters to a flat array of trainable parameters in unconstrained form.

    :param params: ParamSSM object with trainable parameters in unconstrained form
    :param props: properties of the paramters

    :return: flat array of concatenated trainable parameters in unconstrained form. If all parameters are
             in constrained form, returns an empty array.
    '''
    list_trainable_params = []
    for field_name in ['initial', 'dynamics', 'emissions']:
        field = getattr(params, field_name)
        props_field = getattr(props, field_name)
        sublist_trainable_params = []
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            props_subfield = getattr(props_field, subfield_name)
            if props_subfield.trainable:
                if subfield.is_constrained: 
                    # make sure that the parameter to train is unconstrained. 
                    # this will produce error downstream if not done
                    sublist_trainable_params.append(jnp.array([]))
                else:
                    sublist_trainable_params.append(subfield.value)
            else:
                sublist_trainable_params.append(jnp.array([]))
        list_trainable_params.append(sublist_trainable_params)
    train_array, _ = ravel_pytree(list_trainable_params)
    return train_array

def tree_from_params(params):
    tree = []
    for field_name in ['initial', 'dynamics', 'emissions']:
        field = getattr(params, field_name)
        subtree = []
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            subtree.append(subfield.value)
        tree.append(subtree)
    return tree

def params_from_tree(tree, names_tree, is_constrained_tree):
    initial = ParamField(tree[0], names_tree[0], is_constrained_tree[0])
    dynamics = ParamField(tree[1], names_tree[1], is_constrained_tree[1])
    emissions = ParamField(tree[2], names_tree[2], is_constrained_tree[2])
    return ParamSSM(initial, dynamics, emissions)

def join_trees(train_tree, untrain_tree, props):
    r"""Join two trees of parameters, one with trainable parameters and the other with untrainable parameters.
    The trainable parameter tree is the output of 
    """
    new_tree = []
    for i, field_name in enumerate(['initial', 'dynamics', 'emissions']):
        props_field = getattr(props, field_name)
        new_subtree = []
        for j, subfield_name in enumerate(props_field.__dict__):
            props_subfield = getattr(props_field, subfield_name)
            if props_subfield.trainable:
                new_subtree.append(train_tree[i][j])
            else:
                new_subtree.append(untrain_tree[i][j])
        new_tree.append(new_subtree)
    return new_tree

def sample_ssm_params(
    key: PRNGKey, 
    prior: ParamSSM, 
    num_samples: int):
    r"""Sample parameters from the prior distribution. When prior field is tfd.Distribution,
        sample num_samples values from it, and set the corresponding values of is_constrained to False
        (we consider that parameters are sampled always in unconstrained form). Otherwise set the value 
        equal to the provided value and is_constrained to True.
    """
    tree = []
    param_names = []
    is_constrained_tree = []
    key, subkey = jr.split(key)
    for field_name in ['initial', 'dynamics', 'emissions']:
        field = getattr(prior, field_name)
        subtree = []
        param_subnames = []
        is_constrained_subtree = []
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            if isinstance(subfield, tfd.Distribution):
                value = subfield.sample(num_samples, subkey)
                is_constrained_subtree.append(False)
            else:
                value = jnp.tile(subfield, (num_samples, *tuple(1 for _ in range(subfield.ndim))))
                is_constrained_subtree.append(True)
            subtree.append(value)
            param_subnames.append(subfield_name)
        is_constrained_tree.append(is_constrained_subtree)
        param_names.append(param_subnames)
        tree.append(subtree)
        
    return list(map(lambda i: params_from_tree(tree_map(lambda x: x[i], tree), param_names, is_constrained_tree), range(num_samples)))

def initialize(
        prior_fields: List[List[Union[jnp.ndarray, tfd.Distribution]]],
        param_names: List[List[str]],
        constrainers: List[List[Union[None, tfb.Bijector]]] = None
) -> Tuple[ParamSSM, ParamSSM]:
    r"""Initialize model parameters that are set to None, and their corresponding properties.

    Args:
        All arguments can either be Arrays or tfd.Distributions. The prior defaults to delta distributions if
        Arrays are provided, otherwise the prior is set to the provided distribution. Setting parameters to None
        results in delta priors at the default values defined in this function.

    Returns:

    """
    is_trainable = tree_map(lambda field: isinstance(field, tfd.Distribution), prior_fields, is_leaf=lambda x: not isinstance(x, list))
    properties_tree = tree_map(lambda is_trainable, constrainer: ParameterProperties(is_trainable, constrainer), is_trainable, constrainers)

    prior = ParamSSM(
        initial = Field(
        prior_fields[0],
        param_names[0]),
        dynamics = Field(
            prior_fields[1],
            param_names[1]),
        emissions = Field(
            prior_fields[2],
            param_names[2])
            )

    props = ParamSSM(
        initial = Field(
        properties_tree[0],
        param_names[0]),
        dynamics = Field(
            properties_tree[1],
            param_names[1]),
        emissions = Field(
            properties_tree[2],
            param_names[2])
            )

    return props, prior

@partial(jit, static_argnums=(1,))
def log_prior(cond_params, prior):
    r"""Compute the log prior of the parameters.

    :param params: parameters
    :param prior: prior distribution

    :return: log prior

    """
    log_prior = 0
    idx = 0
    for field_name in prior.__dict__:
        field = getattr(prior, field_name)
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            if isinstance(subfield, tfd.Distribution):
                shape = tuple(subfield.event_shape)
                flat_shape = reduce(lambda x, y: x*y, shape)
                value = cond_params[idx:idx+flat_shape]
                log_prior += subfield.log_prob(value)
                idx += flat_shape
    return log_prior

def jitter(key, params, props, jitter_scale=1e-2):
    r"""Compute the log prior of the parameters.

    :param params: parameters
    :param prior: prior distribution

    :return: log prior

    """
    for field_name in params.__dict__:
        field = getattr(params, field_name)
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            subfield_props = getattr(getattr(props, field_name), subfield_name)
            if subfield_props.trainable:
                key, subkey = jr.split(key)
                shape = tuple(subfield.value.shape)
                noise = jitter_scale * jr.normal(subkey, shape)
                subfield.value += noise
