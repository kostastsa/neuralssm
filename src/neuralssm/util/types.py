from typing import Optional, Union
from typing_extensions import Protocol
from jaxtyping import Array, Float
import jax.random as jr


PRNGKey = jr.PRNGKey

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float
