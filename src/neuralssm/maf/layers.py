from flax import nnx
from jax import numpy as jnp
from flax import linen as nn
import jax.random as jr
import jax


def relu(x):
    return jnp.maximum(0, x)


activations = {
    'relu': relu,
    'tanh': jnp.tanh,
    'sigmoid': jax.nn.sigmoid,
}


class MaskInitializer:

    def __init__(self, mask):

        self.mask = mask  # Store mask

    def __call__(self):

        return self.mask  # Return mask when called


class MaskedLinear(nnx.Module):

  def __init__(self, mask, din: int, dout: int, *, rngs: nnx.Rngs):

    key = rngs.params()
    std = jnp.sqrt(din)
    self.w = nnx.Param(jr.uniform(key, (din, dout), minval=-1/std, maxval=1/std))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout
    self.mask = nn.Variable(None, "constants", "mask", MaskInitializer(mask))

  def __call__(self, x: jax.Array):

    w = jnp.asarray(self.w)  # Convert self.w to a standard JAX array
    b = jnp.asarray(self.b)

    return x @ jnp.multiply(self.mask.unbox(), w)+ b


class BatchNormLayer(nnx.Module):
    """
    Difference is in handling the conditional input. This computes statistics for the conditional input as well.
    At the end it concatenates the conditional input with the transformed input.

    The conditioning columns are copied unchanged, so the Jacobian of the layer involves the data columns
    only: log_det = sum_{j > dcond} (gamma_j - 0.5 log v_j). Before 2026-10-20 the sum ran over all columns
    and added dcond, which left the trained function and the posterior unchanged but shifted the reported NLL
    (loss_fn, loglik_fn, marg_loglik) by an amount that depends on dcond, hence on the lag. Models and 'mll'
    files saved before that date carry the old offset in 'mll'.
    """
    def __init__(self, din, dcond=0, eps=1e-5):

        super().__init__()
        self.eps = eps
        self.dcond = dcond
        self.gamma = nnx.Param(jnp.zeros((1, din + dcond)))
        self.beta = nnx.Param(jnp.zeros((1, din + dcond)))
        self.use_running_average = None
        self.batch_mean = None
        self.batch_var = None

    def __call__(self, x):

        if not self.use_running_average:

            m = jnp.mean(x, axis=0)
            v = jnp.var(x, axis=0) + self.eps 
            self.batch_mean = None

        else:

            if self.batch_mean is None:

                self.set_batch_stats_func(x)

            m = self.batch_mean
            v = self.batch_var

        gamma = jnp.asarray(self.gamma)
        beta = jnp.asarray(self.beta)
        x_hat = (x - m) / jnp.sqrt(v)
        x_hat = x_hat * jnp.exp(gamma) + beta
        log_det = jnp.sum(gamma[:, self.dcond:] - 0.5 * jnp.log(v[self.dcond:]))
        x_out = jnp.concatenate([x[:, :self.dcond], x_hat[:, self.dcond:]], axis=1)

        return x_out, log_det
    
    def backward(self, x):

        if not self.use_running_average:

            m = jnp.mean(x, axis=0)
            v = jnp.var(x, axis=0) + self.eps 
            self.batch_mean = None

        else:

            if self.batch_mean is None:

                self.set_batch_stats_func(x)

            m = self.batch_mean
            v = self.batch_var

        x_hat = (x - self.beta) * jnp.exp(-self.gamma) * jnp.sqrt(v) + m
        log_det = jnp.sum(-self.gamma[:, self.dcond:] + 0.5 * jnp.log(v[self.dcond:]))
        x_out = jnp.concatenate([x[:, :self.dcond], x_hat[:, self.dcond:]], axis=1)

        return x_out, log_det

    def set_batch_stats_func(self, x):
        # print("setting batch stats for validation")
        self.batch_mean = jnp.mean(x, axis=0)
        self.batch_var = jnp.var(x, axis=0) + self.eps