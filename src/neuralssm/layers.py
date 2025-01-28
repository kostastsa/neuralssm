from flax import nnx
from jax import numpy as jnp
from flax import linen as nn
import jax.random as jr
import jax


class MaskedLinear(nnx.Module):
  def __init__(self, mask, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    std = jnp.sqrt(din)
    self.w = nnx.Param(jr.uniform(key, (din, dout), minval=-1/std, maxval=1/std))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout
    self.mask = nn.Variable(None, "constants", "mask", lambda: mask)

  def __call__(self, x: jax.Array):
    w = jnp.asarray(self.w)  # Convert self.w to a standard JAX array
    b = jnp.asarray(self.b)
    return x @ jnp.multiply(self.mask.unbox(), w)+ b

class BatchNormLayerv2(nnx.Module):
    def __init__(self, din, dcond=0, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.dcond = dcond
        self.gamma = nnx.Param(jnp.zeros((1, din)))
        self.beta = nnx.Param(jnp.zeros((1, din)))
        self.use_running_average = None
        self.batch_mean = None
        self.batch_var = None

    def __call__(self, x):
        xcond = x[:, :self.dcond]
        x = x[:, self.dcond:]
        if not self.use_running_average:
            m = jnp.mean(x, axis=0)
            v = jnp.var(x, axis=0) + self.eps 
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean.clone()
            v = self.batch_var.clone()
        gamma = jnp.asarray(self.gamma)
        beta = jnp.asarray(self.beta)
        x_hat = (x - m) / jnp.sqrt(v)
        x_hat = x_hat * jnp.exp(gamma) + beta
        log_det = jnp.sum(self.gamma - 0.5 * jnp.log(v))
        x_out = jnp.concatenate([xcond, x_hat], axis=1)
        return x_out, log_det
    
    def backward(self, x):
        xcond = x[:, :self.dcond]
        x = x[:, self.dcond:]
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
        log_det = jnp.sum(-self.gamma + 0.5 * jnp.log(v))
        x_out = jnp.concatenate([xcond, x_hat], axis=1)
        return x_out, log_det

    def set_batch_stats_func(self, x):
        # print("setting batch stats for validation")
        self.batch_mean = jnp.mean(x, axis=0)
        self.batch_var = jnp.var(x, axis=0) + self.eps

class BatchNormLayer(nnx.Module):
    """
    Difference is in handling the conditional input. This computes statistics for the conditional input as well.
    At the end it concatenates the conditional input with the transformed input.
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
            m = self.batch_mean.clone()
            v = self.batch_var.clone()
        gamma = jnp.asarray(self.gamma)
        beta = jnp.asarray(self.beta)
        x_hat = (x - m) / jnp.sqrt(v)
        x_hat = x_hat * jnp.exp(gamma) + beta
        log_det = jnp.sum(self.gamma - 0.5 * jnp.log(v), initial=self.dcond)
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
        log_det = jnp.sum(-self.gamma + 0.5 * jnp.log(v))
        x_out = jnp.concatenate([x[:, :self.dcond], x_hat[:, self.dcond:]], axis=1)
        return x_out, log_det

    def set_batch_stats_func(self, x):
        # print("setting batch stats for validation")
        self.batch_mean = jnp.mean(x, axis=0)
        self.batch_var = jnp.var(x, axis=0) + self.eps