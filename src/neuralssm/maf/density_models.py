from flax import nnx
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import jax
from .masks import create_masks, create_degrees
from .layers import BatchNormLayer, MaskedLinear, activations


class MADEnet(nnx.Module):
   
    def __init__(self, 
                 din: int, 
                 dhidden: int, 
                 nhidden: int,
                 act_fun: str,
                 dcond: int = 0,
                 rngs: nnx.Rngs = nnx.Rngs(0), 
                 random: bool=False, 
                 reverse: bool=False, 
                 batch_norm: bool=False,
                 dropout: bool=False):
        try: 

            act_fun = activations[act_fun]

        except:

            raise ValueError(f"Activation function {act_fun} not found in nnx module")

        dout = 2 * din
        self.din, self.dhidden, self.nhidden = din, dhidden, nhidden
        degrees = create_degrees(rngs.split(), din, dhidden, nhidden, random, reverse)
        masks = create_masks(degrees, dcond)
        self.input_order = degrees[0]
        self.layers = []

        for mask in masks[0]:

            d1, d2 = jnp.shape(mask)
            self.layers.append(MaskedLinear(mask, d1, d2, rngs=rngs))

            if batch_norm: 
                
                self.layers.append(nnx.BatchNorm(d2, rngs=rngs))

            if dropout: 
                
                self.layers.append(nnx.Dropout(rate=0.1, rngs=rngs))

            self.layers.append(act_fun)

        out_mask = jnp.concatenate([masks[1], masks[1]], axis=1)
        self.layers.append(MaskedLinear(out_mask, dhidden, dout, rngs=rngs))

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            x = layer(x)
        return x.reshape(-1, 2, self.din)


class MADE(nnx.Module):
    
    def __init__(self, 
                 din: int, 
                 dhidden: int, 
                 nhidden: int,
                 act_fun: str,
                 dcond: int = 0,
                 rngs: nnx.Rngs = nnx.Rngs(0), 
                 random: bool=False, 
                 reverse:bool =False,
                 batch_norm: bool=False,
                 dropout: bool=False):
        self.network = MADEnet(din, dhidden, nhidden, act_fun, dcond, rngs, random, reverse, batch_norm, dropout)
        self.din = din
        self.dcond = dcond
        self.input_order = self.network.input_order

    def __call__(self, x: jax.Array):
        """
        Implements the inverse MADE transformation x -> u
        A note on the order of the inputs: the made_params call function computes the parameters mu and alpha for the 
        ordering given by the 
        """
        params = self.network(x)
        mus = params[:, 0]
        mus = mus[:, jnp.argsort(self.input_order)]
        alphas = params[:, 1]
        alphas = alphas[:, jnp.argsort(self.input_order)]
        xin = x[:, self.dcond:]
        us = jnp.exp(-alphas) * (xin[:, jnp.argsort(self.input_order)] - mus)
        return jnp.concatenate([x[:, :self.dcond], us], axis=1), jnp.sum(-alphas, axis=1)

    def backward(self, u):
        '''
        Takes as input a seed u and returns the corresponding x. If self.cond!=0 
        the first self.cond elements of u are assumed to be the conditioning variables, in which
        case they are copied to the output.
        '''
        assert len(u.shape) == 2
        assert u.shape[1] == self.din + self.dcond
        x = jnp.zeros(jnp.shape(u))
        x = x.at[:, :self.dcond].set(u[:, :self.dcond])
        for i in range(1, self.din + 1):
            idx = jnp.argwhere(self.input_order == i)[0, 0]
            params = self.network(x)    
            mus = params[:, 0]
            alphas = params[:, 1]
            alphas = jnp.clip(alphas, a_max=10)
            new_x = mus[:, idx] + jnp.exp(alphas[:, idx]) * u[:, self.dcond+i-1]
            x = x.at[:, self.dcond+idx].set(new_x)
        return x, None
  
    def generate(self, key, num_samples, cond_samples=None):
        u = jr.normal(key, (num_samples, self.din))
        if cond_samples is not None:
            u = jnp.concatenate([cond_samples, u], axis=1)
        samples, _ = self.backward(u)
        return samples
    
    def loss_fn(self, x):
        out, log_det = self(x)
        u = out[:, self.dcond:]
        loss = jnp.sum(u ** 2, axis=1)/2
        loss += 0.5 * u.shape[1] * jnp.log(2 * jnp.pi)
        loss -= log_det
        return jnp.mean(loss)


class MAF(nnx.Module):
   
    def __init__(self,
                din: int,
                nmade: int,
                dhidden: int,
                nhidden: int,
                act_fun: str,
                dcond : int = 0,
                rngs: nnx.Rngs = nnx.Rngs(0),
                random_order: bool = False,
                reverse: bool = False,
                batch_norm: bool = False,
                dropout: bool = False):

        self.din, self.dcond, self.nmade, self.nhidden = din, dcond, nmade, nhidden
        self.layers = []

        for _ in range(nmade):

            self.layers.append(MADE(din, dhidden, nhidden, act_fun, dcond, rngs, random_order, reverse, batch_norm, dropout))
            self.layers.append(BatchNormLayer(din, dcond))

    def __call__(self, x: jax.Array):
        """
        computes the inverse normalizing flow transformation x->u
        """

        log_det_sum = jnp.zeros(x.shape[0])

        for layer in self.layers:

            x, log_det = layer(x)
            log_det_sum += log_det

        return x, log_det_sum

    def loss_fn(self, x):

        out, log_det = self(x)
        u = out[:, self.dcond:]
        loss = jnp.sum(u ** 2, axis=1)/2
        loss += 0.5 * u.shape[1] * jnp.log(2 * jnp.pi)
        loss -= log_det

        return jnp.mean(loss)
    
    def backward(self, u):
        """
        computes the normalizing flow transformation u->x
        """

        for layer in self.layers[::-1]:
                
                u, _ = layer.backward(u)

        return u
    
    def generate(self, key, num_samples, cond_samples=None):

        u = jr.normal(key, (num_samples, self.din))

        if cond_samples is not None:

            tiles = jnp.tile(cond_samples, (num_samples, 1))
            u = jnp.concatenate([tiles, u], axis=1)

        u = self.backward(u)

        return u

