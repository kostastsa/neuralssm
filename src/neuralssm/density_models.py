from flax import nnx
import jax.numpy as jnp
import jax.random as jr
import jax
from masks import create_masks, create_degrees, create_masks2, create_degrees2
from layers import BatchNormLayer, MaskedLinear


### MADE IMPLEMENTATION 2 (Implements conditional with full dependencies)

class MADEnet2(nnx.Module):
   
    def __init__(self, 
                 din: int, 
                 dhidden: int, 
                 nhidden: int,
                 dcond: int = 0,
                 rngs: nnx.Rngs = nnx.Rngs(0), 
                 random: bool=False, 
                 reverse: bool=False, 
                 batch_norm: bool=False,
                 dropout: bool=False):
        dout = 2 * din
        self.din, self.dhidden, self.nhidden = din, dhidden, nhidden
        degrees = create_degrees2(rngs.split(), din, dhidden, nhidden, dcond, random, reverse)
        masks = create_masks2(degrees, dcond)
        self.input_order = degrees[0]
        self.layers = []
        for mask in masks[0]:
            d1, d2 = jnp.shape(mask)
            self.layers.append(MaskedLinear(mask, d1, d2, rngs=rngs))
            if batch_norm: self.layers.append(nnx.BatchNorm(d2, rngs=rngs))
            if dropout: self.layers.append(nnx.Dropout(rate=0.1, rngs=rngs))
            self.layers.append(nnx.relu)
        out_mask = jnp.concatenate([masks[1], masks[1]], axis=1)
        self.layers.append(MaskedLinear(out_mask, dhidden, dout, rngs=rngs))

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            x = layer(x)
        return x.reshape(-1, 2, self.din)
  
class MADE2(nnx.Module):
    
    def __init__(self, 
                 din: int, 
                 dhidden: int, 
                 nhidden: int,
                 dcond: int = 0,
                 rngs: nnx.Rngs = nnx.Rngs(0), 
                 random: bool=False, 
                 reverse:bool =False,
                 batch_norm: bool=False,
                 dropout: bool=False):
        self.network = MADEnet(din, dhidden, nhidden, dcond, rngs, random, reverse, batch_norm, dropout)
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
        mus = mus[:, jnp.argsort(self.input_order[self.dcond:])]
        alphas = params[:, 1]
        alphas = alphas[:, jnp.argsort(self.input_order[self.dcond:])]
        x = x[:, self.dcond:]
        us = jnp.exp(-alphas) * (x[:, jnp.argsort(self.input_order[self.dcond:])] - mus)
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
        for i in range(1 + self.dcond, self.dcond + self.din + 1):
            idx = jnp.argwhere(self.input_order == i)[0, 0]
            shift_idx = idx - self.dcond
            params = self.network(x)    
            mus = params[:, 0]
            alphas = params[:, 1]
            alphas = jnp.clip(alphas, a_max=10)
            new_x = mus[:, shift_idx] + jnp.exp(alphas[:, shift_idx]) * u[:, i-1]
            x = x.at[:, idx].set(new_x)
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

##### MADE IMPLEMENTATION (Implements conditional with dependencies as maf paper)

class MADEnet(nnx.Module):
   
    def __init__(self, 
                 din: int, 
                 dhidden: int, 
                 nhidden: int,
                 dcond: int = 0,
                 rngs: nnx.Rngs = nnx.Rngs(0), 
                 random: bool=False, 
                 reverse: bool=False, 
                 batch_norm: bool=False,
                 dropout: bool=False):
        dout = 2 * din
        self.din, self.dhidden, self.nhidden = din, dhidden, nhidden
        degrees = create_degrees(rngs.split(), din, dhidden, nhidden, random, reverse)
        masks = create_masks(degrees, dcond)
        self.input_order = degrees[0]
        self.layers = []
        for mask in masks[0]:
            d1, d2 = jnp.shape(mask)
            self.layers.append(MaskedLinear(mask, d1, d2, rngs=rngs))
            if batch_norm: self.layers.append(nnx.BatchNorm(d2, rngs=rngs))
            if dropout: self.layers.append(nnx.Dropout(rate=0.1, rngs=rngs))
            self.layers.append(nnx.relu)
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
                 dcond: int = 0,
                 rngs: nnx.Rngs = nnx.Rngs(0), 
                 random: bool=False, 
                 reverse:bool =False,
                 batch_norm: bool=False,
                 dropout: bool=False):
        self.network = MADEnet(din, dhidden, nhidden, dcond, rngs, random, reverse, batch_norm, dropout)
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

## MAF IMPLEMENTATION

class MAF(nnx.Module):
   
    def __init__(self,
                din: int,
                nmade: int,
                dhidden: int,
                nhidden: int,
                dcond : int = 0,
                rngs: nnx.Rngs = nnx.Rngs(0),
                random: bool = False,
                reverse: bool = False,
                batch_norm: bool = False,
                dropout: bool = False):
        self.din, self.dcond, self.nmade, self.nhidden = din, dcond, nmade, nhidden
        self.layers = []
        for _ in range(nmade):
            self.layers.append(MADE(din, dhidden, nhidden, dcond, rngs, random, reverse, batch_norm, dropout))
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
        return  jnp.mean(loss)
    
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
            u = jnp.concatenate([cond_samples, u], axis=1)
        u = self.backward(u)
        return u

######### Unconditional implenetations of MADE and MAF

class OMADEnet(nnx.Module):
    def __init__(self, 
                 din: int, 
                 dhidden: int, 
                 nhidden: int,
                 rngs: nnx.Rngs, 
                 random: bool=False, 
                 reverse: bool=False, 
                 batch_norm: bool=False,
                 dropout: bool=False):
        dout = 2 * din
        self.din, self.dhidden, self.nhidden = din, dhidden, nhidden
        degrees = create_degrees(rngs.split(), din, dhidden, nhidden, random, reverse)
        masks = create_masks(degrees)
        self.input_order = degrees[0]
        self.layers = []
        for mask in masks[0]:
            d1, d2 = jnp.shape(mask)
            self.layers.append(MaskedLinear(mask, d1, d2, rngs=rngs))
            if batch_norm: self.layers.append(nnx.BatchNorm(d2, rngs=rngs))
            if dropout: self.layers.append(nnx.Dropout(rate=0.1, rngs=rngs))
            self.layers.append(nnx.relu)
        out_mask = jnp.concatenate([masks[1], masks[1]], axis=1)
        self.layers.append(MaskedLinear(out_mask, dhidden, dout, rngs=rngs))

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            x = layer(x)
        return x.reshape(-1, 2, self.din)
    
    def init_first_dim(self, x):
        idx = jnp.argwhere(self.input_order == 1)[0, 0]
        bmu = jnp.mean(x[:, idx])
        balpha = jnp.log(jnp.std(x[:, idx]))
        self.layers[-1].b.value = self.layers[-1].b.value.at[idx].set(bmu)
        self.layers[-1].b.value = self.layers[-1].b.value.at[self.din + idx].set(balpha)
  
class OMADE(nnx.Module):
    def __init__(self, 
                 din: int, 
                 dhidden: int, 
                 nhidden: int,
                 rngs: nnx.Rngs = None, 
                 random: bool=False, 
                 reverse:bool =False,
                 batch_norm: bool=False,
                 dropout: bool=False):
        self.network = OMADEnet(din, dhidden, nhidden, rngs, random, reverse, batch_norm, dropout)
        self.din = din
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
        us = jnp.exp(-alphas) * (x[:, jnp.argsort(self.input_order)] - mus)
        return us, jnp.sum(-alphas, axis=1)

    def backward(self, u):
        assert len(u.shape) == 2
        x = jnp.zeros(jnp.shape(u))
        for i in range(1, self.din+1):
            idx = jnp.argwhere(self.input_order == i)[0, 0]
            params = self.network(x)    
            mus = params[:, 0]
            alphas = params[:, 1]
            alphas = jnp.clip(alphas, a_max=10)
            new_x = mus[:, idx] + jnp.exp(alphas[:, idx]) * u[:, i-1]
            x = x.at[:, idx].set(new_x)
        return x, None
  
    def generate(self, key, num_samples):
        u = jr.normal(key, (num_samples, self.din))
        samples, _ = self.backward(u)
        return samples
    
    def loss_fn(self, x):
        u, log_det = self(x)
        loss = jnp.sum(u ** 2, axis=1)/2
        loss += 0.5 * x.shape[1] * jnp.log(2 * jnp.pi)
        loss -= log_det
        return  jnp.mean(loss)
    
####################################################################################

class OMAF(nnx.Module):
    def __init__(self,
                din: int,
                nmade: int,
                dhidden: int,
                nhidden: int,
                rngs: nnx.Rngs = nnx.Rngs(0),
                random: bool = False,
                reverse: bool = False,
                batch_norm: bool = False,
                dropout: bool = False):
        self.din, self.nmade, self.nhidden = din, nmade, nhidden
        self.layers = []
        for _ in range(nmade):
            self.layers.append(OMADE(din, dhidden, nhidden, rngs, random, reverse, batch_norm, dropout))
            self.layers.append(BatchNormLayer(din))

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
        u, log_det = self(x)
        loss = jnp.sum(u ** 2, axis=1)/2
        loss += 0.5 * x.shape[1] * jnp.log(2 * jnp.pi)
        loss -= log_det
        return  jnp.mean(loss)
    
    def backward(self, u):
        """
        computes the normalizing flow transformation u->x
        """
        for layer in self.layers[::-1]:
                u, _ = layer.backward(u)
        return u
    
    def generate(self, key, num_samples):
        u = jr.normal(key, (num_samples, self.din))
        u = self.backward(u)
        return u
