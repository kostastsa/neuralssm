import tensorflow_probability as tfp


class MatrixNormal(tfd.Normal):
    
    def _set_shape(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols

    def sample(self, num_samples, key):
        return super().sample((num_samples, self.nrows, self.ncols), key).reshape(num_samples, self.nrows, self.ncols)
    
class ScaledIdenity(tfd.Normal):
    
    def _set_shape(self, ndim):
        self.nrows = ndim

    def sample(self, num_samples, key):
        factors = super().sample((num_samples, ), key)
        return  jnp.einsum('i, jk-> ijk', factors, jnp.eye(self.nrows))

dist = ScaledIdenity(loc=1.0, scale=0.3)
