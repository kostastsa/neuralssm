import jax.numpy as jnp

class ExperimentResults:

    def __init__(self):
        
        self.results = []

    def make_jnp(self):

        self.results = jnp.array(self.results)



class ABC_Results(ExperimentResults):

    def __init__(self):
        super().__init__()
        self.name = 'SMC-ABC'
        self.marker = 'o'
        self.color = 'b'


class MCMC_Results(ExperimentResults):

    def __init__(self):
        super().__init__()
        self.name = 'BPF-MCMC'
        self.marker = 'x'
        self.color = 'r'


class SNL_Results(ExperimentResults):

    def __init__(self):
        super().__init__()
        self.name = 'SNL'
        self.marker = 's'
        self.color = 'tomato'


class TSNL_Results(ExperimentResults):

    def __init__(self):
        super().__init__()
        self.name = 'T-SNL'
        self.marker = 'd'
        self.color = 'forestgreen'