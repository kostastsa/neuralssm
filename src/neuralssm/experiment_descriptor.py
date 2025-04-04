# This module is taken/adapted from the repository ([https://github.com/gpapamak/snl.git])
# Originally authored by George Papamakarios, under the MIT License

import os
import re
import util.misc

main_dir = '/Users/kostastsampourakis/Desktop/code/Python/projects/neuralssm/src/neuralssm'

class ParseError(Exception):
    """
    Exception to be thrown when there is a parsing error.
    """

    def __init__(self, str):
        self.str = str

    def __str__(self):
        return self.str


class SimulatorDescriptor:

    @staticmethod
    def get_descriptor(str):

        if re.match('lgssm', str):
            return LGSSM_Descriptor(str)
        
        elif re.match('lvssm', str):
            return LVSSM_Descriptor(str)

        elif re.match('svssm', str):
            return SVSSM_Descriptor(str)
        
        elif re.match('sirssm', str):
            return SIRSSM_Descriptor(str)

        else:
            raise ParseError(str)


class LGSSM_Descriptor(SimulatorDescriptor):
    
    def __init__(self, str=None):

        self.state_dim = None
        self.emission_dim = None
        self.input_dim = 0
        self.num_timesteps = None
        self.target_vars = None
        self.name = 'lgssm'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'lgssm\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(self.state_dim)
        str += '\t\temission_dim: {0},\n'.format(self.emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(self.num_timesteps)
        str += '\t\ttarget_vars: {0}\n'.format('_'.join(self.target_vars))
        str += '\t}'

        return str

    def create_desc(self, state_dim, emission_dim, num_timesteps, dt_obs, target_vars):
        
        str = 'sim: lgssm\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(state_dim)
        str += '\t\temission_dim: {0},\n'.format(emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(num_timesteps)
        str += '\t\ttarget_vars: {0}\n'.format(target_vars)
        str += '\t}'

        return str



    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'lgssm\{state_dim:(.*),emission_dim:(.*),num_timesteps:(.*),target_vars:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.state_dim = int(m.group(1))
        self.emission_dim = int(m.group(2))
        self.num_timesteps = int(m.group(3))
        self.target_vars = m.group(4).split('_')

    def get_dir(self):

        return os.path.join('lgssm', 'state-dim_{0}_emission-dim_{1}_num-timesteps_{2}_target-vars_{3}'.format(self.state_dim, self.emission_dim, self.num_timesteps, '_'.join(self.target_vars)))


class SVSSM_Descriptor(SimulatorDescriptor):
    
    def __init__(self, str=None):

        self.state_dim = None
        self.emission_dim = None
        self.input_dim = 0
        self.num_timesteps = None
        self.target_vars = None
        self.name = 'svssm'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'svssm\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(self.state_dim)
        str += '\t\temission_dim: {0},\n'.format(self.emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(self.num_timesteps)
        str += '\t\ttarget_vars: {0}\n'.format('_'.join(self.target_vars))
        str += '\t}'

        return str
    
    def create_desc(self, state_dim, emission_dim, num_timesteps, dt_obs, target_vars):

        str = 'sim: svssm\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(state_dim)
        str += '\t\temission_dim: {0},\n'.format(emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(num_timesteps)
        str += '\t\ttarget_vars: {0}\n'.format(target_vars)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'svssm\{state_dim:(.*),emission_dim:(.*),num_timesteps:(.*),target_vars:(.*)\}\Z', str)

        if m is None:

            raise ParseError(str)

        self.state_dim = int(m.group(1))
        self.emission_dim = int(m.group(2))
        self.num_timesteps = int(m.group(3))
        self.target_vars = m.group(4).split('_')

    def get_dir(self):

        return os.path.join('svssm', 'state-dim_{0}_emission-dim_{1}_num-timesteps_{2}_target-vars_{3}'.format(self.state_dim, self.emission_dim, self.num_timesteps, '_'.join(self.target_vars)))


class LVSSM_Descriptor(SimulatorDescriptor):
    
    def __init__(self, str=None):

        self.state_dim = 2
        self.input_dim = 0
        self.emission_dim = None
        self.num_timesteps = None
        self.dt_obs = None
        self.target_vars = None
        self.name = 'lvssm'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'lvssm\n'
        str += '\t{\n'
        str += '\t\temission_dim: {0},\n'.format(self.emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(self.num_timesteps)
        str += '\t\tdt_obs: {0},\n'.format(self.dt_obs)
        str += '\t\ttarget_vars: {0}\n'.format('_'.join(self.target_vars))
        str += '\t}'

        return str
    
    def create_desc(self, state_dim, emission_dim, num_timesteps, dt_obs, target_vars):

        str = 'sim: lvssm\n'
        str += '\t{\n'
        str += '\t\temission_dim: {0},\n'.format(emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(num_timesteps)
        str += '\t\tdt_obs: {0},\n'.format(dt_obs)
        str += '\t\ttarget_vars: {0}\n'.format(target_vars)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'lvssm\{emission_dim:(.*),num_timesteps:(.*),dt_obs:(.*),target_vars:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.emission_dim = int(m.group(1))
        self.num_timesteps = int(m.group(2))
        self.dt_obs = float(m.group(3))
        self.target_vars = m.group(4).split('_')

    def get_dir(self):

        return os.path.join('lvssm', 'emission-dim_{0}_num-timesteps_{1}_dt_obs_{2}_target-vars_{3}'.format(self.emission_dim, self.num_timesteps, self.dt_obs, '_'.join(self.target_vars)))


class SIRSSM_Descriptor(SimulatorDescriptor):
    
    def __init__(self, str=None):

        self.state_dim = 3
        self.emission_dim = 1
        self.input_dim = 0
        self.num_timesteps = None
        self.dt_obs = None
        self.target_vars = None
        self.name = 'sirssm'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'sirssm\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(self.state_dim)
        str += '\t\temission_dim: {0},\n'.format(self.emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(self.num_timesteps)
        str += '\t\tdt_obs: {0},\n'.format(self.dt_obs)
        str += '\t\ttarget_vars: {0}\n'.format('_'.join(self.target_vars))
        str += '\t}'

        return str
    
    def create_desc(self, state_dim, emission_dim, num_timesteps, dt_obs, target_vars):

        str = 'sim: sirssm\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(state_dim)
        str += '\t\temission_dim: {0},\n'.format(emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(num_timesteps)
        str += '\t\tdt_obs: {0},\n'.format(dt_obs)
        str += '\t\ttarget_vars: {0}\n'.format(target_vars)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'sirssm\{state_dim:(.*),emission_dim:(.*),num_timesteps:(.*),dt_obs:(.*),target_vars:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.state_dim = int(m.group(1))
        self.emission_dim = int(m.group(2))
        self.num_timesteps = int(m.group(3))
        self.dt_obs = float(m.group(4))
        self.target_vars = m.group(5).split('_')

    def get_dir(self):

        return os.path.join('sirssm', 'state-dim_{0}_emission-dim_{1}_num-timesteps_{2}_dt_obs_{3}_target-vars_{4}'.format(self.state_dim, self.emission_dim, self.num_timesteps, self.dt_obs, '_'.join(self.target_vars)))


class InferenceDescriptor:

    @staticmethod
    def get_descriptor(str=None):
        if re.match('smc_abc', str):
            return SMC_ABC_Descriptor(str)

        elif re.match('snl', str):
            return SNL_Descriptor(str)
        
        elif re.match('tsnl', str):
            return TSNL_Descriptor(str)
        
        elif re.match('bpf_mcmc', str):
            return BPF_MCMC_Descriptor(str)

        else:
            raise ParseError(str)


class ABC_Descriptor(InferenceDescriptor):

    def get_id(self):

        raise NotImplementedError('abstract method')

    def get_dir(self):

        return os.path.join('abc', self.get_id())


class SMC_ABC_Descriptor(ABC_Descriptor):

    def __init__(self, str=None):

        self.n_samples = None
        self.qmax = None
        self.sigma = None

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'smc_abc\n'
        str += '\t{\n'
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\qmax: {0},\n'.format(self.qmax)
        str += '\t\tsigma: {0}\n'.format(self.sigma)
        str += '\t}'

        return str
    
    def create_desc(self, n_samples, qmax, sigma):

        str = 'inf: smc_abc\n'
        str += '\t{\n'
        str += '\t\tn_samples: {0},\n'.format(n_samples)
        str += '\t\tqmax: {0},\n'.format(qmax)
        str += '\t\tsigma: {0}\n'.format(sigma)
        str += '\t}'

        return str


    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'smc_abc\{n_samples:(.*),qmax:(.*),sigma:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_samples = int(m.group(1))
        self.qmax = float(m.group(2))
        self.sigma = float(m.group(3))

    def get_id(self, delim='_'):

        id = 'smcabc'
        id += delim + 'samples' + delim + str(self.n_samples)
        id += delim + 'qmax' + delim + str(self.qmax)
        id += delim + 'sigma' + delim + str(self.sigma)

        return id


class PRT_MCMC_Descriptor(InferenceDescriptor):

    def get_id(self):

        raise NotImplementedError('abstract method')

    def get_dir(self):

        return os.path.join('mcmc', self.get_id())


class BPF_MCMC_Descriptor(PRT_MCMC_Descriptor):

    def __init__(self, str=None):

        self.num_prt = None
        self.num_iters = None
        self.mcmc_steps = None

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):   

        str = 'bpf_mcmc\n'
        str += '\t{\n'
        str += '\t\tnum_prt: {0},\n'.format(self.num_prt)
        str += '\t\tnum_iters: {0},\n'.format(self.num_iters)
        str += '\t\tmcmc_steps: {0}\n'.format(self.mcmc_steps)
        str += '\t}'

        return str
    
    def create_desc(self, num_prt, num_iters, mcmc_steps):

        str = 'inf: bpf_mcmc\n'
        str += '\t{\n'
        str += '\t\tnum_prt: {0},\n'.format(num_prt)
        str += '\t\tnum_iters: {0},\n'.format(num_iters)
        str += '\t\tmcmc_steps: {0}\n'.format(mcmc_steps)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'bpf_mcmc\{num_prt:(.*),num_iters:(.*),mcmc_steps:(.*)\}\Z', str)
        if m is None:
            raise ParseError(str)

        self.num_prt = int(m.group(1))
        self.num_iters = int(m.group(2))
        self.mcmc_steps = int(m.group(3))

    def get_id(self, delim='_'):

        id = 'bpf'
        id += delim + 'numprt' + delim + str(self.num_prt)
        id += delim + 'numiters' + delim + str(self.num_iters)
        id += delim + 'mcmcsteps' + delim + str(self.mcmc_steps)

        return id


class SNL_Descriptor(InferenceDescriptor):

    def __init__(self, str=None):

        self.model = None
        self.n_samples = None
        self.n_rounds = None
        self.train_on = None
        self.mcmc_steps = None
        self.lag = -1

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'snl\n'
        str += '\t{\n'
        str += '\t\tmodel: {0},\n'.format(self.model.pprint())
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\tn_rounds: {0},\n'.format(self.n_rounds)
        str += '\t\ttrain_on: {0},\n'.format(self.train_on)
        str += '\t\tmcmc_steps: {0}\n'.format(self.mcmc_steps)
        str += '\t}'

        return str

    def create_desc(self, model_args, n_samples, n_rounds, train_on, mcmc_steps):

        model_desc = MAF_Descriptor().create_desc(*model_args)

        str = 'inf: snl\n'
        str += '\t{\n'
        str += model_desc
        str += ',\t\tn_samples: {0},\n'.format(n_samples)
        str += '\t\tn_rounds: {0},\n'.format(n_rounds)
        str += '\t\ttrain_on: {0},\n'.format(train_on)
        str += '\t\tmcmc_steps: {0}\n'.format(mcmc_steps)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'snl\{model:(.*),n_samples:(.*),n_rounds:(.*),train_on:(all|last|best),mcmc_steps:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.model = ModelDescriptor.get_descriptor(m.group(1))
        self.n_samples = int(m.group(2))
        self.n_rounds = int(m.group(3))
        self.train_on = m.group(4)
        self.mcmc_steps = int(m.group(5))

    def get_dir(self):

        return os.path.join('nde','snl','samples_{0}_rounds_{1}_train_on_{2}_mcmc_steps_{3}'.format(self.n_samples, self.n_rounds, self.train_on, self.mcmc_steps), self.model.get_id())


class TSNL_Descriptor(InferenceDescriptor):

    def __init__(self, str=None):

        self.model = None
        self.n_samples = None
        self.n_rounds = None
        self.lag = None
        self.subsample = None
        self.train_on = None
        self.mcmc_steps = None

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'tsnl\n'
        str += '\t{\n'
        str += '\t\tmodel: {0},\n'.format(self.model.pprint())
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\tn_rounds: {0},\n'.format(self.n_rounds)
        str += '\t\tlag: {0},\n'.format(self.lag)
        str += '\t\tsubsample: {0},\n'.format(self.subsample)
        str += '\t\ttrain_on: {0},\n'.format(self.train_on)
        str += '\t\tmcmc_steps: {0}\n'.format(self.mcmc_steps)
        str += '\t}'

        return str

    def create_desc(self, model_args, n_samples, n_rounds, lag, subsample, train_on, mcmc_steps):

        model_desc = MAF_Descriptor().create_desc(*model_args)

        str = 'inf: tsnl\n'
        str += '\t{\n'
        str += '\t\t'
        str += model_desc + ','
        str += '\n\t\tn_samples: {0},\n'.format(n_samples)
        str += '\t\tn_rounds: {0},\n'.format(n_rounds)
        str += '\t\tlag: {0},\n'.format(lag)
        str += '\t\tsubsample: {0},\n'.format(subsample)
        str += '\t\ttrain_on: {0},\n'.format(train_on)
        str += '\t\tmcmc_steps: {0}\n'.format(mcmc_steps)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'tsnl\{model:(.*),n_samples:(.*),n_rounds:(.*),lag:(.*),subsample:(.*),train_on:(all|last|best),mcmc_steps:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.model = ModelDescriptor.get_descriptor(m.group(1))
        self.n_samples = int(m.group(2))
        self.n_rounds = int(m.group(3))
        self.lag = int(m.group(4))
        self.subsample = float(m.group(5))
        self.train_on = m.group(6)
        self.mcmc_steps = int(m.group(7))

    def get_dir(self):

        return os.path.join('nde','tsnl', 'samples_{0}_rounds_{1}_lag_{2}_subsample_{3}_train_on_{4}_mcmc_steps_{5}'.format(self.n_samples, self.n_rounds, self.lag, self.subsample, self.train_on, self.mcmc_steps), self.model.get_id())
        

class ModelDescriptor:

    @staticmethod
    def get_descriptor(str):

        if re.match('maf', str):
            return MAF_Descriptor(str)

        else:
            raise ParseError(str)


class MAF_Descriptor(ModelDescriptor):

    def __init__(self, str = None):

        self.din = None
        self.dcond = None
        self.act_fun = None
        self.nmades = None
        self.dhidden = None
        self.nhidden = None
        self.rng = None
        self.random_order = None
        self.reverse = None
        self.batch_norm = None
        self.dropout = None
        self.nepochs = None
        self.lr = None

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = '\t\tmaf\n'
        str += '\t\t{\n'
        str += '\t\t\tn_mades: {0},\n'.format(self.nmades)
        str += '\t\t\td_hidden: {0},\n'.format(self.dhidden)
        str += '\t\t\tn_hiddens: {0},\n'.format(self.nhidden)
        str += '\t\t\tact_fun: {0},\n'.format(self.act_fun)
        str += '\t\t\trandom_order: {0},\n'.format(self.random_order)
        str += '\t\t\treverse: {0},\n'.format(self.reverse)
        str += '\t\t\tbatch_norm: {0},\n'.format(self.batch_norm)
        str += '\t\t\tdropout: {0}\n'.format(self.dropout)
        str += '\t\t\tnepochs: {0},\n'.format(self.nepochs)
        str += '\t\t\tlr: {0}\n'.format(self.lr)
        str += '\t\t}'

        return str
    
    def create_desc(self, nmades, dhidden, nhidden, act_fun, random_order, reverse, batch_norm, dropout, nepochs, lr):

        str = 'model: maf\n'
        str += '\t\t{\n'
        str += '\t\t\tn_mades: {0},\n'.format(nmades)
        str += '\t\t\td_hidden: {0},\n'.format(dhidden)
        str += '\t\t\tn_hiddens: {0},\n'.format(nhidden)
        str += '\t\t\tact_fun: {0},\n'.format(act_fun)
        str += '\t\t\trandom_order: {0},\n'.format(random_order)
        str += '\t\t\treverse: {0},\n'.format(reverse)
        str += '\t\t\tbatch_norm: {0},\n'.format(batch_norm)
        str += '\t\t\tdropout: {0}\n,'.format(dropout)
        str += '\t\t\tnepochs: {0},\n'.format(nepochs)
        str += '\t\t\tlr: {0}\n'.format(lr)
        str += '\t\t}'

        return str


    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'maf\{n_mades:(.*),d_hidden:(.*),n_hiddens:(.*),act_fun:(relu|tanh|elu),random_order:(.*),reverse:(.*),batch_norm:(.*),dropout:(.*),nepochs:(.*),lr:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.nmades = int(m.group(1))
        self.dhidden = int(m.group(2))
        self.nhidden = int(m.group(3))
        self.act_fun = m.group(4)
        self.random_order = util.misc.str_to_bool(m.group(5))
        self.reverse = util.misc.str_to_bool(m.group(6))
        self.batch_norm = util.misc.str_to_bool(m.group(7))
        self.dropout = util.misc.str_to_bool(m.group(8))
        self.nepochs = int(m.group(9))
        self.lr = float(m.group(10))

    def get_id(self, delim='_'):

        id = 'maf' + delim 
        id += 'nmades' + delim + str(self.nmades) + delim
        id += 'dhidden' + delim + str(self.dhidden) + delim
        id += 'nhiddens' + delim + str(self.nhidden)

        return id


class ExperimentDescriptor:

    def __init__(self, str=None):

        self.sim = None
        self.inf = None
        self.str = str

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'experiment\n'
        str += '{\n'
        str += '\tsim: {0},\n'.format(self.sim.pprint())
        str += '\n'
        str += '\tinf: {0}\n'.format(self.inf.pprint())
        str += '}\n'

        return str

    def create_desc(self, sim_desc, inf_desc):

        str = 'experiment\n'
        str += '{\n\t'
        str += sim_desc
        str += ',\n\t'
        str += inf_desc
        str += '\n}'

        return str

    def parse(self, str):
        '''
        Parses the exp_descr string into sim and inf fields.
        '''
        str = util.misc.remove_whitespace(str)
        m = re.match(r'experiment\{sim:(.*),inf:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.sim = SimulatorDescriptor.get_descriptor(m.group(1))
        self.inf = InferenceDescriptor.get_descriptor(m.group(2))

    def get_dir(self):

        return os.path.join(self.sim.get_dir(), self.inf.get_dir())


def parse(str):
    """
    Parses the string str, and returns a list of experiment descriptor objects described by the string.
    """

    str = util.misc.remove_whitespace(str)
    descs = []
    pattern = re.compile(r'experiment\{')
    match = pattern.search(str)

    while match:

        exp_str = match.group()
        left = 1
        right = 0
        i = match.end()

        # consume the string until curly brackets close
        while left > right:

            try:
                if str[i] == '{':
                    left += 1

                if str[i] == '}':
                    right += 1

            # if we have reached the end of the string, discard current match
            except IndexError:
                print('Experiment not compiled. End reached without bracket closing.')
                print('')
                return descs

            exp_str += str[i]
            i += 1

        try:
            desc = ExperimentDescriptor(exp_str)
            descs.append(desc)

        except ParseError as err:
            print('Experiment not compiled. Parse error in:')
            print(err)
            print('')

        str = str[i:]
        match = pattern.search(str)

    return descs
