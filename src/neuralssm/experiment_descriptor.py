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

        else:
            raise ParseError(str)


class LGSSM_Descriptor(SimulatorDescriptor):
    
    def __init__(self, str):

        self.state_dim = None
        self.emission_dim = None
        self.input_dim = 0
        self.num_timesteps = None
        self.target_vars = None
        self.parse(str)


    def pprint(self):

        str = 'lgssm\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(self.state_dim)
        str += '\t\temission_dim: {0},\n'.format(self.emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(self.num_timesteps)
        str += '\t\ttarget_vars: {0}\n'.format('_'.join(self.target_vars))
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


class InferenceDescriptor:

    @staticmethod
    def get_descriptor(str):
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

    def __init__(self, str):

        self.n_samples = None
        self.eps_init = None
        self.eps_last = None
        self.eps_decay = None
        self.parse(str)

    def pprint(self):

        str = 'smc_abc\n'
        str += '\t{\n'
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\teps_init: {0},\n'.format(self.eps_init)
        str += '\t\teps_last: {0},\n'.format(self.eps_last)
        str += '\t\teps_decay: {0}\n'.format(self.eps_decay)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'smc_abc\{n_samples:(.*),eps_init:(.*),eps_last:(.*),eps_decay:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_samples = int(m.group(1))
        self.eps_init = float(m.group(2))
        self.eps_last = float(m.group(3))
        self.eps_decay = float(m.group(4))

    def get_id(self, delim='_'):

        id = 'smcabc'
        id += delim + 'samples' + delim + str(self.n_samples)
        id += delim + 'epsinit' + delim + str(self.eps_init)
        id += delim + 'epslast' + delim + str(self.eps_last)
        id += delim + 'epsdecay' + delim + str(self.eps_decay)

        return id


class PRT_MCMC_Descriptor(InferenceDescriptor):

    def get_id(self):

        raise NotImplementedError('abstract method')

    def get_dir(self):

        return os.path.join('mcmc', self.get_id())


class BPF_MCMC_Descriptor(PRT_MCMC_Descriptor):

    def __init__(self, str):

        self.num_prt = None
        self.num_iters = None
        self.mcmc_steps = None
        self.parse(str)

    def pprint(self):

        str = 'bpf_mcmc\n'
        str += '\t{\n'
        str += '\t\tnum_prt: {0},\n'.format(self.num_prt)
        str += '\t\tnum_iters: {0}\n'.format(self.num_iters)
        str += '\t\tmcmc_steps: {0}\n'.format(self.mcmc_steps)
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


class NDE_Descriptor(InferenceDescriptor):

    def __init__(self, str):

        self.model = None
        self.target = None
        self.n_samples = None
        self.parse(str)

    def pprint(self):

        str = 'nde\n'
        str += '\t{\n'
        str += '\t\tmodel: {0},\n'.format(self.model.pprint())
        str += '\t\ttarget: {0},\n'.format(self.target)
        str += '\t\tn_samples: {0}\n'.format(self.n_samples)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'nde\{model:(.*),target:(posterior|likelihood),n_samples:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.model = ModelDescriptor.get_descriptor(m.group(1))
        self.target = m.group(2)
        self.n_samples = int(m.group(3))

    def get_dir(self):

        return os.path.join('nde', '{0}_samples_{1}'.format(self.target, self.n_samples), self.model.get_id())


class SNL_Descriptor(InferenceDescriptor):

    def __init__(self, str):

        self.model = None
        self.n_samples = None
        self.n_rounds = None
        self.train_on = None
        self.mcmc_steps = None
        self.parse(str)

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

    def __init__(self, str):

        self.model = None
        self.n_samples = None
        self.n_rounds = None
        self.lag = None
        self.subsample = None
        self.train_on = None
        self.mcmc_steps = None
        self.parse(str)

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

    def __init__(self, str):
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
        self.parse(str)

    def pprint(self):

        str = 'maf\n'
        str += '\t\t{\n'
        str += '\t\t\tn_mades: {0},\n'.format(self.nmades)
        str += '\t\t\td_hidden: {0},\n'.format(self.dhidden)
        str += '\t\t\tn_hiddens: {0},\n'.format(self.nhidden)
        str += '\t\t\tact_fun: {0},\n'.format(self.act_fun)
        str += '\t\t\trandom_order: {0},\n'.format(self.random_order)
        str += '\t\t\treverse: {0},\n'.format(self.reverse)
        str += '\t\t\tbatch_norm: {0},\n'.format(self.batch_norm)
        str += '\t\t\tdropout: {0},\n'.format(self.dropout)
        str += '\t\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'maf\{n_mades:(.*),d_hidden:(.*),n_hiddens:(.*),act_fun:(relu|tanh|elu),random_order:(.*),reverse:(.*),batch_norm:(.*),dropout:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.nmades = int(m.group(1))
        self.dhidden = int(m.group(2))
        self.nhidden = int(m.group(3))
        self.act_fun = m.group(4)
        self.random_order = bool(m.group(5))
        self.reverse = bool(m.group(6))
        self.batch_norm = bool(m.group(7))
        self.dropout = bool(m.group(8))

    def get_id(self, delim='_'):

        id = 'maf' + delim 
        id += 'nmades' + delim + str(self.nmades) + delim
        id += 'dhidden' + delim + str(self.dhidden) + delim
        id += 'nhiddens' + delim + str(self.nhidden)

        return id


class ExperimentDescriptor:

    def __init__(self, str):

        self.sim = None
        self.inf = None
        self.str = str
        self.parse(str)

    def pprint(self):

        str = 'experiment\n'
        str += '{\n'
        str += '\tsim: {0},\n'.format(self.sim.pprint())
        str += '\n'
        str += '\tinf: {0}\n'.format(self.inf.pprint())
        str += '}\n'

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
