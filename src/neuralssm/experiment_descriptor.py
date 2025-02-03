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


class InferenceDescriptor:

    @staticmethod
    def get_descriptor(str):
        if re.match('rej_abc', str):
            return Rej_ABC_Descriptor(str)

        elif re.match('snl', str):
            return SNL_Descriptor(str)
        
        elif re.match('nde', str):
            return NDE_Descriptor(str)

        else:
            raise ParseError(str)


class ABC_Descriptor(InferenceDescriptor):

    def get_id(self):

        raise NotImplementedError('abstract method')

    def get_dir(self):

        return os.path.join('abc', self.get_id())


class Rej_ABC_Descriptor(ABC_Descriptor):

    def __init__(self, str):

        self.n_samples = None
        self.eps = None
        self.parse(str)

    def pprint(self):

        str = 'rej_abc\n'
        str += '\t{\n'
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\teps: {0}\n'.format(self.eps)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'rej_abc\{n_samples:(.*),eps:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_samples = int(m.group(1))
        self.eps = float(m.group(2))

    def get_id(self, delim='_'):

        id = 'rejabc'
        id += delim + 'samples' + delim + str(self.n_samples)
        id += delim + 'eps' + delim + str(self.eps)

        return id


class MCMC_Descriptor:

    @staticmethod
    def get_descriptor(str):

        if re.match('gauss_metropolis', str):
            return GaussianMetropolisDescriptor(str)

        else:
            raise ParseError(str)


class GaussianMetropolisDescriptor(MCMC_Descriptor):

    def __init__(self, str):

        self.n_samples = None
        self.step = None
        self.parse(str)

    def pprint(self):

        str = 'gauss_metropolis\n'
        str += '\t\t{\n'
        str += '\t\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\t\tstep: {0}\n'.format(self.step)
        str += '\t\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'gauss_metropolis\{n_samples:(.*),step:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_samples = int(m.group(1))
        self.step = float(m.group(2))

    def get_id(self, delim='_'):

        id = 'gaussmetropolis'
        id += delim + 'samples' + delim + str(self.n_samples)
        id += delim + 'step' + delim + str(self.step)

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


class ModelDescriptor:

    @staticmethod
    def get_descriptor(str):

        if re.match('maf', str):
            return MAF_Descriptor(str)

        else:
            raise ParseError(str)


class MAF_Descriptor(ModelDescriptor):

    def __init__(self, str):

        self.n_hiddens = None
        self.act_fun = None
        self.n_comps = None
        self.parse(str)

    def pprint(self):

        str = 'maf\n'
        str += '\t\t{\n'
        str += '\t\t\tn_hiddens: {0},\n'.format(self.n_hiddens)
        str += '\t\t\tact_fun: {0},\n'.format(self.act_fun)
        str += '\t\t\tn_comps: {0}\n'.format(self.n_comps)
        str += '\t\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'maf\{n_hiddens:\[(.*)\],act_fun:(.*),n_comps:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_hiddens = map(int, m.group(1).split(',')) if m.group(1) else []
        self.act_fun = m.group(2)
        self.n_comps = int(m.group(3))

    def get_id(self, delim='_'):

        id = 'maf' + delim + 'hiddens' + delim

        for h in self.n_hiddens:
            id += str(h) + delim

        id += 'comps' + delim + str(self.n_comps) + delim
        id += self.act_fun

        return id


class SNL_Descriptor(InferenceDescriptor):

    def __init__(self, str):

        self.model = None
        self.n_samples = None
        self.n_rounds = None
        self.train_on = None
        self.thin = None
        self.parse(str)

    def pprint(self):

        str = 'snl\n'
        str += '\t{\n'
        str += '\t\tmodel: {0},\n'.format(self.model.pprint())
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\tn_rounds: {0},\n'.format(self.n_rounds)
        str += '\t\ttrain_on: {0},\n'.format(self.train_on)
        str += '\t\tthin: {0}\n'.format(self.thin)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'snl\{model:(.*),n_samples:(.*),n_rounds:(.*),train_on:(all|last),thin:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.model = ModelDescriptor.get_descriptor(m.group(1))
        self.n_samples = int(m.group(2))
        self.n_rounds = int(m.group(3))
        self.train_on = m.group(4)
        self.thin = int(m.group(5))

    def get_dir(self):

        return os.path.join('snl_samples_{0}_rounds_{1}_train_on_{2}_thin_{3}'.format(self.n_samples, self.n_rounds, self.train_on, self.thin), self.model.get_id())
        

class ExperimentDescriptor:

    def __init__(self, str):

        self.sim = None
        self.inf = None
        self.pmeta = None # The idea is to keep all parameter metadata in a separate file
        self.parse(str)

    def pprint(self):

        str = 'experiment\n'
        str += '{\n'
        str += '\tsim: {0},\n'.format(self.sim)
        str += '\tinf: {0}\n'.format(self.inf.pprint())
        str += '}\n'

        return str

    def parse(self, str):
        '''
        Parses the exp_descr string into sim and inf fields.
        '''
        
        str = util.misc.remove_whitespace(str)
        m = re.match(r'experiment\{sim:(mg1|lotka_volterra|gauss|hodgkin_huxley),inf:(.*),param_file:(.*)\}\Z', str)
    
        if m is None:
            raise ParseError(str)

        self.sim = m.group(1)
        self.inf = InferenceDescriptor.get_descriptor(m.group(2))
        param_file = main_dir + '/exps/params/' + m.group(3)
        # read param file (python) and assign variables
        variables = {}
        with open(param_file) as f:
            exec(f.read(), variables)
        self.pmeta = variables['pmeta']

    def get_dir(self):

        return os.path.join(self.sim, self.inf.get_dir())


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
