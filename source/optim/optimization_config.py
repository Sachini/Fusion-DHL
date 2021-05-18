import json
import os.path as osp
import sys

import numpy as np

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))

_required_function_params = ('name', 'weight')
_required_variable_params = ('name', 'length')


class BaseParam(object):
    def __init__(self, dictionary=None, **kwargs):
        if dictionary:
            self.add_params(json.loads(dictionary))
        if kwargs:
            self.add_params(kwargs)
        self.change_keys = {}

    def _update_private_var_names(self, values):
        if self._private_variables is not None:
            for var in self._private_variables:
                if var in values:
                    values['_' + str(var)] = values.pop(var)
        return values

    def add_params(self, kwargs):
        self.__dict__.update(self._update_private_var_names(kwargs))


class Function(BaseParam):
    # parameter: name, weight, function, residual, select_param, param
    def __init__(self, feat_dict=None, **kwargs):
        self._private_variables = ['func']
        self.weight = -1
        self.residual = False
        self.select_param = False
        self.param = None
        super(Function, self).__init__(feat_dict, **kwargs)
        assert all(i in self.__dict__ for i in _required_function_params), 'Function: Missing one of {}'.format(
            _required_function_params)
        if self.param is not None:
            self.select_param = True

        print('\tCreated function {}, weight {}'.format(self.name, self.weight))

    def is_active(self):
        return self.weight > 0

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, func):
        self._func = func
        self.residual = hasattr(func, 'loss_func')


class Variable(BaseParam):
    # parameters: name, active, length, default_value, last_value
    # change_key : state
    states = ['active', 'default', 'last']
    state_disp = ['active', 'default', 'last']

    def __init__(self, feat_dict=None, **kwargs):
        self._private_variables = ['default_value', 'last_value']
        self.active = False
        self.default = False
        self.last = False
        self.required = kwargs.get('default_value')
        self._default_value = None
        self._last_value = None
        super(Variable, self).__init__(feat_dict, **kwargs)

        assert all(i in self.__dict__ for i in _required_variable_params), 'Variable: Missing '.format(
            _required_variable_params)

        self._ever_active = self.active
        if self._default_value is None:
            self.default_value = np.zeros(self.length)
        else:
            self.default_value = self._default_value
        if self._last_value is None:
            self.last_value = self._default_value
        print('\tCreated variable {}, length {}, active'.format(self.name, self.length, self.active))

    def set_current_state(self, state, value=True, activate_with_default_vals=False):
        assert state in self.states, 'Invalid state name'
        for i in self.states:
            self.__dict__[i] = False
        if self.required: assert value, 'At least one state should be selected'
        self.__dict__[state] = value
        if state == 'active' and activate_with_default_vals:
            self._last_value = self._default_value
        self._ever_active |= self.active

    def set_state(self, state):
        for i in self.states:
            self.__dict__[i] = False
        if not self.required and state == 'none': return
        assert state in self.states, 'Invalid state name'
        self.__dict__[state] = True

    def get_current_state(self):
        for i in self.states:
            if self.__dict__[i]: return i
        return 'none'

    def get_state_options(self):
        s = self.state.copy()
        if not self.required: s.append('none')
        s.remove(self.get_current_state())
        return s

    @property
    def last_value(self):
        return self._last_value

    @last_value.setter
    def last_value(self, value):
        if isinstance(value, np.ndarray):
            self._last_value = value
        elif isinstance(value, list):
            self._last_value = np.asarray(value)
        elif value is not None:  # scalar
            self._last_value = np.full(self.length, value)

    @property
    def default_value(self):
        return self._default_value

    @default_value.setter
    def default_value(self, value):
        if isinstance(value, np.ndarray):
            self._default_value = value
        elif isinstance(value, list):
            self._default_value = np.asarray(value)
        elif value is not None:  # scalar
            self._default_value = np.full(self.length, value)

    @staticmethod
    def get_var_states(required=True):
        if required:
            return Variable.state_disp
        else:
            my_states = Variable.state_disp.copy()
            my_states.append('none')
            return my_states


def get_active_params(variables):
    # variables is an Ordered Dict
    params = []
    print('Active params:', end=' ')
    for k, v in variables.items():
        if v.active:
            params.append(v.last_value)
            print(k, end=' ')
    print()
    return np.concatenate(params, axis=0)


def get_default_params(variables):
    # variables is an Ordered Dict
    params = []
    print('Active params:', end=' ')
    for k, v in variables.items():
        if v.active:
            params.append(v.default_value)
            print(k, end=' ')
    print()
    return np.concatenate(params, axis=0)


def store_as_last_params(variables, x):
    # variables is an Ordered Dict
    i = 0
    for k, v in variables.items():
        if v.active:
            v.last_value = x[i:i + v.length]
            i += v.length


def print_variable_status(variables):
    print('VARIABLES format: name: status')
    for k, v in variables.items():
        print('\t{}: {}'.format(k, v.get_current_state()), end='')
    print()


def print_residual_status(functors, variables):
    print('RESIDUAL FUNCTIONS format: name: weight, status')
    for k, v in functors.items():
        print('\t{}: {}, {}'.format(k, v.weight,
                                    v.is_active() and variables[v.param].active if v.select_param else v.is_active()),
              end='')
    print()
