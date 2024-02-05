#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from timeit import default_timer as timer

import numpy as np

from scipy import sparse

# -----------------------------------------------------------------------------

from geopt import spsolve

from geopt import spdot

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class GuidedProjection(object):
    epsilon = 0.001

    step = 1

    iterations = 10

    threshold = None

    make_residual = False

    step_control = False

    fairness_reduction = 0

    verbose = True

    _plot_matrix = False

    _weights = {}

    _N = 0

    _X = None

    _X0 = None

    _H = None

    _r = None

    _H0 = None

    _r0 = None

    _K = None

    _s = None

    _K0 = None

    _s0 = None

    _errors = {}

    _values = {}

    _messages = []

    _report = None

    _step = 1

    _counter = 0

    _initialize = True

    _reinitialize = True

    _constant_constraints = []

    _iterative_constraints = []

    _constraints = []

    _residual = None

    _residual_norm = None

    __timer = 0

    __t0 = 0

    __counter = 0

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        if N != self._N:
            self.reinitialize = True
        self._N = N

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X
        self._X0 = np.copy(X)

    @property
    def max_weight(self):
        return max(list(self._weights.values()))

    @property
    def weights(self):
        return self._weights

    @property
    def initialize(self):
        return self._initialize

    @initialize.setter
    def initialize(self, bool):
        self._initialize = bool

    @property
    def reinitialize(self):
        return self._reinitialize

    @reinitialize.setter
    def reinitialize(self, bool):
        self._reinitialize = bool

    @property
    def iteration(self):
        return self._counter

    # --------------------------------------------------------------------------
    #                                 Workfolw
    # --------------------------------------------------------------------------

    def _initialization(self):
        self._reset_counter()
        self.on_initialize()

    def initialization(self):
        self._initialization()
        self.reinitialization()

    def _reinitialization(self):
        self.on_reinitialize()
        self._X = None
        self._X0 = None
        self._residual = None
        self._errors = {}
        self._counter = 0
        self.initialize_unknowns_vector()
        self._make_errors()
        self.make_values()
        if self.verbose:
            # print('  *** Initialized ***')
            print(self._report)

    def reinitialization(self):
        self._preset_weights()
        self._set_dimensions()
        self._reinitialization()

    def _initialize_iteration(self):
        if self.initialize:
            self._initialization()
            self._initialize = False
            self._reinitialize = True
        self._preset_weights()
        self._set_dimensions()
        self.reinitialize_check()
        if self.reinitialize:
            self._reinitialization()
            self._reinitialize = False
        self._constraints = []
        self._constant_constraints = []
        self.__t0 = timer()

    def _pre_iteration_update(self):
        self.balance_weights()
        # self._constraints = self._constant_constraints + []

    def _post_iteration_update(self):
        self.post_iteration_update()
        self._counter += 1
        self.__counter += 1
        self.make_values()

    def _finalize(self):
        t = timer()
        self.last_iteration_time = t - self.__t0
        self.__timer += t - self.__t0
        self._make_X0()
        self._make_errors()
        if self.verbose:
            print(self._report)
        self.on_finalize()

    def _reset_counter(self):
        self._counter = 0
        self.__counter = 0
        self.__timer = 0

    def _make_X0(self):
        X = np.copy(self.X)
        self._X = None
        self.initialize_unknowns_vector()
        self._X = X

    # --------------------------------------------------------------------------
    #                               Initialization
    # --------------------------------------------------------------------------

    def _preset_weights(self):
        self.preset_weights()

    def _set_dimensions(self):
        self.set_dimensions()

    def _make_errors(self):
        self._errors = {}
        self._messages = []
        self.make_messages()
        self.make_errors()
        if self.make_residual:
            self._make_residuals()
        self._make_report()

    # --------------------------------------------------------------------------
    #                                 Overwrite
    # --------------------------------------------------------------------------

    def reinitialize_check(self):
        '''Launch here the function self.reinitialize()'''
        pass

    def on_reinitialize(self):
        pass

    def on_initialize(self):
        pass

    def on_finalize(self):
        pass

    def initialize_unknowns_vector(self):
        if self.verbose:
            pass
            # print('  *** init X ***')
        pass

    def preset_weights(self):
        pass

    def set_dimensions(self):
        pass

    def balance_weights(self):
        pass

    def make_errors(self):
        pass

    def make_messages(self):
        pass

    def make_values(self):
        pass

    def post_iteration_update(self):
        '''Update here objects from the unknowns'''
        pass

    # --------------------------------------------------------------------------
    #                                  Add
    # --------------------------------------------------------------------------

    def add_error(self, name, mean_error, max_error, weight=None):
        self._errors[name] = (mean_error, max_error, weight)

    def add_value(self, name, value):
        self._values[name] = value

    def add_message(self, message):
        self._messages.append(message)

    def add_weight(self, name, default=1):
        self._weights[name] = default

    def add_weights(self, weights):
        self._weights = {**self._weights, **weights}

    # --------------------------------------------------------------------------
    #                                  Get
    # --------------------------------------------------------------------------

    def get_value(self, key):
        try:
            return self._values[key]
        except:
            return None

    def get_error(self, key):
        try:
            return self._errors[key]
        except:
            return None

    def get_weight(self, key):
        if key in self._weights:
            return self._weights[key]
        else:
            return None

    # --------------------------------------------------------------------------
    #                                  Set
    # --------------------------------------------------------------------------

    def set_weight(self, name, value):
        if name in self._weights:
            self._weights[name] = value
        else:
            out = ('<{}> weight does not exists!').format(name)
            raise ValueError(out)

    def set_weights(self, weights):
        for name in weights:
            self.set_weight(name, weights[name])

    # --------------------------------------------------------------------------
    #                                Format
    # --------------------------------------------------------------------------

    def error_string(self, name):
        error = self.get_error(name)
        if error is None:
            return '_'
        string = '{:.4E} | {:.4E}'
        out = (string).format(error[0], error[1])
        return out

    # --------------------------------------------------------------------------
    #                                 Report
    # --------------------------------------------------------------------------

    def _out_error(self, name):
        error = self.get_error(name)
        if error is None:
            error = ('-', '-', '-')
            string = '.{:>22}: {:>11} | {:>10} | {:>10} .\n'
        else:
            string = '.{:>22}: {:>11.4E} | {:>10.4E} | {:>10.4E} .\n'
        out = (string).format(name, *error)
        return out

    def _format_message(self, message):
        out = '. ' + str(message) + '\n'
        return out

    def _split_string(self):
        n = 64
        out = '.' * n + '\n'
        return out

    def _open_string(self):
        n = 64
        out = '-' * n + '\n'
        name = 'ITERATION ' + str(self.__counter)
        out += '.' + ' ' * ((n - len(name) - 1) // 2) + name
        out += ' ' * (n - (n - len(name) - 1) // 2 - len(name) - 2) + '.\n'
        out += '-' * n + '\n'
        name = 'RELATIVE ERROR [mean | max | weight]'
        out += '.' + ' ' * ((n - len(name) - 1) // 2) + name
        out += ' ' * ((n - len(name) - 1) // 2) + '.\n'
        return out

    def _optimization_infos(self):
        n = 64
        line = '. cumulative time = {:.3f}'.format(self.__timer)
        out = line + ' ' * (n - len(line) - 1) + '.\n'
        line = '. step = {}'.format(self._step)
        out += line + ' ' * (n - len(line) - 1) + '.\n'
        line = '. number of variables = {}'.format(self.N)
        out += line + ' ' * (n - len(line) - 1) + '.\n'
        constraints = 0
        for c in self._constraints:
            constraints += c[1]
        line = '. number of constraints = {}'.format(constraints)
        out += line + ' ' * (n - len(line) - 1) + '.\n'
        return out

    def _make_report(self):
        out = self._open_string()
        out += self._split_string()
        for error in self._errors:
            out += self._out_error(error)
        out += self._split_string()
        out += self._optimization_infos()
        out += self._split_string()
        for message in self._messages:
            out += self._format_message(message)
        out += self._split_string()
        self._report = out
        if self.make_residual:
            self._make_residuals_report()
        return out

    def _make_residuals_report(self):
        n = 64
        if self._residual is None:
            return
        line = '. RESIDUALS'
        out = line + ' ' * (n - len(line) - 1) + '.\n'
        out += self._split_string()
        O = 0
        for constraint in self._constant_constraints:
            res = self._residual[O:O + constraint[1]]
            res = np.linalg.norm(res)
            line = '. {} = {:.4E}'.format(constraint[0], res)
            out += line + ' ' * (n - len(line) - 1) + '.\n'
            O += constraint[1]
        for constraint in self._constraints:
            res = self._residual[O:O + constraint[1]]
            res = np.linalg.norm(res)
            line = '. {} = {:.4E}'.format(constraint[0], res)
            out += line + ' ' * (n - len(line) - 1) + '.\n'
            O += constraint[1]
        out += self._split_string()
        self._report += out

    # -------------------------------------------------------------------------
    #                                 Save
    # -------------------------------------------------------------------------

    def save_report(self, file_name):
        try:
            name = ('{}_report.txt').format(file_name)
            txt = open(name, 'w')
            txt.write(self._report)
            txt.close()
        except:
            print('Report not available!')

    # -------------------------------------------------------------------------
    #                                Build
    # -------------------------------------------------------------------------

    def add_iterative_constraint(self, H, r, name='constraint'):
        self._H.append(H)
        self._r = np.hstack((self._r, r))
        self._constraints.append((name, H.shape[0]))

    def add_constant_constraint(self, H, r, name='constraint'):
        self._H0.append(H)
        self._r0 = np.hstack((self._r0, r))
        self._constant_constraints.append((name, H.shape[0]))

    def add_iterative_fairness(self, K, s, name='fairness'):
        self._K.append(K)
        self._s = np.hstack((self._s, s))

    def add_constant_fairness(self, K, s, name='fairness'):
        self._K0.append(K)
        self._s0 = np.hstack((self._s0, s))

    def build_iterative_constraints(self):
        pass

    def build_constant_constraints(self):
        pass

    def build_iterative_fairness(self):
        pass

    def build_constant_fairness(self):
        pass

    def build_regularizer(self):
        self._R = self.epsilon ** 2 * sparse.identity(self.N)

    # -------------------------------------------------------------------------
    #                                Build
    # -------------------------------------------------------------------------

    def _build_constant_matrices(self):
        null = np.zeros([0])
        E = sparse.coo_matrix((null, (null, null)), shape=(0, self.N))
        self._H0 = [E]
        self._K0 = [E]
        self._r0 = np.array([])
        self._s0 = np.array([])
        self.build_constant_constraints()
        self.build_constant_fairness()
        self._H0 = sparse.vstack(self._H0)
        self._K0 = sparse.vstack(self._K0)

    def _build_iterative_matrices(self):
        null = np.zeros([0])
        E = sparse.coo_matrix((null, (null, null)), shape=(0, self.N))
        self._H = [E]
        self._K = [E]
        self._r = np.array([])
        self._s = np.array([])
        self._constraints = []
        self.build_iterative_constraints()
        self.build_iterative_fairness()
        self.build_regularizer()
        self._H = sparse.vstack(self._H)
        self._K = sparse.vstack(self._K)

    def _make_residuals(self):
        if self.iteration == 0:
            self._start_residual()
        H = sparse.vstack([self._H0, self._H])
        r = sparse.hstack([self._r0, self._r]).T
        r = r.toarray()
        X = sparse.csc_matrix([self.X]).transpose()
        self._residual = H.dot(X) - r

    def _start_residual(self):
        self._constant_constraints = []
        self._constraints = []
        self.balance_weights()
        self._build_constant_matrices()
        self._build_iterative_matrices()
        # self._post_iteration_update()

    # --------------------------------------------------------------------------
    #                                Solver
    # --------------------------------------------------------------------------

    def optimize(self):
        self._initialize_iteration()
        self._build_constant_matrices()
        X0 = np.array(self.X)
        if self.threshold != None:
            diff = np.linalg.norm(X0) / X0.shape[0]
        iteration = 0
        stop = False
        while iteration < self.iterations and not stop:
            self._pre_iteration_update()
            self._build_iterative_matrices()
            H = sparse.vstack([self._H0, self._H])
            r = np.array([np.hstack((self._r0, self._r))]).T
            K = sparse.vstack([self._K0, self._K])
            s = np.array([np.hstack((self._s0, self._s))]).T
            K = 1.0 / (10 ** (iteration * self.fairness_reduction)) * K
            s = 1.0 / (10 ** (iteration * self.fairness_reduction)) * s
            X = sparse.csc_matrix([self.X]).transpose()
            H = H.tocsr()
            K = K.tocsr()
            R = self._R.tocsr()
            A = spdot(H.transpose(), H) + spdot(K.transpose(), K) + R
            a = spdot(H.transpose(), r) + spdot(K.transpose(), s) + self.epsilon ** 2 * X
            if self.step_control and self._residual_norm is None:
                res0 = H.dot(X) - r
                res0 = np.linalg.norm(res0)
                self._residual_norm = res0
            X = spsolve(A, a)
            iteration += 1
            Xi = np.array(X)
            lam = self.step + 0
            if self.step_control:
                alt = False
                while not alt:
                    X = lam * Xi + (1 - lam) * X0
                    X = sparse.csc_matrix([X]).transpose()
                    res = H.dot(X) - r
                    res = np.linalg.norm(res)
                    if res < self._residual_norm or lam < 1e-3:
                        self._residual_norm = res
                        alt = True
                        Xi = lam * Xi + (1 - lam) * X0
                    else:
                        lam = 0.5 * lam
                print(lam)
            elif self.step != 1:
                Xi = self.step * Xi + (1 - self.step) * X0
            if self.threshold != None:
                diff = np.linalg.norm(X0 - Xi) / X0.shape[0]
                if diff < self.threshold:
                    stop = True
            self._step = lam
            X0 = np.array(Xi)
            self._X = np.array(Xi)
            self._post_iteration_update()
        self._finalize()
        return X0
