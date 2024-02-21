# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import sys

import os

# -----------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'

# -----------------------------------------------------------------------------


def print_progress(iteration, max_iteration):
    out = '|' + '#'*iteration + '.'*(max_iteration-iteration) + '|\r'
    sys.stdout.write(out)
    sys.stdout.flush()


def make_filepath(file_name, ext='txt', overwrite=False):
    folder = os.getcwd()
    file_name = file_name.split('.')
    if len(file_name) > 1:
        ext = file_name[1]
    file_name = file_name[0]
    name = os.path.join(folder, file_name)
    path = '{}.{}'.format(name, ext)
    if not overwrite:
        n = 1
        while os.path.exists(path):
            path = '{}_({}).{}'.format(name, n, ext)
            n += 1
    return path


def serial_number(n, digits=3):
    num = str(n)
    ln = len(num)
    s = max(digits, ln)
    num = '0' * (s - ln) + num
    # for i in range(s - ln):
    #     num = '0' + num
    return num
