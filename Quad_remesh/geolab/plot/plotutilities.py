#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from __future__ import unicode_literals

import numpy as np

from mayavi.core.api import ModuleManager

# -----------------------------------------------------------------------------

from .colorutilities import rgba_lut

from geolab.utilities.arrayutilities import sum_repeated

from geolab.mesh.globalconnectivity import faces_ordered_halfedges, edges

# -----------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------


def face_cell_arrays(faces, halfedges=False):
    if halfedges:
        i = faces_ordered_halfedges(faces)
        vi = faces[i, 0]
        f = faces[i, 1]
        i = np.ones((f.shape[0]), 'i')
        j = np.arange(f.shape[0])
        _, k = np.unique(f, True)
        lengths = sum_repeated(i, f)
        index = j[k]
        cells = np.insert(vi, index, lengths)
        cell_types = lengths - 3
    else:
        N = faces.shape[1]
        f = np.column_stack((np.repeat(N, len(faces)), faces))
        cells = f.flatten('C')
        cell_types = np.repeat(N - 3, len(faces))
    cell_types[np.where(cell_types[:] > 2)[0]] = 2
    return cells, cell_types


def edge_cell_arrays(halfedges):
    v1, v2 = edges(halfedges)
    two = np.repeat(2, len(v1))
    cells = np.ravel(np.column_stack((two, v1, v2)))
    return cells


def polyline_cell_arrays(n, closed=False, offset=0):
    vi = np.arange(n) + offset
    vj = np.roll(vi, -1) + offset
    c = np.repeat(2, n)
    cells = np.vstack((c, vi, vj)).T
    if not closed:
        cells = np.delete(cells, n - 1, axis=0)
    cells = np.ravel(cells)
    return cells


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------


def make_lut_range(lut_range, data_range):
    max_data = data_range[1]
    min_data = data_range[0]
    if max_data == min_data:
        max_data += 1
        min_data -= 1
    if type(lut_range) == str:
        if lut_range == '-:0:+':
            abs_data = max(abs(max_data), abs(min_data))
            lut_range = np.array([-abs_data, abs_data])
        elif lut_range == '0:+':
            lut_range = np.array([0, max_data])
        elif lut_range == '-:0':
            lut_range = np.array([min_data, 0])
        elif lut_range == '-:+':
            lut_range = np.array([min_data, max_data])
    else:
        lut_range = np.array(lut_range)
    return lut_range


def make_lut_table(module, color, opacity, lut_expansion, reverse_lut, **kwargs):
    module.scalar_lut_manager.lut.table = rgba_lut(color, opacity, lut_expansion)
    module.scalar_lut_manager.reverse_lut = reverse_lut


# -----------------------------------------------------------------------------
#                            Make the ModuleManager
# -----------------------------------------------------------------------------


def make_module(color, opacity, lut_range, lut_expansion, reverse_lut,
                data_range, **kwargs):

    module = ModuleManager()
    if data_range is not None:
        module.scalar_lut_manager.use_default_range = False
        max_data = data_range[1]
        min_data = data_range[0]
        if max_data == min_data:
            max_data += 1
            min_data -= 1

        if type(lut_range) == str:

            if lut_range == '-:0:+':
                abs_data = max(abs(max_data), abs(min_data))
                lut_range = np.array([-abs_data, abs_data])

            elif lut_range == '0:+':
                lut_range = np.array([0, max_data])

            elif lut_range == '-:0':
                lut_range = np.array([min_data, 0])

            elif lut_range == '-:+':
                lut_range = np.array([min_data, max_data])

        module.scalar_lut_manager.data_range = lut_range
    else:
        module.scalar_lut_manager.use_default_range = True

    LUT = rgba_lut(color, opacity, lut_expansion)
    if type(LUT) is str:
        module.scalar_lut_manager.lut_mode = LUT
    else:
        module.scalar_lut_manager.lut.table = LUT
    module.scalar_lut_manager.reverse_lut = reverse_lut
    return module


# -----------------------------------------------------------------------------
#                   Make the ModuleManager for Vectors
# -----------------------------------------------------------------------------


def make_vector_module(color, opacity, lut_range, lut_expansion, reverse_lut):

    module = ModuleManager()
    LUT = rgba_lut(color, opacity, lut_expansion)
    if type(LUT) is str:
        module.vector_lut_manager.lut_mode = LUT
    else:
        module.vector_lut_manager.lut.table = LUT
    module.vector_lut_manager.use_default_range = True
    return module
