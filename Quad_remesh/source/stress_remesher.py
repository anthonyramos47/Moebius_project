# -*- coding: utf-8 -*-


# !/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy import sparse

import scipy

# -----------------------------------------------------------------------------

import sys

import geolab as geo

import geopt

# -----------------------------------------------------------------------------

from orthogonal_mapper import OrthoOptimizer

# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # -------------------------------------------------------------------------


    name = '_'

    D1 = None

    D2 = None

    save = 1

    # -------------------------------------------------------------------------

    M = geopt.Mesh(file_name='{}.obj'.format(name))

    opt = OrthoOptimizer(M.vertices, M.halfedges, D1, D2)
    opt.iterations = 200
    opt.step = 0.25
    opt.set_weight('mesh_fairness', 0.3)
    opt.set_weight('closeness', 0.025)
    opt.optimize()

    if save:
        opt.save_mesh('{}_deformed'.format(name), field_scale=3, save_all=True)
        geo.save_mesh_obj(M.vertices, M.halfedges, '{}_start'.format(name), overwrite=True)

    v1, v2 = opt.vectors()

    V = opt.optimized_vertices()
    B1 = geo.faces_centroid(V, M.halfedges)
    plotter = geo.plotter()
    plotter.plot_edges(M)
    plotter.plot_vectors(D1, anchor=B, position='center', scale_factor=0.3, color='cornflower')
    plotter.plot_vectors(D2, anchor=B, position='center', scale_factor=0.3, color='cornflower')
    plotter.plot_edges(V, M.halfedges, color='r')
    plotter.plot_vectors(v1, anchor=B1, position='center', scale_factor=0.3)
    plotter.plot_vectors(v2, anchor=B1, position='center', scale_factor=0.3)
    plotter.show()
