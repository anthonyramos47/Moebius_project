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

from pathlib import Path
<<<<<<< HEAD
import os 
=======
>>>>>>> c4b7e4f (Before pull)
import sys
import os

#path_root = Path(__file__).parents[1]
#sys.path.append(str(path_root))
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
print(path)

# Add hananLab to path
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
print(path)
import geolab as geo
import geopt

# -----------------------------------------------------------------------------

from orthogonal_mapper import OrthoOptimizer

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # -------------------------------------------------------------------------

    name = '-'

    save = 1

    # per face directions
    D1 = None

    D2 = None

    # -------------------------------------------------------------------------
    plotter = geo.plotter()

    V, F = geo.read_mesh('{}.obj'.format(name))
    H = geo.halfedges(F)
    B = geo.faces_centroid(V, H)
    opt = OrthoOptimizer(V, H, D1, D2)
    opt.iterations = 200
    opt.step = 0.25
    opt.set_weight('mesh_fairness', 1.2)
    opt.set_weight('closeness', 0.035)
    opt.optimize()

    if save:
        opt.save_mesh('{}_deformed'.format(name), field_scale=3, save_all=True)
        geo.save_mesh_obj(V, F, '{}_start'.format(name), overwrite=True)

    v1, v2 = opt.vectors()

    V = opt.optimized_vertices()
    B1 = geo.faces_centroid(V, H)
    plotter = geo.plotter()
    plotter.plot_edges(V, H)
    plotter.plot_vectors(D1, anchor=B, position='center', scale_factor=0.3, color='cornflower')
    plotter.plot_vectors(D2, anchor=B, position='center', scale_factor=0.3, color='cornflower')
    plotter.plot_edges(V, H, color='r')
    plotter.plot_vectors(v1, anchor=B1, position='center', scale_factor=0.3)
    plotter.plot_vectors(v2, anchor=B1, position='center', scale_factor=0.3)
    plotter.show()
