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
import os 
import sys

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

    name = 'test_remeshed_1'

    save = 1

    # per face directions
    D1 = None

    D2 = None

    # -------------------------------------------------------------------------
    plotter = geo.plotter()

    V, F = geo.read_mesh('{}.obj'.format(name))
    H = geo.halfedges(F)

    plotter.plot_faces(V, H, color="cornflower", opacity=0.5)
    plotter.plot_edges(V, H, color='black', width=2)

    print("Edge length: ", np.mean(geo.edges_length(V, H)) )

    B = geo.faces_centroid(V, H)

    # Read directions
    
    D1 = np.loadtxt('tor_D1.dat', delimiter=',', dtype=np.float64)
    D2 = np.loadtxt('tor_D2.dat', delimiter=',', dtype=np.float64)

    opt = OrthoOptimizer(V, H, D1, D2)
    opt.iterations = 100
    opt.step = 0.25
    opt.set_weight('mesh_fairness', 1.2)
    opt.set_weight('closeness', 0.035)
    opt.optimize()

    if save:
        opt.save_mesh('{}_deformed'.format(name), field_scale=1, save_all=True)
        geo.save_mesh_obj(V, F, '{}_start'.format(name), overwrite=True)

    v1, v2 = opt.vectors()

    V = opt.optimized_vertices()
    
    D1 /= np.linalg.norm(D1, axis=1)[:, None]
    D2 /= np.linalg.norm(D2, axis=1)[:, None]


    v1 /= np.linalg.norm(v1, axis=1)[:, None]
    v2 /= np.linalg.norm(v2, axis=1)[:, None]    

    print("orth before:",np.sum(D1 * D2, axis=1)@np.sum(D1 * D2, axis=1)) 
    print("orth after:",np.sum(v1 * v2, axis=1)@np.sum(v1 * v2, axis=1)) 
      
    
    
    plotter.plot_vectors(D1, anchor=B, position='center', scale_factor=0.1, color='black')
    plotter.plot_vectors(D2, anchor=B, position='center', scale_factor=0.1, color='black')
    
    # Reescale V to unit cube
    # max_v = np.max(V, axis=0)
    # min_v = np.min(V, axis=0)
    # V = (V - min_v) / (max_v - min_v)

    B1 = geo.faces_centroid(V, H)

    plotter.plot_faces(V, H, color='white', opacity=0.5)
    plotter.plot_edges(V, H, color='black', width=2)
    plotter.plot_vectors(v1, anchor=B1, position='center', scale_factor=0.01)
    plotter.plot_vectors(v2, anchor=B1, position='center', scale_factor=0.01)
    plotter.show()
