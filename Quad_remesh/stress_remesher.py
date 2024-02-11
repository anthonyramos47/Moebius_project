# -*- coding: utf-8 -*-


# !/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy import sparse

import scipy
import pickle

import argparse
import json



# -----------------------------------------------------------------------------

from pathlib import Path
import sys
import os 

# Obtain path
path = os.path.dirname(Path(__file__).resolve().parent)
print(path)

import geolab as geo
import geopt

# -----------------------------------------------------------------------------

from orthogonal_mapper import OrthoOptimizer

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Open the file containing the pickled object. The 'rb' parameter denotes 'read binary'
    # Open a file for writing. The 'wb' parameter denotes 'write binary'
   

    # Crear el analizador de argumentos
    parser = argparse.ArgumentParser(description='Get a json file with parameters for the optimization.')
    # Añadir un argumento posicional
    parser.add_argument('json', type=str, help='Json with parameters for the optimization file path')
    parser.add_argument('pkl_name', type=str, help='output file name')
    parser.add_argument('file_name', type=str, help='output file name')


    # Analizar los argumentos de la línea de comandos
    args = parser.parse_args()
    
    
    with open(args.pkl_name, 'rb') as file:
        V, F, t1, t2, _, _, _, _, _, _, _  = pickle.load(file)

    # Open json file
    with open(args.json, 'r') as f:
        parameters = json.load(f)

    weights = parameters["weights_QR"]
    name = args.file_name

    save = 1


    # per face directions
    # D1 = np.loadtxt(dir_data+'tor_D1.dat', delimiter=',', dtype=np.float64)

    # D2 = np.loadtxt(dir_data+'tor_D2.dat', delimiter=',', dtype=np.float64)
    D1 = t1 
    D2 = t2

    # -------------------------------------------------------------------------
    plotter = geo.plotter()

    H = geo.halfedges(F)
    B = geo.faces_centroid(V, H)

    plotter.plot_faces(V, H, opacity=0.8, color='white')
    plotter.plot_edges(V, H, width=2, color='black')
    plotter.plot_vectors(D1, anchor=B, position='center', scale_factor=0.07, color='black')
    plotter.plot_vectors(D2, anchor=B, position='center', scale_factor=0.07, color='black')

    opt = OrthoOptimizer(V, H, D1, D2)
    opt.iterations = parameters["iterations_QR"]
    opt.step = 0.25
    opt.set_weight('mesh_fairness', weights["mesh_fairnes"])
    opt.set_weight('closeness', weights["closeness"])
    opt.optimize()

    if save:
        opt.save_mesh('{}_deformed'.format(name), field_scale=1, save_all=True)
        geo.save_mesh_obj(V, F, '{}_start'.format(name), overwrite=True)

    v1, v2 = opt.vectors()

    V = opt.optimized_vertices()
    #V += np.array([0,0,1])

    B1 = geo.faces_centroid(V, H)
    

    plotter.plot_faces(V, H, color='b')
    plotter.plot_edges(V, H, color='white', width=4)
    plotter.plot_vectors(v1, anchor=B1, position='center', scale_factor=1.6)
    plotter.plot_vectors(v2, anchor=B1, position='center', scale_factor=1.6)
    plotter.show()
