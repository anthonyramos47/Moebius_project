"""
A mesh obj reader
"""

__author__ = 'Davide Pellis'

import numpy as np

# -----------------------------------------------------------------------------

from geolab.utilities.arrayutilities import sum_repeated, normalize

from geolab.mesh.globalconnectivity import faces_edge_vertices

from geolab.mesh.globalconnectivity import vertices_ring_faces

from geolab.mesh.globalconnectivity import edges_faces

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# de Goes et al. 2020
# Discrete Differential Operators on Polygonal Meshes
# -----------------------------------------------------------------------------


def graph_harmonic_relaxation(vertices, halfedges):
    pass