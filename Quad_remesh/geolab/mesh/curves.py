"""
Mesh curve functions
"""

__author__ = 'Davide Pellis'

import numpy as np

from geolab.mesh.globalconnectivity import vertices_ring_vertices

from geolab.mesh.boundary import boundary_contiguous_halfedges

from geolab.mesh.boundary import boundary_vertices

from geolab.mesh.boundary import mesh_corners

from geolab.mesh.boundary import boundary_contiguous_vertices


# -----------------------------------------------------------------------------
#                                 Boundary
# -----------------------------------------------------------------------------


def mesh_curves(halfedges, corners=[], seed=0):
    _, _, valence = vertices_ring_vertices(halfedges, return_lengths=True)
    bv = boundary_vertices(halfedges)
    H = halfedges
    boundaries = boundary_contiguous_halfedges(halfedges, corners)
    done = []
    curves = []
    for boundary in boundaries:
        family = []
        for h in boundary:
            if H[h, 0] not in done:
                curve = [H[h, 0]]
                if valence[H[h, 0]] <= 3:
                    turn = 1 + int(seed)
                else:
                    turn = 2 + int(seed)
                for i in range(turn):
                    h = H[H[h, 4], 2]
                vertex = H[H[h, 4], 0]
                stop = False
                exclude = False
                if vertex in bv:
                    stop = True
                    exclude = True
                while not stop:
                    curve.append(vertex)
                    if vertex in bv:
                        stop = True
                        done.append(vertex)
                    if valence[vertex] <= 4:
                        turn = 1
                    else:
                        turn = 2
                    for i in range(turn):
                        h = H[H[H[h, 2], 4], 2]
                    vertex = H[H[h, 4], 0]
                if not exclude:
                    family.append(curve)
        if len(family) > 0:
            curves.append(family)
    curves.append(boundary_contiguous_vertices(halfedges, corners))
    return curves
