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
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
import geolab as geo
import geopt

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------

    name = '_'

    save = 1

    remap = 1

    serial = 1

    # -------------------------------------------------------------------------

    if remap:

        V0, F0 = geo.read_mesh('{}_start.obj'.format(name))
        V1, F1 = geo.read_mesh('{}_deformed.obj'.format(name))
        Vq, Fq = geo.read_mesh('{}_quad.obj'.format(name))
        H0 = geo.halfedges(F0)
        H1 = geo.halfedges(F1)
        Hq = geo.halfedges(Fq)

        for i in range(4):
            V1, H1 = geo.loop_subdivision(V1, H1)
            V0, H0 = geo.loop_subdivision(V0, H0)

        c = geo.closest_vertices(V1, Vq)



        plotter = geo.plotter()
        plotter.background = (1, 1, 1)
        # plotter.plot_edges(V1, H1, color='b')
        plotter.plot_edges(V0[c], Hq, color='r')
        plotter.plot_edges(V0, H0, color='gray_50', opacity=0.3)
        plotter.show()

        open_name = '{}_final.obj'.format(name)

    if save:

        geo.save_mesh_obj(V0[c], Hq, '{}_final'.format(name), overwrite=True)




