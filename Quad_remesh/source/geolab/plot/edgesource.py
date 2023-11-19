#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from tvtk.api import tvtk

from mayavi.sources.vtk_data_source import VTKDataSource

from mayavi.modules.surface import Surface

from mayavi.filters.tube import Tube

# -----------------------------------------------------------------------------

from geolab.plot.plotutilities import make_module, polyline_cell_arrays, \
    make_lut_range, make_lut_table

from geolab.mesh.globalconnectivity import edges

from geolab.mesh.halfedges import are_halfedges

from geolab.mesh.halfedges import halfedges

# -----------------------------------------------------------------------------

'''edgesource.py: The mesh edges plot source class'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   Edges
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class Edges(object):

    def __init__(self, vertices, connectivity=None, **kwargs):
        # plotutilities.check_arguments(**kwargs)

        if hasattr(vertices, 'vertices'):
            self._vertices = vertices.vertices
        else:
            self._vertices = vertices

        if hasattr(vertices, 'halfedges'):
            self._connectivity = vertices.halfedges
        else:
            self._connectivity = connectivity

        self._data_range = None

        self._edge_data = kwargs.get('edge_data', None)

        self._vertex_data = kwargs.get('vertex_data', None)

        self._color = kwargs.get('color', 'cornflower')

        self._opacity = kwargs.get('opacity', 1)

        self._line_width = kwargs.get('line_width', 1)

        self._tube_radius = kwargs.get('tube_radius', None)

        self._tube_sides = kwargs.get('tube_sides', 10)

        self._lut_range = kwargs.get('lut_range', '-:0:+')

        self._lut_expansion = kwargs.get('lut_expansion', 2)

        self._reverse_lut = kwargs.get('reverse_lut', False)

        self._shading = kwargs.get('shading', True)

        self._glossy = kwargs.get('glossy', 0.3)

        self._force_opaque = kwargs.get('force_opaque', False)

        self._closed = kwargs.get('closed', False)

        self.name = kwargs.get('name', 'edges')

        self._initialize()

    @property
    def type(self):
        return 'Edges-source'

    # -------------------------------------------------------------------------
    #                                 Pipeline
    # -------------------------------------------------------------------------
    # - .source  =   VTKDataSource
    # - ._data    =     \_Poly_data
    # - ._tube    =       None  \_Tube
    # - ._module  =        \_______\_Module_manager
    # - ._surface =                   \__Surface
    # -------------------------------------------------------------------------

    def _initialize(self):

        self._make_data()

        self._make_tube()

        self._module = make_module(self._color,
                                   self._opacity,
                                   self._lut_range,
                                   self._lut_expansion,
                                   self._reverse_lut,
                                   self._data_range)

        self._make_surface()

        self._assemble_pipeline()

        self.on_scene = False

        self.hidden = False

    # -------------------------------------------------------------------------
    #                             Data Structure
    # -------------------------------------------------------------------------

    def _make_data(self):
        if self._connectivity is not None:
            try:
                if self._connectivity.shape[1] == 2:
                    e = self._connectivity
                else:
                    if are_halfedges(self._vertices, self._connectivity):
                        self._connectivity = edges(self._connectivity)
                    else:
                        H = halfedges(self._connectivity)
                        self._connectivity = edges(H)
                    e = self._connectivity
            except AttributeError:
                H = halfedges(self._connectivity)
                self._connectivity = edges(H)
                e = self._connectivity
            E = len(e)
            cells = np.ravel(np.column_stack((np.repeat(2, E), e)))
        else:
            cells = polyline_cell_arrays(len(self._vertices), self._closed)
            N = cells.shape[0]
            cell_array = tvtk.CellArray()
            cell_array.set_cells(N, cells)
            E = len(self._vertices)
            if not self._closed:
                E -= 1
        cell_array = tvtk.CellArray()
        cell_array.set_cells(E, cells)
        self._data = tvtk.PolyData(points=self._vertices, lines=cell_array)
        if self._edge_data is not None:
            scalars = np.array(self._edge_data)
            self._data.cell_data.scalars = self._edge_data
            self._data.cell_data.scalars.name = self.name
            self._data_range = [np.min(scalars), np.max(scalars)]
        elif self._vertex_data is not None:
            scalars = np.array(self._vertex_data)
            self._data.point_data.scalars = self._vertex_data
            self._data.point_data.scalars.name = self.name
            self._data_range = [np.min(scalars), np.max(scalars)]
        else:
            scalars = np.zeros([E])
            self._data.cell_data.scalars = scalars
            self._data.cell_data.scalars.name = self.name
            self._data_range = [0, 0]

    # -------------------------------------------------------------------------
    #                                Surface
    # -------------------------------------------------------------------------

    def _make_surface(self):
        self._surface = Surface()
        if self._line_width is not None:
            self._surface.actor.property.line_width = self._line_width
        if type(self._opacity) == int or type(self._opacity) == float:
            self._surface.actor.property.opacity = self._opacity
        if not self._shading:
            self._surface.actor.actor.property.lighting = False
        self._surface.actor.actor.property.specular = self._glossy
        self._surface.actor.actor.property.specular_power = 21. * self._glossy
        self._surface.actor.actor.force_opaque = self._force_opaque

    # -------------------------------------------------------------------------
    #                                 Tube
    # -------------------------------------------------------------------------

    def _make_tube(self):
        self._tube = None
        if self._tube_radius == 'adaptive':
            self._surface.actor.actor.property.representation = 'wireframe'
            self._surface.actor.actor.property.render_lines_as_tubes = True
        elif self._tube_radius is not None:
            self._tube = Tube()
            self._tube.filter.radius = self._tube_radius
            self._tube.filter.number_of_sides = self._tube_sides

    # -------------------------------------------------------------------------
    #                               Pipeline
    # -------------------------------------------------------------------------

    def _assemble_pipeline(self):
        src = VTKDataSource(data=self._data)
        self._module.add_child(self._surface)
        if self._tube is None:
            src.add_child(self._module)
        else:
            self._tube.add_child(self._module)
            src.add_child(self._tube)
        self.source = src

    # -------------------------------------------------------------------------
    #                                 Update
    # -------------------------------------------------------------------------

    def _update_data(self, **kwargs):
        self._vertices = kwargs.get('vertices', self._vertices)
        self._connectivity = kwargs.get('connectivity', self._connectivity)
        if self._connectivity is not None:
            if self._connectivity.shape[1] == 2:
                e = self._connectivity
            else:
                e = edges(self._connectivity)
            E = len(e)
            cells = np.ravel(np.column_stack((np.repeat(2, E), e)))
        else:
            cells = polyline_cell_arrays(len(self._vertices), self._closed)
            N = cells.shape[0]
            cell_array = tvtk.CellArray()
            cell_array.set_cells(N, cells)
            E = len(self._vertices)
            if not self._closed:
                E -= 1
        if True:
            cell_array = tvtk.CellArray()
            cell_array.set_cells(E, cells)
            points = kwargs.get('vertices', self._vertices)
            cell_array = tvtk.CellArray()
            cell_array.set_cells(E, cells)
            self._data.set(lines=cell_array)
            self._data.set(points=points)
            self._edge_data = None
            self._vertex_data = None
        self._edge_data = kwargs.get('edge_data', self._edge_data)
        self._vertex_data = kwargs.get('vertex_data', self._vertex_data)
        if self._edge_data is not None:
            scalars = np.array(self._edge_data)
            self._data.cell_data.scalars = scalars
            self._data.cell_data.scalars.name = self.name
            self._data_range = [np.min(scalars), np.max(scalars)]
        elif self._vertex_data is not None:
            scalars = np.array(self._vertex_data)
            self._data.point_data.scalars = scalars
            self._data.point_data.scalars.name = self.name
            self._data_range = [np.min(scalars), np.max(scalars)]
        else:
            scalars = np.zeros([E])
            self._data.cell_data.scalars = scalars
            self._data.cell_data.scalars.name = self.name
            self._data_range = [0, 0]
        self._data.modified()

    def _update_lut_range(self, **kwargs):
        self._lut_range = kwargs.get('lut_range', self._lut_range)
        lut_range = make_lut_range(self._lut_range, self._data_range)
        self._module.scalar_lut_manager.data_range = lut_range

    def _update_tube_radius(self, **kwargs):
        previous_tube = self._tube_radius
        self._tube_radius = kwargs.get('tube_radius', self._tube_radius)
        if type(previous_tube) != type(self._tube_radius):
            try:
                self.source.remove()
            except:
                pass
            self._initialize()
        elif self._tube_radius is not None:
            try:
                self._tube.filter.radius = self._tube_radius
            except:
                pass

    def _update_color(self, **kwargs):
        if 'color' in kwargs:
            if kwargs['color'] != self._color:
                self._color = kwargs['color']
                make_lut_table(self._module,
                               self._color,
                               self._opacity,
                               self._lut_expansion,
                               self._reverse_lut)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def update(self, **kwargs):
        self._update_data(**kwargs)
        self._update_lut_range(**kwargs)
        self._update_color(**kwargs)
        self._update_tube_radius(**kwargs)
        self.source.update()

    def kwargs(self):
        kwargs = {'name': self.name,
                  'vertices': np.copy(self.vertices),
                  'edge_data': np.copy(self._edge_data),
                  'vertex_data': np.copy(self._vertex_data)}
        return kwargs

    # -------------------------------------------------------------------------
    # ------------------------------------------------------------------------
