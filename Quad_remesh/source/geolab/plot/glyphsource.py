#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from tvtk.api import tvtk

from mayavi.filters.poly_data_normals import PolyDataNormals

from mayavi.sources.vtk_data_source import VTKDataSource

from mayavi.modules.surface import Surface

import numpy as np

# -----------------------------------------------------------------------------

from geolab.plot import plotutilities

from geolab.plot import colorutilities

from geolab.plot import glyphs

# -----------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Glyph:

    _glyph_type = None

    def __init__(self, geometry, **kwargs):

        self._geometry = geometry

        self._mesh = None

        self._data_range = None

        self._glyph_data = kwargs.get('data', None)

        self._data_scale = kwargs.get('data_scale', False)

        self._color = kwargs.get('color', 'cornflower')

        self._opacity = kwargs.get('opacity', 1)

        self._lut_range = kwargs.get('lut_range', '-:+')

        self._lut_expansion = kwargs.get('lut_expansion', 2)

        self._reverse_lut = kwargs.get('reverse_lut', False)

        self._smooth = kwargs.get('smooth', True)

        self._shading = kwargs.get('shading', True)

        self._line_width = kwargs.get('line_width', 1)

        self._edge_color = kwargs.get('edge_color', 'black')

        self._edge_visibility = kwargs.get('edge_visibility', False)

        self._glossy = kwargs.get('glossy', 0.3)

        self._specular = kwargs.get('specular', 1)

        self._force_opaque = kwargs.get('force_opaque', False)

        self.name = kwargs.get('name', 'glyph')

        self._kwargs = kwargs

        # --------------------------------------------------------------------------
        #                                 Pipeline
        # --------------------------------------------------------------------------
        # - .source  =   VTKDataSource
        # - .data    =     \_Unstructured_grid
        # - .normals =       None  \_Poly_data_normals
        # - .module  =        \_______\_Module_manager
        # - .surface =                   \__Surface
        # --------------------------------------------------------------------------

        self._make_data()

        self._make_normals()

        self._module = plotutilities.make_module(self._color,
                                                 self._opacity,
                                                 self._lut_range,
                                                 self._lut_expansion,
                                                 self._reverse_lut,
                                                 self._data_range)

        self._make_surface()

        self._assemble_pipeline()

        self.on_scene = False

        self.hidden = False

    @property
    def type(self):
        return 'Faces-source'

    # --------------------------------------------------------------------------
    #                               Data Structure
    # --------------------------------------------------------------------------

    def _make_glyph(self):
        pass

    def _make_data(self):
        self._make_glyph()
        F = self._mesh.F
        points = self._mesh.vertices
        cells, types = self._mesh.cell_arrays()
        cell_types = np.array([tvtk.Triangle().cell_type,
                               tvtk.Quad().cell_type,
                               tvtk.Polygon().cell_type])
        cell_types = cell_types[types]
        cells = np.array(cells)
        cell_array = tvtk.CellArray()
        cell_array.set_cells(F, cells)
        self._data = tvtk.UnstructuredGrid(points=points)
        self._data.set_cells(cell_types, [], cell_array)
        if self._glyph_data is not None:
            scalars = np.repeat(self._glyph_data, self._mesh.G)
            self._data_range = [np.min(self._glyph_data),
                                np.max(self._glyph_data)]
        else:
            scalars = np.zeros(len(self._mesh.vertices))
            self._data_range = [-1, 1]
        self._data.point_data.scalars = scalars
        self._data.point_data.scalars.name = self.name

    # -------------------------------------------------------------------------
    #                                Surface
    # -------------------------------------------------------------------------

    def _make_surface(self):
        self._surface = Surface()
        if self._line_width is not None:
            self._surface.actor.property.line_width = self._line_width
        if self._edge_visibility:
            self._surface.actor.property.edge_visibility = True
            edge_color = colorutilities.map_color(self._edge_color)
            self._surface.actor.property.edge_color = edge_color
        if not self._shading:
            self._surface.actor.actor.property.lighting = False
        if type(self._opacity) == int or type(self._opacity) == float:
            if self._opacity < 1:
                self._surface.actor.actor.property.specular = 0.8
                self._surface.actor.actor.force_opaque = self._force_opaque
        self._surface.actor.actor.property.specular = min(self._glossy, 1)
        sp = 21. * self._specular * self._glossy
        self._surface.actor.actor.property.specular_power = sp + 0.001
        if False:
            self._surface.actor.property.edge_visibility = True

    # -------------------------------------------------------------------------
    #                                Normals
    # -------------------------------------------------------------------------

    def _make_normals(self):
        if self._smooth:
            self._normals = PolyDataNormals()
            if self._glyph_type == 'Pipe':
                pass
                # print(dir(self._normals))
        else:
            self._normals = None

    # -------------------------------------------------------------------------
    #                                Pipeline
    # -------------------------------------------------------------------------

    def _assemble_pipeline(self):
        src = VTKDataSource(data=self._data)
        self._module.add_child(self._surface)
        if self._normals is None:
            src.add_child(self._module)
        else:
            self._normals.add_child(self._module)
            src.add_child(self._normals)
        self.source = src

    # -------------------------------------------------------------------------
    #                                 Update
    # -------------------------------------------------------------------------

    def _update_lut_range(self, **kwargs):
        self._lut_range = kwargs.get('lut_range', self._lut_range)
        lut_range = plotutilities.make_lut_range(self._lut_range,
                                                 self._data_range)
        self._module.scalar_lut_manager.data_range = lut_range

    def _update_color(self, **kwargs):
        if 'color' in kwargs:
            color = kwargs['color']
            if color is not self._color:
                self._color = kwargs['color']
                plotutilities.make_lut_table(self._module,
                                             self._color,
                                             self._opacity,
                                             self._lut_expansion,
                                             self._reverse_lut,
                                             points=self._mesh.vertices)

    def _update_data(self, **kwargs):
        self._data = kwargs.get('data', self._data)
        self._geometry = kwargs.get('geometry', self._geometry)
        if self._glyph_type == 'TriMesh':
            self._geometry = kwargs.get('vertices', self._geometry)
        #self._process_inputs()
        F = self._mesh.F
        cells, types = self._mesh.cell_arrays()
        cell_types = np.array([tvtk.Triangle().cell_type,
                               tvtk.Quad().cell_type,
                               tvtk.Polygon().cell_type])
        cell_types = cell_types[types]
        cells = np.array(cells)
        cell_array = tvtk.CellArray()
        cell_array.set_cells(F, cells)
        cell_array.set_cells(self._mesh.F, cells)
        self._data.set(points=self._mesh.vertices)
        self._data.set_cells(cell_types, [], cell_array)
        self._data.modified()

    def _update_glossy(self, **kwargs):
        self._glossy = kwargs.get('glossy', self._glossy)
        self._surface.actor.actor.property.specular = min(self._glossy, 1)
        sp = 21. * self._specular * self._glossy
        self._surface.actor.actor.property.specular_power = sp + 0.001

    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------

    def update(self, **kwargs):
        self._update_data(**kwargs)
        self._update_lut_range(**kwargs)
        self._update_color(**kwargs)
        self._update_glossy(**kwargs)
        self.source.update()

    def kwargs(self):
        kwargs = {}
        kwargs['name'] = self.name
        kwargs['vertices'] = np.copy(self._mesh.vertices)
        kwargs['data'] = np.copy(self._data)
        return kwargs

    # -------------------------------------------------------------------------
    #                                Legend
    # -------------------------------------------------------------------------

    def show_legend(self, **kwargs):
        label = kwargs.get('label', 'faces')
        self._module.scalar_lut_manager.show_legend = True
        self._module.scalar_lut_manager.data_name = label
        # print(dir(self._module.scalar_lut_manager.scalar_bar))

    def hide_legend(self):
        self._module.scalar_lut_manager.show_legend = False


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#                                  Glyps
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class TriMesh(Glyph):
    _glyph_type = 'TriMesh'

    def _make_glyph(self):
        t = self._kwargs.get('triangles', None)
        self._mesh = glyphs.MeshGlyph(self._geometry, t)
        self._glyph_data = self._kwargs.get('vertex_data', None)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Pipe(Glyph):
    _glyph_type = 'Pipe'

    def _make_glyph(self):
        self._mesh = glyphs.Pipe(self._geometry, **self._kwargs)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Arrows(Glyph):
    _glyph_type = 'Arrows'

    def _make_glyph(self):
        self._mesh = glyphs.Arrows(self._geometry, **self._kwargs)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Rings(Glyph):
    _glyph_type = 'Rings'

    def _make_glyph(self):
        self._mesh = glyphs.Rings(self._geometry, **self._kwargs)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Discs(Glyph):
    _glyph_type = 'Discs'

    def _make_glyph(self):
        self._mesh = glyphs.Discs(self._geometry, **self._kwargs)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Cones(Glyph):
    _glyph_type = 'Cones'

    def _make_glyph(self):
        self._mesh = glyphs.Cones(self._geometry, **self._kwargs)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Spheres(Glyph):
    _glyph_type = 'Spheres'

    def _make_glyph(self):
        self._mesh = glyphs.Spheres(self._geometry, **self._kwargs)
