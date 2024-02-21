#!/usr/bin/env python

# -*- coding: utf-8 -*-

import numpy as np

from tvtk.api import tvtk

from mayavi.filters.poly_data_normals import PolyDataNormals

from mayavi.sources.vtk_data_source import VTKDataSource

from mayavi.modules.surface import Surface

from mayavi.modules.iso_surface import IsoSurface

# -----------------------------------------------------------------------------

from geolab.plot import plotutilities

from geolab.plot import colorutilities

from geolab.mesh.halfedges import are_halfedges

# -----------------------------------------------------------------------------

'''facesource.py: The mesh faces plot source class'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                    Faces
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class Faces(object):

    def __init__(self, vertices, connectivity=None, **kwargs):
        #plotutilities.check_arguments(**kwargs)

        if hasattr(vertices, 'vertices'):
            self._vertices = vertices.vertices
            print('0')
        else:
            self._vertices = vertices

        if connectivity is None:
            if hasattr(vertices, 'halfedges'):
                self._connectivity = vertices.halfedges
            elif hasattr(vertices, 'faces'):
                self._connectivity = vertices.halfedges
            elif 'halfedges' in kwargs:
                if kwargs['halfedges'] is not None:
                    self._connectivity = kwargs['halfedges']
            elif 'faces' in kwargs:
                if kwargs['faces'] is not None:
                    self._connectivity = kwargs['faces']
        else:
            self._connectivity = connectivity

        self._data_range = None

        self._face_data = kwargs.get('face_data', None)

        self._vertex_data = kwargs.get('vertex_data', None)

        self._color = kwargs.get('color', 'cornflower')

        self._edge_color = kwargs.get('edge_color', 'black')

        self._opacity = kwargs.get('opacity', 1)

        self._line_width = kwargs.get('line_width', 1)

        self._edge_visibility = kwargs.get('edge_visibility', False)

        self._lut_range = kwargs.get('lut_range', '-:+')

        self._lut_expansion = kwargs.get('lut_expansion', 2)

        self._reverse_lut = kwargs.get('reverse_lut', False)

        self._smooth = kwargs.get('smooth', True)

        self._shading = kwargs.get('shading', True)

        self._glossy = kwargs.get('glossy', 0.3)

        self._specular = kwargs.get('specular', 1)

        self._backface_culling = kwargs.get('backface_culling', False)

        self._iso_surface = kwargs.get('iso_surface', False)

        self._tube_radius = kwargs.get('tube_radius', None)

        self._number_of_contours = kwargs.get('number_of_contours', 50)

        self._force_opaque = kwargs.get('force_opaque', False)

        self._data_points = kwargs.get('data_points', vertices)

        self._iso_points = kwargs.get('iso_points', False)

        self._feature_angle = kwargs.get('feature_angle', 30)

        self.name = kwargs.get('name', 'faces')

        # -------------------------------------------------------------------------
        #                                 Pipeline
        # -------------------------------------------------------------------------
        # - .source  =   VTKDataSource
        # - ._data    =     \_Unstructured_grid
        # - ._normals =       None  \_Poly_data_normals
        # - ._module  =        \_______\_Module_manager
        # - ._surface =                   \__Surface
        # -------------------------------------------------------------------------

        self._make_data()

        self._make_normals()

        self._module = plotutilities.make_module(self._color,
                                                 self._opacity,
                                                 self._lut_range,
                                                 self._lut_expansion,
                                                 self._reverse_lut,
                                                 self._data_range,
                                                 points=self._data_points)

        if self._iso_surface or self._iso_points:
            self._make_iso_surface()
        else:
            self._make_surface()

        self._assemble_pipeline()

        self.on_scene = False

        self.hidden = False

        self._kwargs = kwargs

    @property
    def type(self):
        return 'Faces-source'

    # -------------------------------------------------------------------------
    #                               Data Structure
    # -------------------------------------------------------------------------

    def _make_data(self):
        if are_halfedges(self._vertices, self._connectivity):
            cells, types = plotutilities.face_cell_arrays(self._connectivity, True)
            F = np.max(self._connectivity[:, 1]) + 1
        else:
            cells, types = plotutilities.face_cell_arrays(self._connectivity, False)
            F = len(self._connectivity)
        cell_types = np.array([tvtk.Triangle().cell_type,
                               tvtk.Quad().cell_type,
                               tvtk.Polygon().cell_type])
        cell_types = cell_types[types]
        cells = np.array(cells)
        cell_array = tvtk.CellArray()
        cell_array.set_cells(F, cells)
        self._data = tvtk.UnstructuredGrid(points=self._vertices)
        self._data.set_cells(cell_types, [], cell_array)
        if self._face_data is not None:
            scalars = np.array(self._face_data)
            self._data.cell_data.scalars = scalars
            self._data.cell_data.scalars.name = self.name
            self._data_range = [np.min(scalars), np.max(scalars)]
        elif self._vertex_data is not None:
            scalars = np.array(self._vertex_data)
            self._data.point_data.scalars = scalars
            self._data.point_data.scalars.name = self.name
            self._data_range = [np.min(scalars), np.max(scalars)]
        elif self._color == 'xyz' or self._color == 'uv_wheel':
            scalars = np.arange(len(self._vertices))
            self._data.point_data.scalars = scalars
            self._data.point_data.scalars.name = self.name
            self._data_range = [0, np.max(scalars)]
        else:
            scalars = np.zeros([F])
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
        if self._backface_culling:
            self._surface.actor.actor.property.backface_culling = True
        self._surface.actor.actor.property.specular = min(self._glossy, 1)
        sp = 21. * self._specular * self._glossy
        self._surface.actor.actor.property.specular_power = sp + 0.001

    def _make_iso_surface(self):
        self._surface = IsoSurface()
        if self._line_width is not None:
            self._surface.actor.property.line_width = self._line_width
        self._surface.contour.auto_contours = True
        self._surface.contour.number_of_contours = self._number_of_contours
        if self._tube_radius is not None:
            self._surface.actor.property.render_lines_as_tubes = True
            self._surface.actor.property.edge_visibility = True
            self._surface.actor.property.render_points_as_spheres = True
            self._surface.actor.property.line_width = 1 + self._tube_radius
            self._surface.actor.property.point_size = 1 + self._tube_radius
            if self._iso_surface:
                self._surface.actor.property.representation = 'surface'
            else:
                self._surface.actor.property.representation = 'points'
        color = colorutilities.rgb_float_color(self._color)
        self._surface.actor.property.edge_color = color
        self._surface.actor.property.vertex_color = color
        self._surface.actor.actor.property.specular = min(self._glossy, 1)
        sp = 21. * self._specular * self._glossy
        self._surface.actor.actor.property.specular_power = sp + 0.001

    # -------------------------------------------------------------------------
    #                                Normals
    # -------------------------------------------------------------------------

    def _make_normals(self):
        if self._smooth:
            self._normals = PolyDataNormals()
            self._normals.filter.feature_angle = self._feature_angle
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
        lut_range = plotutilities.make_lut_range(self._lut_range, self._data_range)
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
                                             points=self._vertices)

    def _update_data(self, **kwargs):
        if True:
            kwargs['connectivity'] = self._connectivity
            self._vertices = kwargs.get('vertices', self._vertices)
            self._data.set(points=self._vertices)
            #self._connectivity # kwargs.get('connectivity', self._connectivity) NOT UPDATING!
            if are_halfedges(self._vertices, self._connectivity):
                cells, types = plotutilities.face_cell_arrays(self._connectivity, True)
                F = np.max(self._connectivity[:, 1]) + 1
            else:
                cells, types = plotutilities.face_cell_arrays(self._connectivity, False)
                F = len(self._connectivity)
            cell_types = np.array([tvtk.Triangle().cell_type,
                                   tvtk.Quad().cell_type,
                                   tvtk.Polygon().cell_type])
            cell_types = cell_types[types]
            cells = np.array([cells])
            cell_array = tvtk.CellArray()
            cell_array.set_cells(F, cells)
            self._data.set_cells(cell_types, [], cell_array)
        self._face_data = kwargs.get('face_data', self._face_data)
        self._vertex_data = kwargs.get('vertex_data', self._vertex_data)
        if self._face_data is not None:
            scalars = np.array(self._face_data)
            self._data.cell_data.scalars = scalars
            self._data.cell_data.scalars.name = self.name
            self._data_range = [np.min(scalars), np.max(scalars)]
        elif self._vertex_data is not None:
            scalars = np.array(self._vertex_data)
            self._data.point_data.scalars = scalars
            self._data.point_data.scalars.name = self.name
            self._data_range = [np.min(scalars), np.max(scalars)]
        elif self._color == 'xyz':
            scalars = np.arange(len(self._vertices))
            self._data.point_data.scalars = scalars
            self._data.point_data.scalars.name = self.name
            self._data_range = [np.min(scalars), np.max(scalars)]
        else:
            scalars = np.zeros([F])
            self._data.cell_data.scalars = scalars
            self._data.cell_data.scalars.name = self.name
            self._data_range = [0, 0]
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
        kwargs = self._kwargs
        kwargs['vertices'] = np.copy(self._vertices)
        kwargs['faces'] = np.copy(self._faces)
        kwargs['halfedges'] = np.copy(self._halfedges)
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


def faces(vertices, edges, **kwargs):
    return Faces(vertices, edges, **kwargs)