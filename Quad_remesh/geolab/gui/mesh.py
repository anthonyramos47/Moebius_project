#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from tvtk.api import tvtk

import numpy as np

# ------------------------------------------------------------------------------

from geolab.plot.plotmanager import PlotManager

from geolab.mesh.iomesh import save_mesh_obj, read_mesh_obj

from geolab.mesh.halfedges import halfedges as make_halfedges

from geolab.mesh.globalconnectivity import faces_list

from geolab.mesh.boundary import mesh_corners, boundary_contiguous_vertices

from geolab.mesh.geometry import mean_edge_length

from geolab.mesh.counting import number_of_faces, number_of_vertices, number_of_edges

# ------------------------------------------------------------------------------

'''-'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                               Mesh PlotManager
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class Mesh(PlotManager):

    def __init__(self, **kwargs):
        PlotManager.__init__(self, **kwargs)

        self.name = kwargs.get('name', 'mesh')

        if 'file_name' in kwargs:
            V, F = read_mesh_obj(kwargs.get('file_name'))
            kwargs['vertices'] = V
            kwargs['faces'] = F
            self.name = kwargs.get('file_name').split('.')[0]

        self._vertices = kwargs['vertices']

        self._halfedges = self._make_halfedges(**kwargs)

        self._vertices_0 = np.copy(self._vertices)

        self._halfedges_0 = np.copy(self._halfedges)

        self.corner_tolerance = 0.5

        self.scale = 1

        self.glyph_scale = 1

        self.vector_scale = 1

        self.selected_vertices = []

        self.selected_faces = []

        self.selected_edges = []

        self.selection_color = 'yellow'

        self.view_mode = 'solid'

        self._show_virtual = False

        self._attributes = {}

        self._r = None

        self._g = None

        self._update_reference_size()

    # -------------------------------------------------------------------------
    #                              Properties
    # -------------------------------------------------------------------------

    @property
    def object_type(self):
        return 'mesh'

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        if len(vertices) != len(self._vertices):
            self.clear_all()
        self._vertices = vertices

    @property
    def halfedges(self):
        return self._halfedges

    @halfedges.setter
    def halfedges(self, halfedges):
        self._halfedges = halfedges
        self.clear_all()

    @property
    def faces(self):
        return faces_list(self._halfedges)

    @faces.setter
    def faces(self, faces):
        self.halfedges = self._make_halfedges(faces=faces)

    @property
    def r(self):
        return self._r * self.scale

    @property
    def g(self):
        return self._r * self._g * self.glyph_scale

    @property
    def v(self):
        return 12 * self._r * self.vector_scale

    @property
    def object_selection_actors(self):
        return self.get_actor('virtual-mesh').actors

    @property
    def V(self):
        return len(self.vertices)

    @property
    def F(self):
        return number_of_faces(self.halfedges)

    @property
    def E(self):
        return number_of_edges(self.halfedges)

    @property
    def vertices_0(self):
        return self._vertices_0

    # -------------------------------------------------------------------------
    #                              Make halfedges
    # -------------------------------------------------------------------------

    def _make_halfedges(self, **kwargs):
        halfedges = kwargs.get('halfedges', None)
        if halfedges is None:
            faces = kwargs.get('faces', None)
            if faces is not None:
                halfedges = make_halfedges(faces)
        return halfedges

    # -------------------------------------------------------------------------
    #                              Attributes
    # -------------------------------------------------------------------------

    def set_attribute(self, name, attribute):
        self._attributes[name] = attribute

    def get_attribute(self, name):
        return self._attributes[name]

    # -------------------------------------------------------------------------
    #                                 I/O
    # -------------------------------------------------------------------------

    def open(self, file_name):
        v, f = read_mesh_obj(file_name)
        h = make_halfedges(f)
        self._vertices = v
        self._halfedges = h
        self._vertices_0 = np.copy(v)
        self._halfedges_0 = np.copy(h)

    def save(self, file_name='mesh', overwrite=False):
        save_mesh_obj(self._vertices, self._halfedges,
                      file_name=file_name, overwrite=overwrite)

    def reset_geometry(self):
        self._vertices = np.copy(self._vertices_0)
        self._halfedges = np.copy(self._halfedges_0)

    # -------------------------------------------------------------------------
    #                              Size set
    # -------------------------------------------------------------------------

    def _set_r(self):
        if self._vertices is None:
            self._r = None
        else:
            d = mean_edge_length(self._vertices, self._halfedges)
            r = d / 50 * max(1, np.log10(self.E))
            self._r = r

    def _set_g(self):
        if self._vertices is None:
            self._g = None
        else:
            g = max(1, np.log10(self.E / 100))
            self._g = g

    def _update_reference_size(self):
        self._set_r()
        self._set_g()

    # -------------------------------------------------------------------------
    #                             Functions
    # -------------------------------------------------------------------------

    def corners(self):
        return mesh_corners(self.vertices, self.halfedges, self.corner_tolerance)

    # -------------------------------------------------------------------------
    #                                Clear
    # -------------------------------------------------------------------------

    def clear_all(self, delete=True):
        PlotManager.clear_all(self)
        self.picker_off()
        self.selected_vertices = []
        self.selected_edges = []
        self.selected_faces = []
        self._update_reference_size()

    # -------------------------------------------------------------------------
    #                            Plot functions
    # -------------------------------------------------------------------------

    def plot_faces(self, **kwargs):
        kwargs['name'] = kwargs.get('name', 'faces')
        kwargs['vertices'] = self._vertices
        kwargs['halfedges'] = self._halfedges
        kwargs['smooth'] = kwargs.get('smooth', False)
        PlotManager.plot_faces(self, **kwargs)

    def plot_edges(self, **kwargs):
        kwargs['name'] = kwargs.get('name', 'edges')
        kwargs['vertices'] = self._vertices
        kwargs['halfedges'] = self._halfedges
        kwargs['connectivity'] = self._halfedges
        PlotManager.plot_edges(self, **kwargs)

    def plot_vertices(self, **kwargs):
        kwargs['name'] = kwargs.get('name', 'vertices')
        kwargs['points'] = self._vertices
        PlotManager.plot_points(self, **kwargs)

    def plot_glyph(self, **kwargs):
        kwargs['name'] = kwargs.get('name', 'glyph')
        kwargs['points'] = self._vertices
        PlotManager.plot_points(self, **kwargs)

    def hide_edges(self):
        self.hide_source('edges')

    def remove_edges(self):
        self.remove_sources('edges')

    def hide_faces(self):
        self.hide_source('faces')

    def remove_faces(self):
        self.remove_sources('faces')

    def hide_vertices(self):
        self.hide_source('vertices')

    def remove_vertices(self):
        self.remove_sources('vertices')

    # -------------------------------------------------------------------------
    #                             Virtual plot
    # -------------------------------------------------------------------------

    def selection_on(self):
        self.plot_edges(name='virtual-mesh', color=(0.5, 0.5, 0.5), opacity=0.001)
        self.show_source('virtual-mesh')
        self._set_pickable(True, 'virtual-mesh')
        return self.E

    def selection_off(self):
        self.remove_sources('virtual-mesh')
        self._set_pickable(False)

    def virtual_edges_on(self):
        self.plot_edges(name='virtual-edges', color=(0.5, 0.5, 0.5), opacity=0.001)
        self.show_source('virtual-edges')

    def virtual_edges_off(self):
        self.hide_source('virtual-edges')

    def virtual_faces_on(self):
        self.plot_faces(name='virtual-faces', smooth=True,
                        color=(0.5, 0.5, 0.5), opacity=0.001)
        self.show_source('virtual-faces')

    def virtual_faces_off(self):
        self.hide_source('virtual-faces')

    def virtual_vertices_on(self):
        self.plot_vertices(name='virtual-vertices', glyph_type='cube',
                           radius=self.r, opacity=0.001)
        self.show_source('virtual-vertices')

    def virtual_vertices_off(self):
        self.hide_source('virtual-vertices')

    # -------------------------------------------------------------------------
    #                            Selection plot
    # -------------------------------------------------------------------------

    def highlight(self):
        self.plot_edges(line_width=5 * self.scale,
                        color=self.selection_color, shading=False,
                        name='highlight')
        self.show_source('highlight')

    def highlight_off(self):
        self.remove_sources('highlight')

    def plot_selected_vertices(self):
        if len(self.selected_vertices) > 0:
            if self.view_mode == 'wireframe' or self.view_mode == '3d':
                glyph = 'wireframe'
                radius = None
            else:  # self.view_mode == 'solid':
                radius = self.g * 2.52
                glyph = 'sphere'
            self.plot_vertices(vertex_indices=self.selected_vertices,
                               color=self.selection_color,
                               radius=radius,
                               shading=False,
                               glyph_type=glyph,
                               force_opaque=True,
                               name='selected-vertices',
                               resolution=18)
            self.show_source('selected-vertices')
        else:
            self.hide_source('selected-vertices')

    def plot_selected_edges(self):
        d = np.zeros(self.E)
        d[self.selected_edges] = 1
        self.plot_edges(edge_data=d,
                        line_width=5 * self.scale,
                        color=[(0, 0, 0), self.selection_color],
                        opacity=[0, 1], lut_range='0:+',
                        shading=False, name='selected-edges',
                        force_opaque=True)
        self.show_source('selected-edges')

    def plot_selected_faces(self):
        d = np.zeros(np.max(self.halfedges[:, 5]))
        d[self.selected_faces] = 1
        self.plot_faces(face_data=d, smooth=True,
                        color=[(0, 0, 0), self.selection_color],
                        opacity=[0, 1], lut_range='0:+', shading=False,
                        name='selected-faces', force_opaque=True)
        self.show_source('selected-faces')

    def hide_selected_vertices(self):
        self.hide_source('selected-vertices')

    def hide_selected_edges(self):
        self.hide_source('selected-edges')

    def hide_selected_faces(self):
        self.hide_source('selected-faces')

    # -------------------------------------------------------------------------
    #                               Selection
    # -------------------------------------------------------------------------

    def select_all_vertices(self):
        self.select_off()
        self.virtual_vertices_on()
        self.selected_vertices = list(range(len(self._vertices)))
        self.plot_selected_vertices()

        def callback(p_id):
            if p_id == -1:
                return
            v = p_id // 6
            if v not in self.selected_vertices:
                self.selected_vertices.append(v)
            else:
                self.selected_vertices.remove(v)
            self.plot_selected_vertices()

        self.picker_callback(callback, mode='cell', name='virtual-vertices')

    def select_vertices(self):
        self.select_off()
        self.virtual_vertices_on()
        self.selected_vertices = []

        def callback(p_id):
            if p_id == -1:
                return
            v = p_id // 6
            if v not in self.selected_vertices:
                self.selected_vertices.append(v)
            else:
                self.selected_vertices.remove(v)
            self.plot_selected_vertices()

        self.picker_callback(callback, mode='cell', name='virtual-vertices')

    def on_vertex_selection(self, vertex_callback):
        self.select_off()
        self.virtual_vertices_on()
        self.selected_vertices = []

        def callback(p_id):
            if p_id == -1:
                return
            v = p_id // 6
            self.selected_vertices = [v]
            if vertex_callback is not None:
                vertex_callback(v)
            self.virtual_vertices_on()

        self.picker_callback(callback, mode='cell', name='virtual-vertices')

    def select_vertices_off(self):
        self.virtual_vertices_off()
        self.hide_selected_vertices()
        self.picker_off()
        self.selected_vertices = []

    def select_boundary_vertices(self):
        self.select_off()
        self.virtual_vertices_on()
        self.selected_vertices = []

        def v_callback(p_id):
            if p_id == -1:
                return
            v = p_id // 6
            corners = mesh_corners(self._vertices, self._halfedges, self.corner_tolerance)
            boundaries = boundary_contiguous_vertices(self._halfedges, corners)
            if v not in corners:
                if v not in self.selected_vertices:
                    for boundary in boundaries:
                        if int(v) in boundary:
                            self.selected_vertices.extend(boundary)
                    self.plot_selected_vertices()
                elif v in self.selected_vertices:
                    for boundary in boundaries:
                        if int(v) in boundary:
                            for w in boundary:
                                self.selected_vertices.remove(w)
                    self.plot_selected_vertices()

        self.picker_callback(v_callback, mode='cell', name='virtual-vertices')

    def select_boundary_vertices_off(self):
        self.select_vertices_off()

    def select_edges(self):
        self.select_off()
        self.virtual_edges_on()
        self.selected_edges = []

        def e_callback(cell_id):
            if cell_id == -1:
                return
            e = cell_id
            if e not in self.selected_edges:
                self.selected_edges.append(e)
            else:
                self.selected_edges.remove(e)
            self.plot_selected_edges()

        self.picker_callback(e_callback, mode='cell', name='virtual-edges')

    def on_edge_selection(self, edge_callback):
        self.select_off()
        self.virtual_edges_on()
        self.selected_edges = []

        def e_callback(cell_id):
            if cell_id == -1:
                return
            e = cell_id
            self.selected_edges = [e]
            if edge_callback is not None:
                edge_callback(e)
            self.virtual_edges_on()

        self.picker_callback(e_callback, mode='cell', name='virtual-edges')

    def select_edges_off(self):
        self.virtual_edges_off()
        self.hide_selected_edges()
        self.picker_off()
        self.selected_edges = []

    def select_faces(self):
        self.select_off()
        self.virtual_faces_on()
        self.selected_faces = []

        def f_callback(cell_id):
            if cell_id == -1:
                return
            f = cell_id
            if f not in self.selected_faces:
                self.selected_faces.append(f)
            else:
                self.selected_faces.remove(f)
            self.plot_selected_faces()

        self.picker_callback(f_callback, mode='cell', name='virtual-faces')

    def on_face_selection(self, face_callback):
        self.virtual_faces_on()
        self.selected_faces = []

        def f_callback(cell_id):
            if cell_id == -1:
                return
            f = cell_id
            self.selected_faces = [f]
            if face_callback is not None:
                face_callback(f)
            self.virtual_faces_on()

        self.picker_callback(f_callback, mode='cell', name='virtual-faces')

    def select_faces_off(self):
        self.virtual_faces_off()
        self.hide_selected_faces()
        self.picker_off()
        self.selected_faces = []

    def clear_selection(self):
        self.hide_selected_vertices()
        self.hide_selected_edges()
        self.hide_selected_faces()
        self.selected_vertices = []
        self.selected_edges = []
        self.selected_faces = []
        self.virtual_edges_off()
        self.virtual_vertices_off()
        self.remove_widgets()

    def select_off(self):
        self.select_edges_off()
        self.select_vertices_off()
        self.select_faces_off()
        self.move_vertex_off()
        self.move_vertices_off()

    # -------------------------------------------------------------------------
    #                               Moving
    # -------------------------------------------------------------------------

    def move_vertex(self, vertex_index, interaction_callback=None,
                    end_callback=None):
        self.remove_widgets()
        S = tvtk.SphereWidget()
        point = self._vertices[vertex_index]
        S.center = point
        S.radius = self.g * 2
        S.representation = 'surface'
        self.plot_points(points=point, radius=self.g * 2.52,
                         color=self.selection_color, shading=False,
                         name='widget-point')
        self.show_source('widget-point')
        self.virtual_vertices_on()

        def i_callback(obj, event):
            self.virtual_vertices_off()
            c = obj.GetCenter()
            center = np.array([c[0], c[1], c[2]])
            self._vertices[vertex_index, :] = center
            self.update_source('faces', vertices=self._vertices)
            self.update_source('edges', vertices=self._vertices)
            self.update_source('vertices', points=self._vertices)
            self.update_source('selected_vertices', points=self._vertices)
            self.plot_points(name='widget-point', points=center)
            if interaction_callback is not None:
                interaction_callback()

        S.add_observer("InteractionEvent", i_callback)

        def e_callback(obj, event):
            if end_callback is not None:
                end_callback()
            self.virtual_vertices_on()
            pt = self.vertices[vertex_index]
            S.center = pt
            self.plot_points(name='widget-point', points=pt)

        S.add_observer("EndInteractionEvent", e_callback)
        self.add_widget(S, name='vertex_handler')

    def move_vertex_off(self):
        self.remove_widgets()
        self.hide_source('widget-point')
        self.virtual_vertices_off()

    def move_vertices(self, interaction_callback=None, start_callback=None,
                      end_callback=None):
        self.select_off()
        self.virtual_vertices_on()
        self.virtual_edges_off()

        def v_callback(p_id):
            self.hide_selected_vertices()
            if p_id == -1:
                return
            v = p_id // 6
            self.selected_vertices = [v]
            if start_callback is not None:
                try:
                    start_callback()
                except:
                    pass
            self.move_vertex(v, interaction_callback, end_callback)

        self.picker_callback(v_callback, mode='cell', name='virtual-vertices')

    def move_vertices_off(self):
        self.selected_vertices = []
        self.move_vertex_off()
        self.picker_off()
