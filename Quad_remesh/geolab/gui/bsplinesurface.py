#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from tvtk.api import tvtk

import numpy as np

# ------------------------------------------------------------------------------

from geolab.plot.plotmanager import PlotManager

from geolab.bspline.bsplinesurface import BsplineSurface as BSurf

# ------------------------------------------------------------------------------

'''-'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                         B-Spline Surface PlotManager
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class BsplineSurfacePlotManager(PlotManager, BSurf):

    def __init__(self, **kwargs):
        PlotManager.__init__(self, **kwargs)

        self.scale = 1

        self.glyph_scale = 1.5

        self.vector_scale = 1

        self.selected_vertices = []

        self.vertex_color = 'cornflower'

        self.control_color = (10, 10, 10)

        self.surface_color = 'cornflower'

        self.selection_color = 'yellow'

        self.view_mode = 'solid'

        self._show_virtual = False

        self._r = None

        self._g = None

        self._bspline = None

        self._face_sources = []

        self._edge_sources = []

        self._updating = True

        self.__dimensions = (0, 0, 0, 0)

        self.__clear = False

    # -------------------------------------------------------------------------
    #                              Properties
    # -------------------------------------------------------------------------

    @property
    def type(self):
        return 'Bspline_surface_plot_manager'

    @property
    def bspline(self):
        return self._bspline

    @bspline.setter
    def bspline(self, bspline):
        if bspline.type != 'Bspline_Surface':
            raise ValueError('*bspline* attribute must be a B-Spline_Surface!')
        self._bspline = bspline
        self._set_r()
        self._set_g()
        self.vertex_data = np.zeros(bspline.S)

    @property
    def geometry(self):
        return self._bspline

    @geometry.setter
    def geometry(self, bspline):
        self.bspline = bspline

    @property
    def r(self):
        return self._r * self.scale

    @property
    def g(self):
        return self._g * self.glyph_scale

    @property
    def v(self):
        return 12 * self._r * self.vector_scale

    @property
    def updating(self):
        return self._updating

    @updating.setter
    def updating(self, updating):
        if type(updating) == bool:
            self._updating = updating
        else:
            raise ValueError('the updating attribute must be a boolean!')

    @property
    def vertex_data(self):
        if self._vertex_data is None:
            self._vertex_data = np.zeros(self.bspline.S)
        return self._vertex_data

    @vertex_data.setter
    def vertex_data(self, vertex_data):
        self._vertex_data = vertex_data

    @property
    def object_selection_actors(self):
        return self.get_actor('virtual-bspline').actors

    # -------------------------------------------------------------------------
    #                              Size set
    # -------------------------------------------------------------------------

    def update_size(self):
        self._set_r()
        self._set_g()

    def check_updates(self):
        updated = False
        if self.__dimensions[0] != self.bspline.Cu:
            updated = True
        if self.__dimensions[1] != self.bspline.Cv:
            updated = True
        if self.__dimensions[2] != self.bspline.u_sampling:
            updated = True
        if self.__dimensions[3] != self.bspline.v_sampling:
            updated = True
        if updated:
            # self._set_r()
            # self._set_g()
            self._vertex_data = None
            self.__clear = True
        self.__dimensions = (self.bspline.Cu, self.bspline.Cv,
                             self.bspline.u_sampling, self.bspline.v_sampling)
        return updated

    def _set_r(self):
        if self.bspline is None or self.bspline.C == 0:
            self._r = None
        else:
            box = (np.max(self.bspline.control_points, axis=0) -
                   np.min(self.bspline.control_points, axis=0))
            r = np.linalg.norm(box) / 800
            self._r = r

    def _set_g(self):
        if self.bspline is None or self.bspline.C == 0:
            self._g = None
        else:
            g = self.r
            self._g = g

    # -------------------------------------------------------------------------
    #                                Clear
    # -------------------------------------------------------------------------

    def clear(self, delete=True):
        PlotManager.clear(self)
        # self.picker_off()
        # self.remove_widgets()
        # self.selected_vertices = []

    # -------------------------------------------------------------------------
    #                            Plot functions
    # -------------------------------------------------------------------------

    def remove(self, names):
        PlotManager.remove(self, names)
        if type(names) is str:
            names = [names]
        for name in names:
            if name in self._face_sources:
                self._face_sources.remove(name)
            elif name in self._edge_sources:
                self._edge_sources.remove(name)

    def remove_face_sources(self):
        for name in self._face_sources:
            PlotManager.remove(self, name)
        self._face_sources = []

    def remove_edge_sources(self):
        for name in self._edge_sources:
            PlotManager.remove(self, name)
        self._edge_sources = []

    def clear_check(self, name):
        clear = False
        if name not in self.sources:
            clear = True
        if not self.updating:
            clear = True
        if self.__clear:
            clear = True
        return clear

    def start_plot(self):
        PlotManager.start_plot(self)
        self.check_updates()

    def core_plot(self, **kwargs):
        PlotManager.core_plot(self)

    def end_plot(self):
        self.plot_selected_vertices()
        self.__clear = False
        PlotManager.end_plot(self)

    def initialize_plot(self):
        pass
        # self.plot_edges()

    def plot_surface(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'surface'
        else:
            self._face_sources.append(kwargs['name'])
        if 'color' not in kwargs:
            kwargs['color'] = self.surface_color
        if self.clear_check(kwargs['name']):
            self.bspline.plot_mode = 'surface'
            F = Faces(self.bspline, **kwargs)
            self.add(F)
            self._face_sources.append(kwargs['name'])
        else:
            self.bspline.plot_mode = 'surface'
            self.update(**kwargs)

    def plot_control_polygon(self, **kwargs):
        if 'update' not in kwargs:
            kwargs['update'] = True
        if 'name' not in kwargs:
            kwargs['name'] = 'control_polygon'
        else:
            self._edge_sources.append(kwargs['name'])
        if 'tube_radius' not in kwargs:
            if self.view_mode == 'wireframe':
                tube_radius = None
            elif self.view_mode == '3d':
                tube_radius = None
            else:
                tube_radius = self.r
            kwargs['tube_radius'] = tube_radius
        if 'color' not in kwargs:
            kwargs['color'] = self.control_color
        if 'glossy' not in kwargs:
            kwargs['glossy'] = 0.1
        if self.clear_check(kwargs['name']):
            self.bspline.plot_mode = 'control'
            M = Edges(self.bspline, **kwargs)
            self.add(M)
            self._edge_sources.append(kwargs['name'])
        else:
            self.bspline.plot_mode = 'control'
            self.update(**kwargs)

    def plot_vertices(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'vertices'
        if 'radius' not in kwargs:
            kwargs['radius'] = self.r * 2.5
        if 'color' not in kwargs:
            kwargs['color'] = self.vertex_color
        if self.clear_check(kwargs['name']):
            P = Points(self.spline.control_points, **kwargs)
            self.add(P)
        else:
            self.update(**kwargs)

    def plot_glyph(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'glyph'
        if 'points' not in kwargs:
            kwargs['points'] = self.bspline.control_points
        if 'radius' not in kwargs:
            kwargs['radius'] = self.r * 2.5
        if 'color' not in kwargs:
            kwargs['color'] = self.vertex_color
        if self.clear_check(kwargs['name']):
            P = Points(**kwargs)
            self.add(P)
        else:
            self.update(**kwargs)

    def hide_control_polygon(self):
        self.hide('control_polygon')

    def remove_control_polygon(self):
        self.remove('control_polygon')

    def hide_surface(self):
        self.hide('surface')

    def remove_surface(self):
        self.remove('surface')
        for name in self._face_sources:
            self.remove(name)
        self._face_sources = []

    def hide_vertices(self):
        self.hide('vertices')

    def remove_vertices(self):
        self.remove('vertices')

    # -------------------------------------------------------------------------
    #                             Virtual plot
    # -------------------------------------------------------------------------

    def selection_on(self):
        self.bspline.plot_mode = 'control'
        M = Edges(self.bspline,
                  name='virtual-bspline',
                  color=(0.5, 0.5, 0.5),
                  opacity=0.001, )
        self.add(M, pickable=True)
        return self.bspline.E

    def selection_off(self):
        self.remove('virtual-bspline')

    def virtual_vertices_on(self):
        if True:  # self.clear_check('virtual-vertices'):
            opacity = 0.001
            if self._show_virtual:
                opacity = 1
            M = Points(self.bspline.control_points,
                       name='virtual-vertices',
                       glyph_type='cube',
                       radius=self.r * 3,
                       opacity=opacity)
            self.add(M, pickable=True)
        else:
            self.update(name='virtual-vertices', radius=self.r * 1.1)

    def virtual_vertices_off(self):
        self.hide('virtual-vertices')

    # --------------------------------------------------------------------------
    #                            Selection plot
    # --------------------------------------------------------------------------

    def highlight(self):
        self.bspline.plot_mode = 'control'
        tube_radius = self.g * 1.03
        M = Edges(self.bspline,
                  tube_radius=None,
                  line_width=5 * self.scale,
                  color=self.selection_color,
                  shading=False,
                  name='highlight')
        self.hide('control_polygon')
        self.add(M)

    def highlight_off(self):
        self.show('control_polygon')
        self.remove('highlight')

    def plot_selected_vertices(self):
        if len(self.selected_vertices) > 0:
            if self.clear_check('selected-vertices'):
                # if self.view_mode == 'wireframe' or self.view_mode == '3d':
                # glyph = 'wirefrane'
                # radius = None
                # elif self.view_mode == 'solid':
                radius = self.g * 2.52
                glyph = 'sphere'
                P = Points(self.bspline.control_points,
                           vertex_indices=self.selected_vertices,
                           color=self.selection_color,
                           radius=radius,
                           shading=False,
                           glyph_type=glyph,
                           name='selected-vertices',
                           resolution=18)
                self.add(P)
            else:
                self.update(name='selected-vertices',
                            vertex_indices=self.selected_vertices,
                            radius=self.g * 2.52)
        else:
            self.hide('selected-vertices')

    def hide_selected_vertices(self):
        self.hide('selected-vertices')

    # -------------------------------------------------------------------------
    #                               Selection
    # -------------------------------------------------------------------------

    def select_all_vertices(self):
        self.select_off()
        self.virtual_vertices_on()
        self.selected_vertices = list(range(self.bspline.C))
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

    def select_boundary_vertices(self):
        self.select_off()
        self.virtual_vertices_on()
        self.selected_vertices = []

        def v_callback(p_id):
            if p_id == -1:
                return
            v = p_id // 6
            boundaries = self.bspline.boundary_control_curves()
            if len(boundaries) == 4:
                corners = [boundaries[0][0], boundaries[0][-1],
                           boundaries[1][0], boundaries[1][-1]]
            else:
                corners = []
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

    def clear_selection(self):
        self.hide_selected_vertices()
        self.selected_vertices = []
        self.virtual_vertices_off()
        self.remove_widgets()

    def select_off(self):
        self.select_vertices_off()
        self.move_vertex_off()
        self.move_vertices_off()

    # -------------------------------------------------------------------------
    #                               Moving
    # -------------------------------------------------------------------------

    def move_vertex(self, vertex_index, interaction_callback=None,
                    end_callback=None):
        self.remove_widgets()
        S = tvtk.SphereWidget()
        point = self.bspline.control_points[vertex_index]
        S.center = point
        S.radius = self.g * 2
        S.scale = False
        S.representation = 'surface'
        # S.handle_property.color = (0, 0, 0)
        # S.handle_property.opacity = 0.3
        # S.modified()
        P = Points(points=point,
                   radius=self.g * 2.52,
                   color=self.selection_color,
                   shading=False,
                   name='widget-point')
        self.add(P)
        self.virtual_vertices_on()

        def i_callback(obj, event):
            self.virtual_vertices_off()
            c = obj.GetCenter()
            center = np.array([c[0], c[1], c[2]])
            self.bspline.control_points[vertex_index, :] = center
            self.update(name='widget-point', points=center)
            if interaction_callback is not None:
                interaction_callback()

        S.add_observer("InteractionEvent", i_callback)

        def e_callback(obj, event):
            if end_callback is not None:
                end_callback()
            self.virtual_vertices_on()
            point = self.bspline.control_points[vertex_index]
            S.center = point
            self.update(name='widget-point', points=point)

        S.add_observer("EndInteractionEvent", e_callback)

        self.add_widget(S, name='vertex_handler')

    def move_vertex_off(self):
        self.remove_widgets()
        self.hide('widget-point')
        self.virtual_vertices_off()

    def move_vertices(self, interaction_callback=None, start_callback=None,
                      end_callback=None):
        self.select_off()
        self.virtual_vertices_on()

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
        self.select_vertices_off()
        self.picker_off()
