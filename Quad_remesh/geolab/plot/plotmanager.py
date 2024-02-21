# !/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

# -----------------------------------------------------------------------------

from tvtk.api import tvtk

from mayavi.core.api import ModuleManager

from traits.api import on_trait_change

from traits.api import HasTraits

# -----------------------------------------------------------------------------

from geolab.plot.facesource import Faces

from geolab.plot.edgesource import Edges

from geolab.plot.pointsource import Points

from geolab.plot.vectorsource import Vectors

from geolab.plot.glyphsource import Arrows, Spheres, Rings, Pipe, Discs

# -----------------------------------------------------------------------------

'''plotmanager.py: The plot manager class'''

__author__ = 'Davide Pellis'

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                 Sources Manager
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class SourcesManager(HasTraits):

    def __init__(self):
        HasTraits.__init__(self)

        self.rename_sources = True

        self._sources = {}

        self._counter = 0

    @property
    def sources(self):
        return self._sources

    def _get_source_actor(self, source):
        obj = source.source
        while True:
            obj = obj.children[0]
            if isinstance(obj, ModuleManager):
                break
        actor = obj.children[0].actor.actors[0]
        return actor

    def _format_source_name(self, name):
        if self.rename_sources:
            name = '{}_{}'.format(name, self._counter)
            self._counter += 1
        return name

    def _add_source(self, source, **kwargs):
        name = kwargs.pop('name')
        if name not in self._sources:
            src = source(**kwargs)
            self._sources[name] = src
            self._get_source_actor(src).pickable = 0
        else:
            self._sources[name].update(**kwargs)

    # -------------------------------------------------------------------------
    #                                API
    # -------------------------------------------------------------------------

    def sources_list(self):
        return list(self._sources.values())

    def add_sources(self, sources):
        if sources is None:
            return None
        if type(sources) is not list:
            sources = [sources]
        for source in sources:
            if self.rename_sources:
                counter = 0
                while source.name in self._sources:
                    source.name = 'src_{}'.format(counter)
                    counter += 1
            self._sources[source.name] = source

    def clear_sources(self):
        self._sources = {}

    def remove_source(self, name):
        self._sources.pop(name)

    def get_source(self, name):
        try:
            return self._sources[name]
        except KeyError:
            return None

    def update_source(self, name, **kwargs):
        if name in self._sources:
            self._sources[name].update(**kwargs)

    # -------------------------------------------------------------------------
    #                              Plot API
    # -------------------------------------------------------------------------

    def plot_faces(self, vertices, connectivity=None, **kwargs):
        """Plot the faces of a mesh.

        Parameters
        ----------
        vertices : (V, 3) array
            The vertices of the mesh.
        connectivity : (F, n) or (2E, 5) array
            The connectivity of the mesh. It can be an array of faces or halfeges.

        Keyword Parameters
        -------------------
        color : str / tuple / list of str and tuple
            The color of the plotted object. Colors can be specified as:
                - `(r,g,b)` tuples, with `r`, `g`, `b` in the interval [0,1]
                - a keyword color
                - a keyword for a color Lookup Table (LUT)
                - a list of `(r,g,b)` tuples and keyword colors
            Default is `cornflower`. If a keyword for a color Lookup Table
            (LUT) is specified, the points are colored according to a
            `point_data` -> color mapping.

        opacity : float / list of float
            a [0,1] float / a list of [0,1] floats of the same length of colors.
        lut_range : str / list
            A keyword that specifies the range of the color Lookup Table
            (LUT) for data display.
                - `'-:0:+'` for symmetric LUT range, zero is the central color
                - `'0:+'` for positive LUT range, zero is the first color
                - `'-:0'` for negative LUT range, zero is the last color
                - `'-:+'` for min-max LUT range
                - `[min,max]` for user defined LUT range
            Default is `'-:0:+'`.
        face_data: (F,) array of float
            The data associated to the faces.
        vertex_data: (V,) array of float
            The data associated to the vertices.
        """
        kwargs['name'] = kwargs.pop('name', self._format_source_name('faces'))
        kwargs['vertices'] = vertices
        kwargs['connectivity'] = connectivity
        self._add_source(Faces, **kwargs)

    def plot_edges(self, vertices, connectivity=None, **kwargs):
        """Plot the edges of a mesh or of a polyline.

        Parameters
        ----------
        vertices : (V, 3) array
            The vertices of the mesh or of the polyline.

        Optional Parameters
        -------------------
        connectivity : (E, 2) or (2E, 5) array
            The connectivity of the mesh. It can be an array of edges or halfeges.
            Default is None. In this case, a polyline is plotted.

        Keyword Parameters
        -------------------
        closed: bool
            If True, the polyline is closed.
            Default is False.
        color : str / tuple / list of str and tuple
            The color of the plotted object. Colors can be specified as:
                - `(r,g,b)` tuples, with `r`, `g`, `b` in the interval [0,1]
                - a keyword color
                - a keyword for a color Lookup Table (LUT)
                - a list of `(r,g,b)` tuples and keyword colors
            Default is `cornflower`. If a keyword for a color Lookup Table
            (LUT) is specified, the points are colored according to a
            `point_data` -> color mapping.
        opacity : float / list of float
            a [0,1] float / a list of [0,1] floats of the same length of colors.
        lut_range : str / list
            A keyword that specifies the range of the color Lookup Table
            (LUT) for data display.
                - `'-:0:+'` for symmetric LUT range, zero is the central color
                - `'0:+'` for positive LUT range, zero is the first color
                - `'-:0'` for negative LUT range, zero is the last color
                - `'-:+'` for min-max LUT range
                - `[min,max]` for user defined LUT range
            Default is `'-:0:+'`.
        edge_data: (E,) array of float
            The data associated to the edges.
        vertex_data: (V,) array of float
            The data associated to the vertices.
        tube_radius : float
            The radius of the tube used to plot the edges.
            Default is None, in which case the edges are plotted as lines.
        """
        kwargs['name'] = kwargs.pop('name', self._format_source_name('edges'))
        kwargs['vertices'] = vertices
        kwargs['connectivity'] = connectivity
        self._add_source(Edges, **kwargs)

    def plot_points(self, points, **kwargs):
        """Plot points with a glyph.

        Parameters
        ----------
        points : (V, 3) array
            The coordinates of the points

        Keyword Parameters
        -------------------
        radius : float
            The radius of the glyph.
        point_data : np.array, (V,)
            An array containing additional data corresponding to the points.
            The i-th entry of the array is associate with the i-th vertex row.
        vertex_indices : np.array of int, (N,)
            If specified, only the vertices with index contained in this
            array will be plotted.
        color : str / tuple / list of str and tuple
            The color of the plotted object. Colors can be specified as:
                - `(r,g,b)` tuples, with `r`, `g`, `b` in the interval [0,1]
                - a keyword color
                - a keyword for a color Lookup Table (LUT)
                - a list of `(r,g,b)` tuples and keyword colors
            Default is `cornflower`. If a keyword for a color Lookup Table
            (LUT) is specified, the points are colored according to a
            `point_data` -> color mapping.
        opacity : float / list of float
            a [0,1] float / a list of [0,1] floats of the same length of colors
        lut_range : str / list
            A keyword that specifies the range of the color Lookup Table
            (LUT) for data display.
                - `'-:0:+'` for symmetric LUT range, zero is the central color
                - `'0:+'` for positive LUT range, zero is the first color
                - `'-:0'` for negative LUT range, zero is the last color
                - `'-:+'` for min-max LUT range
                - `[min,max]` for user defined LUT range
            Default is `'-:0:+'`.
        glyph_type: str
            Specifies how the points are displayed. It can be
            `sphere`, `cube`, `axes`, `cone`, `wireframe`.
            Default is `sphere`
        """
        kwargs['name'] = kwargs.pop('name', self._format_source_name('points'))
        kwargs['points'] = points
        self._add_source(Points, **kwargs)

    def plot_vectors(self, vectors, **kwargs):
        kwargs['name'] = kwargs.pop('name', self._format_source_name('vectors'))
        kwargs['vectors'] = vectors
        self._add_source(Vectors, **kwargs)

    def plot_arrows(self, vectors, **kwargs):
        kwargs['name'] = kwargs.pop('name', self._format_source_name('arrows'))
        kwargs['geometry'] = vectors
        self._add_source(Arrows, **kwargs)

    def plot_spheres(self, center, **kwargs):
        kwargs['name'] = kwargs.pop('name', self._format_source_name('arrows'))
        kwargs['geometry'] = center
        self._add_source(Spheres, **kwargs)

    def plot_rings(self, frame, **kwargs):
        kwargs['name'] = kwargs.pop('name', self._format_source_name('rings'))
        kwargs['geometry'] = frame
        self._add_source(Rings, **kwargs)

    #def plot_pipe(self, vertices, **kwargs):
        #kwargs['name'] = kwargs.pop('name', self._format_source_name('pipe'))
        #kwargs['geometry'] = vertices
        #self._add_source(Pipe, **kwargs)

    def plot_discs(self, frame, **kwargs):
        kwargs['name'] = kwargs.pop('name', self._format_source_name('discs'))
        kwargs['geometry'] = frame
        self._add_source(Discs, **kwargs)

    def plot_frame(self, frame, colors=('r', 'g', 'b'), **kwargs):
        name = kwargs.pop('name', self._format_source_name('frame'))
        kwargs['name'] = name + 'e1'
        kwargs['anchor'] = frame[0]
        kwargs['position'] = 'tail'
        self.plot_arrows(frame[1], color=colors[0], **kwargs)
        kwargs['name'] = name + 'e2'
        self.plot_arrows(frame[2], color=colors[1], **kwargs)
        kwargs['name'] = name + 'e3'
        self.plot_arrows(frame[3], color=colors[2], **kwargs)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                 Plot Manager
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class PlotManager(SourcesManager):

    def __init__(self, **kwargs):
        SourcesManager.__init__(self)

        self.picker_tolerance = 0.003

        self.updating = False

        self.position = None

        self.background = (1, 1, 1)

        self.magnification = 4

        self._scene = kwargs.get('scene', None)

        self._widgets = {}

        self._counter = 0

        self._freeze_view = True

        self._freeze_position = None

        self._plot_callbacks = {}

    @property
    def scene(self):
        return self._scene

    def set_scene(self, scene):
        self._scene = scene

    @property
    def freeze_view(self):
        return self._freeze_view

    @freeze_view.setter
    def freeze_view(self, bool):
        self._freeze_view = bool

    # -------------------------------------------------------------------------
    #                              Plot Sources
    # -------------------------------------------------------------------------

    def _show_sources(self):
        e = self.scene.engine
        if self._freeze_view:
            self._fix_view()
        self.disable_render()
        for name in self.sources:
            src = self.sources[name]
            if not src.on_scene and not src.hidden:
                try:
                    src.source.remove()
                except ValueError:
                    pass
                e.add_source(src.source, scene=self.scene)
                src.on_scene = True
        if self._freeze_view:
            self._apply_view()
        self.enable_render()

    def initialize_plot(self):
        self._freeze_view = False
        self.update_plot()
        self._freeze_view = True

    def clear_all(self):
        self._fix_view()
        for key in self.sources:
            try:
                self._sources[key].source.remove()
                self._sources[key].on_scene = False
            except ValueError:
                pass
        self.clear_sources()
        self.remove_widgets()
        self._apply_view()

    def hide_source(self, name=None):
        self._fix_view()
        try:
            self.sources[name].source.remove()
            self.sources[name].on_scene = False
            self.sources[name].hidden = True
        except ValueError:
            pass
        except KeyError:
            pass
        self._apply_view()

    def remove_sources(self, names):
        if type(names) is str:
            names = [names]
        self._fix_view()
        for name in names:
            try:
                self.sources[name].source.remove()
                self.sources[name].on_scene = False
                self.sources.pop(name)
            except ValueError:
                pass
            except KeyError:
                pass
        self._apply_view()

    def show_only(self, name):
        for key in self._sources:
            if key not in name:
                try:
                    self.sources[key].remove()
                    self.sources[name].on_scene = False
                    self.sources[name].hidden = True
                except:
                    pass

    def show_source(self, name):
        e = self.scene.engine
        self._fix_view()
        try:
            e.add_source(self.sources[name].source, scene=self.scene)
            self.sources[name].on_scene = True
            self.sources[name].hidden = False
        except KeyError:
            pass
        self._apply_view()

    # -------------------------------------------------------------------------
    #                                Position
    # -------------------------------------------------------------------------

    def get_position(self):
        cc = self.scene.camera
        p = [cc.position[0], cc.position[1], cc.position[2]]
        p.extend([cc.view_up[0], cc.view_up[1], cc.view_up[2]])
        p.extend([cc.focal_point[0], cc.focal_point[1], cc.focal_point[2]])
        p.extend([cc.view_angle])
        p.extend([cc.clipping_range[0], cc.clipping_range[1]])
        p.extend([cc.parallel_projection])
        return p

    def save_position(self, print_position=True):
        self.position = self.get_position()
        if print_position:
            print('position = ' + str(self.position))

    def set_position(self, position=None):
        if position is not None:
            p = position
        else:
            p = self.position
        cc = self.scene.camera
        if p is not None:
            cc.position = np.array([p[0], p[1], p[2]])
            cc.view_up = np.array([p[3], p[4], p[5]])
            cc.focal_point = np.array([p[6], p[7], p[8]])
            cc.view_angle = float(p[9])
            cc.clipping_range = np.array([p[10], p[11]])
            try:
                cc.parallel_projection = p[12]
            except:
                pass
            self.scene.mayavi_scene.render()

    def _fix_view(self):
        try:
            self.scene.disable_render = True
            p = self.get_position()
            self._freeze_position = p
        except AttributeError:
            pass

    def _apply_view(self):
        try:
            self.set_position(self._freeze_position)
            self.scene.disable_render = False
        except AttributeError:
            pass

    # -------------------------------------------------------------------------
    #                                View
    # -------------------------------------------------------------------------

    def parallel_projection(self, parallel=True):
        self.scene.camera.parallel_projection = parallel

    def vertical_view(self):
        position = np.array(self.get_position())
        position[3:6] = np.array([0, 0, 1])
        self.set_position(position)

    def camera_rotation(self, degrees):
        center = self.position[6:9]
        position = np.array(self.get_position())
        P = position[0:2]
        F = position[6:8]
        cos = np.cos(np.radians(degrees))
        sin = np.sin(np.radians(degrees))
        V = P - center[0:2]
        Vx = V[0] * cos - V[1] * sin + center[0]
        Vy = V[0] * sin + V[1] * cos + center[1]
        position[0] = Vx
        position[1] = Vy
        V = F - center[0:2]
        Vx = V[0] * cos - V[1] * sin + center[0]
        Vy = V[0] * sin + V[1] * cos + center[1]
        position[6] = Vx
        position[7] = Vy
        position[3:6] = np.array([0, 0, 1])
        self.set_position(position)

    # -------------------------------------------------------------------------
    #                             Scene settings
    # -------------------------------------------------------------------------

    def set_background(self, color):
        self.scene.background = color

    def set_magnification(self, magnification):
        self.scene.magnification = magnification

    # -------------------------------------------------------------------------
    #                                Interaction
    # -------------------------------------------------------------------------

    def interaction_2d(self):
        self.scene.interactor.interactor_style = tvtk.InteractorStyleImage()

    def interaction_locked(self):
        self.scene.interactor.interactor_style = None

    # -------------------------------------------------------------------------
    #                               Widgets
    # -------------------------------------------------------------------------

    def add_widget(self, widget, name='widget'):
        self._widgets[name] = widget
        self.scene.add_actors(widget)

    def get_widget(self, name='widget'):
        try:
            return self._widgets[name]
        except KeyError:
            return None

    def remove_widgets(self):
        for key in self._widgets:
            try:
                self.scene.remove_actor(self._widgets[key])
            except:
                pass
        self._widgets = {}

    # -------------------------------------------------------------------------
    #                                 Render
    # -------------------------------------------------------------------------

    def disable_render(self):
        try:
            self.scene.disable_render = True
        except:
            pass

    def enable_render(self):
        try:
            self.scene.disable_render = False
        except:
            pass

    def render(self):
        self.scene._renwin.render()

    # -------------------------------------------------------------------------
    #                            Get from pipeline
    # -------------------------------------------------------------------------

    def get_actor(self, name):
        stop = False
        try:
            obj = self._sources[name].source
        except KeyError:
            return
        while not stop:
            obj = obj.children[0]
            if isinstance(obj, ModuleManager):
                stop = True
        actor = obj.children[0].actor
        return actor

    # -------------------------------------------------------------------------
    #                                 Picker
    # -------------------------------------------------------------------------

    def _set_pickable(self, pick=True, name=None):
        if name is None:
            for key in self.sources:
                obj = self.sources[key].source
                if pick:
                    pick = 1
                else:
                    pick = 0
                stop = False
                while not stop:
                    obj = obj.children[0]
                    if isinstance(obj, ModuleManager):
                        stop = True
                actor = obj.children[0].actor.actors[0]
                actor.pickable = pick
        else:
            try:
                obj = self.sources[name].source
            except:
                return
            if pick:
                pick = 1
            else:
                pick = 0
            stop = False
            while not stop:
                obj = obj.children[0]
                if isinstance(obj, ModuleManager):
                    stop = True
            actor = obj.children[0].actor.actors[0]
            actor.pickable = pick

    def picker_callback(self, routine, mode='point', name=None, add=False):
        if name is not None:
            self._set_pickable(False)
            self._set_pickable(True, name)
        else:
            self._set_pickable(True)

        def picker_callback(picker):
            pick_id = picker.cell_id
            if mode == 'point':
                pick_id = picker.cell_id
            routine(pick_id)

        # s = self.scene_model.engine.current_scene
        s = self.scene.mayavi_scene
        p = s._mouse_pick_dispatcher.callbacks
        if add:
            s.on_mouse_pick(picker_callback, 'cell')
        elif len(p) == 0:
            s.on_mouse_pick(picker_callback, 'cell')
        else:
            p[0] = (picker_callback, 'cell', 'Left')
        a = s._mouse_pick_dispatcher._active_pickers['cell']
        a.tolerance = self.picker_tolerance

    def picker_off(self):
        # scene = self.scene_model.engine.current_scene
        p = self.scene.mayavi_scene._mouse_pick_dispatcher.callbacks

        def picker_callback(picker):
            return

        for i in range(len(p)):
            p[i] = (picker_callback, 'cell', 'Left')

    def iterate(self, function, times=1):
        for i in range(times):
            function()
            self.scene._renwin.render()

    # -------------------------------------------------------------------------
    #                                 Plot
    # -------------------------------------------------------------------------

    def update_plot(self):
        self._show_sources()
        for key in self._plot_callbacks:
            callback = self._plot_callbacks[key]
            if callable(callback):
                callback()

    def add_plot_callback(self, callback, name=None):
        if callable(callback):
            self._plot_callbacks[name] = callback

    def remove_plot_callback(self, name):
        try:
            self._plot_callbacks.pop(name)
        except KeyError:
            pass
