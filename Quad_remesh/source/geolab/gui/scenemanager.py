#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from traits.api import on_trait_change

# -----------------------------------------------------------------------------

from geolab.plot.plotmanager import PlotManager

from geolab.gui.mesh import Mesh


# -----------------------------------------------------------------------------

'''plotmanager.py: The scene manager class'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                Scene Manager
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class SceneManager(PlotManager):

    def __init__(self, scene=None):
        PlotManager.__init__(self, scene=scene)

        self._objects = {}

        self._selected_object = None

        self._last_object = None

        self._handler = None

        self._update_callbacks = {'geometry_change': [],
                                  'object_changed': [],
                                  'object_changing': [],
                                  'object_saved': [],
                                  'object_added': []
                                  }

        self._counter = 0

    # -------------------------------------------------------------------------
    #                                Properties
    # -------------------------------------------------------------------------

    @property
    def current_object(self):
        if self._selected_object is None:
            try:
                return self._objects[list(self._objects.keys())[0]]
            except KeyError:
                return None
        try:
            return self._objects[self._selected_object]
        except KeyError:
            self._selected_object = None
            return self.current_object

    @property
    def last_object(self):
        if self._last_object is None:
            try:
                return self._objects[list(self._objects.keys())[0]]
            except KeyError:
                return None
        try:
            return self._objects[self._last_object]
        except KeyError:
            self._last_object = None
            return self.last_object

    @property
    def engine(self):
        return self._scene_model.engine

    @property
    def objects(self):
        return self._objects

    @property
    def freeze_view(self):
        return self._freeze_view

    @freeze_view.setter
    def freeze_view(self, bool):
        self._freeze_view = bool
        for key in self._objects:
            self._objects[key].freeze_view = bool

    # -------------------------------------------------------------------------
    #                                Set scene
    # -------------------------------------------------------------------------

    def set_scene(self, scene):
        PlotManager.set_scene(self, scene)
        for obj in self._objects.values():
            obj.set_scene(scene)

    # -------------------------------------------------------------------------
    #                                Callbacks
    # -------------------------------------------------------------------------

    def add_update_callback(self, callback_type, callback):
        self._update_callbacks[callback_type].append(callback)

    def object_changed(self):
        for callback in self._update_callbacks['object_changed']:
            if callable(callback):
                callback()

    def object_changing(self):
        for callback in self._update_callbacks['object_changing']:
            if callable(callback):
                callback()

    def object_saved(self, file_name):
        for callback in self._update_callbacks['object_saved']:
            if callable(callback):
                try:
                    callback(file_name)
                except TypeError:
                    callback()

    def object_added(self):
        for callback in self._update_callbacks['object_added']:
            if callable(callback):
                callback()

    # -------------------------------------------------------------------------
    #                                Sources
    # -------------------------------------------------------------------------

    def add_geometry(self, obj, **kwargs):
        kwargs['name'] = kwargs.get('name', 'mesh')
        obj.set_scene(self.scene)
        self._objects[kwargs['name']] = obj
        self._last_object = kwargs['name']
        self.object_added()
        if self._selected_object is None:
            self._selected_object = kwargs['name']
            self.object_changed()

    def get_object(self, name):
        return self._objects[name]

    def add_mesh(self, vertices, **kwargs):
        mesh = Mesh(vertices=vertices, **kwargs)
        self.add_geometry(mesh)

    def save(self, name, file_name=None):
        obj = self.get_object(name)
        self.object_saved(obj.save(file_name))

    # -------------------------------------------------------------------------
    #                                Pipeline
    # -------------------------------------------------------------------------

    def update_plot(self):
        PlotManager.update_plot(self)
        for obj in self._objects.values():
            obj.update_plot()

    def initialize_plot(self):
        self.freeze_view = False
        self.update_plot()
        self.update_plot()
        self.freeze_view = True

    def clear_all(self):
        PlotManager.clear_all(self)
        for obj in self._objects.values():
            obj.clear_all()

    def remove_sources(self, names):
        PlotManager.remove_sources(self, names)
        for obj in self._objects.values():
            obj.remove_sources(names)

    def show_object(self, names):
        for key in self._objects:
            if key in names:
                self._objects[key].update_plot()

    def remove_object(self, names):
        for key in self._objects:
            if key in names:
                self._objects[key].clear_all()
                self._objects.pop(key)

    def hide_object(self, names):
        for key in self._objects:
            if key in names:
                self._objects[key].hide_source()

    # -------------------------------------------------------------------------
    #                                Settings
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    #                                Selection
    # -------------------------------------------------------------------------

    def select_object(self):
        if self.current_object is None:
            return
        self.current_object.highlight()
        ranges = {}
        count = 0
        for key in self._objects:
            obj = self._objects[key]
            N = obj.selection_on()
            ranges[key] = [count, count + N]
            count += N

        def picker_callback(picker):
            pick_id = picker.cell_id
            if pick_id >= 0:
                for key in self._objects:
                    try:
                        obj = self._objects[key]
                        if picker.actor in obj.object_selection_actors:
                            if key != self._selected_object:
                                self.current_object.highlight_off()
                                self.object_changing()
                                self._selected_object = key
                                self.object_changed()
                                self.current_object.highlight()
                    except IndexError:
                        pass

        s = self.engine.current_scene
        p = s._mouse_pick_dispatcher.callbacks
        if len(p) == 0:
            s.on_mouse_pick(picker_callback, 'cell')
        else:
            p[0] = (picker_callback, 'cell', 'Left')
        a = s._mouse_pick_dispatcher._active_pickers['cell']
        a.tolerance = self.picker_tolerance

    def select_off(self):
        for key in self._objects:
            obj = self._objects[key]
            obj.select_off()
            try:
                obj.hilight_off()
            except AttributeError:
                pass
        scene = self.engine.current_scene
        p = scene._mouse_pick_dispatcher.callbacks

        def picker_callback(picker):
            return

        for i in range(len(p)):
            p[i] = (picker_callback, 'cell', 'Left')
