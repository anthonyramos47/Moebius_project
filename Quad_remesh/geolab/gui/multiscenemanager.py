#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from mayavi.api import Engine

# -----------------------------------------------------------------------------

from geolab.gui.scenemanager import SceneManager

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                            Multiple Scene Manager
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MultiSceneManager(SceneManager):

    def __init__(self):
        SceneManager.__init__(self)

        self._scene_managers = {}

        self._objects_scene = {}

    @property
    def scene_managers(self):
        return self._scene_managers

    @property
    def current_object(self):
        if self._selected_object is None:
            try:
                return self.get_object(self.object_keys[0])
            except KeyError:
                return None
        else:
            return self.get_object(self._selected_object)

    @property
    def last_object(self):
        if self._last_object is None:
            try:
                return self.get_object(self.object_keys[0])
            except KeyError:
                return None
        else:
            return self.get_object(self._last_object)

    @property
    def current_object_type(self):
        if self._selected_object is None:
            try:
                return self.get_object(self.object_keys[0]).object_type
            except:
                return 'none'
        else:
            return self._selected_object

    @property
    def object_keys(self):
        return list(self._objects_scene.keys())

    @property
    def objects(self):
        objects = {}
        for scene in self._scene_managers:
            for key in self._scene_managers[scene].objects:
                objects[key] = self._scene_managers[scene].objects[key]
        return objects

    @property
    def freeze_view(self):
        return self._freeze_view

    @freeze_view.setter
    def freeze_view(self, bool):
        self._freeze_view = bool
        for key in self._scene_managers:
            self.get_scene(key).freeze_view = bool

    # -------------------------------------------------------------------------
    #                                 Plot
    # -------------------------------------------------------------------------

    def update_plot(self, scene=None):
        if scene is None:
            for key in self._scene_managers:
                self.get_scene(key).update_plot()
        else:
            scene = self._format_scene(scene)
            self.get_scene(scene).update_plot()
        SceneManager.update_plot(self)

    def initialize_plot(self):
        self.freeze_view = False
        self.update_plot()
        self.update_plot()
        self.freeze_view = True

    # -------------------------------------------------------------------------
    #                                 Scenes
    # -------------------------------------------------------------------------

    def add_scene(self, scene, name):
        manager = SceneManager(scene)
        self._scene_managers[name] = manager
        # self._scene_register[scene.mayavi_scene] = manager

    def get_scene(self, name):
        return self._scene_managers[name]

    def get_object_scene(self, name):
        scene = self.get_scene(self._objects_scene[name])
        return scene

    def _format_scene(self, scene):
        if type(scene) is dict:
            scene = scene.get('scene', None)
        if scene is None:
            scene = list(self._scene_managers.keys())[0]
        return scene

    # -------------------------------------------------------------------------
    #                                   Add
    # -------------------------------------------------------------------------

    def add_geometry(self, obj, **kwargs):
        name = kwargs.get('name', None)
        scene = kwargs.get('scene', None)
        if name is None:
            name = 'obj_{}'.format(self._counter)
            self._counter += 1
        kwargs['name'] = name
        scene = self._format_scene(scene)
        self.get_scene(scene).add_geometry(obj, **kwargs)
        self._objects_scene[name] = scene
        self._last_object = kwargs['name']
        self.object_added()
        pass

    def add_mesh(self, vertices, **kwargs):
        name = kwargs.get('name', None)
        scene = kwargs.get('scene', None)
        if name is None:
            name = 'obj_{}'.format(self._counter)
            self._counter += 1
        kwargs['name'] = name
        scene = self._format_scene(scene)
        self.get_scene(scene).add_mesh(vertices, **kwargs)
        self._objects_scene[name] = scene
        self._last_object = kwargs['name']
        self.object_added()

    # -------------------------------------------------------------------------
    #                                 Objects
    # -------------------------------------------------------------------------

    def get_object(self, name):
        scene = self.get_scene(self._objects_scene[name])
        obj = scene.get_object(name)
        return obj

    def hide_object(self, names):
        for key in self.scene_managers:
            self.get_scene(key).hide_object(names)

    def remove_object(self, names):
        for key in self.scene_managers:
            self.get_scene(key).hide_object(names)

    def clear_all(self):
        for key in self.scene_managers:
            self.get_scene(key).clear_all()

    # -------------------------------------------------------------------------
    #                                 Scene
    # -------------------------------------------------------------------------

    def set_background(self, color):
        for key in self._scene_managers:
            self.get_scene(key).set_background(color)

    def disable_render(self):
        for key in self.scene_managers:
            self.get_scene(key).disable_render()

    def enable_render(self):
        for key in self.scene_managers:
            self.get_scene(key).enable_render()

    # -------------------------------------------------------------------------
    #                                 Select
    # -------------------------------------------------------------------------

    def select_object(self):
        if self.current_object is None:
            return
        self.current_object.highlight()
        ranges = {}
        count = 0
        for key in self.objects:
            obj = self.objects[key]
            N = obj.selection_on()
            ranges[key] = [count, count + N]
            count += N
        if self._selected_object is None:
            self._selected_object = list(self.objects.keys())[0]

        def picker_callback(picker):
            pick_id = picker.cell_id
            if pick_id >= 0:
                for key in self._objects_scene:
                    try:
                        obj = self.get_object(key)
                        if picker.actor in obj.object_selection_actors:
                            if key != self._selected_object:
                                self.current_object.highlight_off()
                                #self.object_changing()
                                self._selected_object = key
                                #self.object_changed()
                                self.current_object.highlight()
                    except IndexError:
                        pass

        for key in self.scene_managers:
            s = self.scene_managers[key].scene.mayavi_scene
            p = s._mouse_pick_dispatcher.callbacks
            if len(p) == 0:
                s.on_mouse_pick(picker_callback, 'cell')
            else:
                p[0] = (picker_callback, 'cell', 'Left')
            a = s._mouse_pick_dispatcher._active_pickers['cell']
            a.tolerance = self.picker_tolerance

    def select_object_off(self):
        self.select_off()

    def select_off(self):
        for key in self.objects:
            obj = self.objects[key]
            obj.select_off()
            obj.highlight_off()

        def picker_callback(picker):
            return

        for key in self.scene_managers:
            s = self.scene_managers[key].scene.mayavi_scene
            p = s._mouse_pick_dispatcher.callbacks
            for i in range(len(p)):
                p[i] = (picker_callback, 'cell', 'Left')
