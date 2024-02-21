#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from traits.api import HasTraits, Instance, Property, Enum, Button, String, \
    on_trait_change, Float, Bool, Int, Constant, ReadOnly, \
    List, Array, Range, Event

from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor, HGroup, \
    Group, ListEditor, Tabbed, VGroup, CheckListEditor, \
    ArrayEditor, Action, ToolBar, Separator, Controller

from tvtk.pyface.scene_editor import SceneEditor

from mayavi.tools.mlab_scene_model import MlabSceneModel

from pyface.image_resource import ImageResource

from pyface.api import FileDialog, OK

from tvtk.pyface.api import Scene

# ------------------------------------------------------------------------------

from geolab.gui.scenemanager import SceneManager

from geolab.gui.multiscenemanager import MultiSceneManager

from geolab.plot.geolabscene import GeolabScene, GeolabSecondaryScene, \
    GeolabBaseScene

from geolab.gui.geolabcomponent import GeolabComponent

from geolab.gui.tools import SaveGeometry, CornerTolerance

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                     Handler
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class GlHandler(Controller):

    def close(self, info, is_ok):
        info.object._closing()
        info.object._closed = True
        Controller.close(self, info, is_ok)
        return True

    # --------------------------------------------------------------------------
    #                                  Tools
    # --------------------------------------------------------------------------

    def open_save_geometry(self):
        self.info.object.open_save_geometry()

    # def open_import(self):
    # self.info.object.open_import()

    def open_corner_tolerance(self):
        self.info.object.open_corner_tolerance()

    # --------------------------------------------------------------------------
    #                                 Selection
    # --------------------------------------------------------------------------

    def background_switch(self):
        self.info.object.background_switch()

    def select_object(self):
        self.info.object.select_object()

    def select_vertices(self):
        self.info.object.select_vertices()

    def select_edges(self):
        self.info.object.select_edges()

    def select_faces(self):
        self.info.object.select_faces()

    def select_boundary_vertices(self):
        self.info.object.select_boundary_vertices()

    def select_all_vertices(self):
        self.info.object.select_all_vertices()

    def move_vertices(self):
        self.info.object.move_vertices()

    # --------------------------------------------------------------------------
    #                                 Mesh state
    # --------------------------------------------------------------------------

    def reset_geometry(self):
        self.info.object.reset_geometry()

    def delete_selected_object(self):
        self.info.object.delete_selected_object()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                       The Graphic User Interface
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class GeolabGUI(MultiSceneManager):
    _handler = GlHandler()

    _closed = Bool(True)

    _current_object = String('none')

    _components = []

    scene_0 = Instance(MlabSceneModel, ())

    scene_1 = Instance(MlabSceneModel, ())

    scene_2 = Instance(MlabSceneModel, ())

    scene_3 = Instance(MlabSceneModel, ())

    components = List(GeolabComponent)

    save_geometry_tool = SaveGeometry()

    corner_tolerance_tool = CornerTolerance()

    _background_switch_counter = Int(1)

    background_switch_button = Action(action='background_switch',
                                      image=ImageResource('../icons/background.png'),
                                      style='push',
                                      tooltip='Switch background color',
                                      show_label=False)

    save_geometry_button = Action(action='open_save_geometry',
                                  image=ImageResource('../icons/save.png'),
                                  style='push',
                                  enabled_when=('current_object_type == \
                                          "mesh" \
                                          or current_object_type == \
                                          "bspline" \
                                          or current_object_type == \
                                          "points"'),
                                  tooltip='Save geometry',
                                  show_label=False)

    # import_button = Action(action='open_import',
    #                        image=ImageResource('../icons/open.png'),
    #                        style='push',
    #                        tooltip='Import geometry',
    #                        show_label=False)

    corner_tolerance_button = Action(action='open_corner_tolerance',
                                     image=ImageResource('../icons/corners.png'),
                                     style='push',
                                     tooltip='Set corner tolerance',
                                     enabled_when=('current_object_type == \
                                              "mesh"'),
                                     show_label=False)

    select_object_button = Action(action='select_object',
                                  image=ImageResource('../icons/selectobject.png'),
                                  style='toggle',
                                  tooltip='Select object',
                                  show_label=False)

    move_vertices_button = Action(action='move_vertices',
                                  image=ImageResource('../icons/movevertex.png'),
                                  style='toggle',
                                  tooltip='Move vertices',
                                  enabled_when=('current_object_type == \
                                           "mesh" \
                                           or current_object_type == \
                                           "bspline" \
                                           or current_object_type == \
                                           "points"'),
                                  show_label=False)

    select_vertices_button = Action(action='select_vertices',
                                    image=ImageResource('../icons/selectvertices.png'),
                                    style='toggle',
                                    tooltip='Select vertices',
                                    enabled_when=('current_object_type == \
                                             "mesh" \
                                             or current_object_type == \
                                             "bspline" \
                                             or current_object_type == \
                                             "points"'),
                                    show_label=False)

    select_edges_button = Action(action='select_edges',
                                 image=ImageResource('../icons/selectedges.png'),
                                 style='toggle',
                                 enabled_when=('current_object_type=="mesh"'),
                                 tooltip='Select edges',
                                 show_label=False)

    select_faces_button = Action(action='select_faces',
                                 image=ImageResource('../icons/selectfaces.png'),
                                 style='toggle',
                                 enabled_when=('current_object_type == \
                                         "mesh"'),
                                 tooltip='Select faces',
                                 show_label=False)

    select_boundary_vertices_button = Action(action='select_boundary_vertices',
                                             image=ImageResource('../icons/boundary.png'),
                                             style='toggle',
                                             enabled_when=('current_object_type == \
                                                  "mesh"\
                                                   current_object_type == \
                                                  "bspline"'),
                                             tooltip='Select boundary vertices',
                                             show_label=False)

    select_all_vertices_button = Action(action='select_all_vertices',
                                        image=ImageResource('../icons/allvertices.png'),
                                        style='toggle',
                                        enabled_when=('current_object_type == \
                                                "mesh" \
                                                or current_object_type == \
                                                "bspline"\
                                                or current_object_type == \
                                                "points"'),
                                        tooltip='Select all vertices',
                                        show_label=False)

    reset_geometry_button = Action(action='reset_geometry',
                                   image=ImageResource('../icons/resetmesh.png'),
                                   style='push',
                                   enabled_when=('current_object_type == \
                                          "mesh" \
                                          or current_object_type == \
                                          "bspline" \
                                          or current_object_type == \
                                          "points"'),
                                   tooltip='Reset geometry',
                                   show_label=False)

    delete_selected_object_button = Action(action='delete_selected_object',
                                           image=ImageResource('../icons/delobject.png'),
                                           style='push',
                                           enabled_when=('_current_object == \
                                          "mesh" \
                                          or current_object_type == \
                                          "bspline" \
                                          or current_object_type == \
                                          "points"'),
                                           tooltip='Delete object',
                                           show_label=False)

    __toolbar = [Separator(),

                 select_object_button,
                 select_vertices_button,
                 select_edges_button,
                 select_faces_button,
                 move_vertices_button,
                 select_all_vertices_button,
                 select_boundary_vertices_button,
                 corner_tolerance_button,

                 Separator(),

                 reset_geometry_button,
                 delete_selected_object_button,

                 Separator(),

                 # import_button,
                 save_geometry_button,

                 Separator(),
                 background_switch_button,
                 ]

    toolbar = {'handler': _handler,
               'resizable': True,
               'title': 'GeoLab',
               'icon': ImageResource('../icons/geolab_logo.png'),
               'toolbar': ToolBar(*__toolbar,
                                  show_labels=False,
                                  image_size=(16, 16)),
               }

    tabs = [Item('components',
                 editor=ListEditor(use_notebook=True, page_name='.name'),
                 style='custom',
                 width=1,
                 resizable=False,
                 show_label=False)
            ]

    __view3D = []

    __windows = []

    height = Int(200)

    width = Int(200)

    side_width = Int(200)

    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------

    def __init__(self):
        MultiSceneManager.__init__(self)

        self.__scenes = []

        self.__scenes_settings = {}

        self.__geometries = {'mesh': [], 'objects': []}

        self.__object_open_callbacks = []

        self.__start_callbacks = []

        self.__initialize_plot = True

        self.__change = False

        self._scene = self.scene_0

    # -------------------------------------------------------------------------

    @property
    def geometries(self):
        return self.__geometries

    @property
    def is_moving_vertex(self):
        return self.move_vertices_button.checked

    # -------------------------------------------------------------------------
    #                             Building up
    # -------------------------------------------------------------------------

    def start(self):
        self.make_scenes()
        if len(self._components) > 0 and len(self.__windows) > 0:
            self.components = self._components
            view = View(HSplit(self.tabs,
                               self.__view3D,
                               self.__windows),
                        **self.toolbar)
        elif len(self._components) > 0:
            self.components = self._components
            view = View(HSplit(self.tabs,
                               self.__view3D),
                        **self.toolbar)
        else:
            view = View(self.__view3D, **self.toolbar)
        self._closed = False

        for callback in self.__start_callbacks:
            if callable(callback):
                callback()

        for component in self._components:
            component.initialize_plot()

        self._initialize_geometries()

        self.object_changed()

        self.configure_traits(view=view)

    def add_component(self, component):
        component.geolab = self
        self._components.append(component)
        component.geolab_settings()
        self.__initialize_plot = False

    def add_scene(self, name, **kwargs):
        if name not in self.__scenes:
            self.__scenes.append(name)
            self.__scenes_settings[name] = kwargs

    def make_scenes(self):
        if len(self.__scenes) == 0:
            self.__scenes = ['scene_0']
            self.__scenes_settings = {'scene_0': {}}
        index = 0
        for key in self.__scenes:
            if index == 0:
                scene = self.scene_0
                self.__view3D = [Item('scene_0',
                                      editor=SceneEditor(scene_class=GeolabScene),
                                      show_label=False,
                                      resizable=True,
                                      height=self.height,
                                      width=self.width)]

            else:
                if index == 1:
                    scene = self.scene_1
                elif index == 2:
                    scene = self.scene_2
                elif index == 3:
                    scene = self.scene_3
                else:
                    return
                name = 'scene_{}'.format(index)
                settings = self.__scenes_settings[key]
                style = settings.get('style', 'geolab')
                if style == 'plain':
                    editor = Scene
                elif style == 'geolab':
                    editor = GeolabSecondaryScene
                elif style == 'base':
                    editor = GeolabBaseScene
                resizable = settings.get('resizable', True)
                self.__windows.append(Item(name,
                                           editor=SceneEditor(scene_class=editor),
                                           show_label=False,
                                           resizable=resizable,
                                           height=100,
                                           width=self.side_width))
            MultiSceneManager.add_scene(self, scene, key)
            index += 1
        if index <= 3:
            self.scene_3.mayavi_scene.remove()
        if index <= 2:
            self.scene_2.mayavi_scene.remove()
        if index <= 1:
            self.scene_1.mayavi_scene.remove()

    def _closing(self):
        for component in self._components:
            component.geolab_closing()
        self.scene_0.scene.close()
        self.scene_1.scene.close()
        self.scene_2.scene.close()
        self.scene_3.scene.close()
        # del self.scene_model_0.engine
        # del self.scene_model_0
        # del self.scene_model_1
        # del self.scene_model_2
        # del self.scene_model_3

    @on_trait_change('scene_0.activated, scene1.activated,\
                     scene_2.activated, scene_3.activated')
    def _scene_settings(self):
        for name in self.__scenes_settings:
            scene = self.get_scene(name)
            try:
                settings = self.__scenes_settings[name]
                interaction = settings.get('interaction', None)
                position = settings.get('position', None)
                if interaction == '2d':
                    scene.interaction_2d()
                elif interaction == 'locked':
                    scene.interaction_locked()
                if position is not None:
                    scene.position = position
                    scene.set_view(position)
            except:
                pass
        # for callback in self.__start_callbacks:
        #     if callable(callback):
        #         callback()
        self.initialize_plot()

    # -------------------------------------------------------------------------
    #                             Standard Methods
    # -------------------------------------------------------------------------

    def buttons_off(self):
        self.set_state()

    def set_state(self, name=None):
        self.select_off()
        if name != 'select_object':
            self.select_object_button.checked = False
        if name != 'move_vertices':
            self.move_vertices_button.checked = False
        if name != 'select_vertices':
            self.select_vertices_button.checked = False
        if name != 'select_faces':
            self.select_faces_button.checked = False
        if name != 'select_edges':
            self.select_edges_button.checked = False
        if name != 'select_boundary_vertices':
            self.select_boundary_vertices_button.checked = False
        if name != 'select_all_vertices':
            self.select_all_vertices_button.checked = False
        for component in self._components:
            component.geolab_set_state(name)

    def object_changed(self, main=True):
        SceneManager.object_changed(self)
        self.close_tools()
        self._current_object = self.current_object_type
        if main:
            self.set_state()
            self.select_off()

    # --------------------------------------------------------------------------
    #                                Background
    # --------------------------------------------------------------------------

    def background_switch(self):
        if self._background_switch_counter == 1:
            self.set_background((1, 1, 1))
            self._background_switch_counter = 0
        else:
            self.set_background((0.5, 0.5, 0.5))
            self._background_switch_counter = 1

    # -------------------------------------------------------------------------
    #                                  Open
    # -------------------------------------------------------------------------

    def add_start_callback(self, callback):
        self.__start_callbacks.append(callback)

    def _initialize_geometries(self):
        for kwargs in self.__geometries['mesh']:
            MultiSceneManager.add_mesh(self, **kwargs)
            self.last_object.set_scene = self.scene_0
            self.last_object.plot_faces(smooth=False)
        for obj in self.__geometries['objects']:
            MultiSceneManager.add_geometry(self, obj)
            self.last_object.set_scene = self.scene_0
            self.last_object.plot_faces(smooth=False)

    def open_mesh(self, vertices, **kwargs):
        kwargs['vertices'] = vertices
        self.__geometries['mesh'].append(kwargs)

    def open_geometry(self, geometry):
        self.__geometries['objects'].append(geometry)

    # def open_obj_file(self, file_name):
    #     G = openobj.open_obj_file(file_name)
    #     self.__geometries.append(G)
    #     if not self._closed:
    #         empty = False
    #         if self.current_object is None:
    #             empty = True
    #         self.add_object(G)
    #         if empty:
    #             self.object_changed()
    #
    # def open_geometry(self, geometry):
    #     self.__geometries.append(geometry)
    #     if not self._closed:
    #         self.add_object(geometry)

    def reset_geometry(self):
        self.set_state()
        self.current_object.reset_geometry()

        # file_name = self.current_object.geometry.name + '.obj'
        # G = openobj.open_obj_file(file_name)
        # self.current_object.geometry = G
        # self.object_added()
        self.object_changed()
        self.current_object.update_plot()

    def set_reference(self):
        self.current_object.geometry.set_reference()

    def delete_selected_object(self):
        self.set_state()
        self.delete_object(self.current_object.name)
        # self._current_object = self.current_object_type
        self.object_changed()

    def add_geometry(self, obj, **kwargs):
        self.set_state()
        MultiSceneManager.add_geometry(self, obj, **kwargs)

    # --------------------------------------------------------------------------
    #                                  Tools
    # --------------------------------------------------------------------------

    # def open_import(self):
    #     extensions = ['*.obj']
    #     descriptions = ["OBJ"]
    #     wildcard = ""
    #     for description, extension in zip(descriptions, extensions):
    #         wildcard += "{} ({})|{}|".format(description,
    #                                          extension,
    #                                          extension)
    #     wildcard += "All (*.*)|(*.*)"
    #     self.set_state()
    #     dialog = FileDialog(title='Import geometry', action='open',
    #                         wildcard=wildcard)
    #     if dialog.open() == OK:
    #         self.open_obj_file(dialog.path)

    def open_save_geometry(self):
        self.save_geometry_tool.geolab_gui = self
        self.save_geometry_tool.start()

    def open_corner_tolerance(self):
        self.set_state(None)
        self.corner_tolerance_tool.geolab_gui = self
        self.corner_tolerance_tool.start()

    @on_trait_change('_closed')
    def close_all(self):
        self.close_tools()

    def close_tools(self):
        self.save_geometry_tool.close()
        self.corner_tolerance_tool.close()

    # --------------------------------------------------------------------------
    #                                Selection
    # --------------------------------------------------------------------------

    def selection_off(self):
        MultiSceneManager.select_off(self)
        self.select_boundary_vertices_button.checked = False
        self.select_edges_button.checked = False
        self.select_object_button.checked = False
        self.select_vertices_button.checked = False
        self.move_vertices_button.checked = False

    def select_object(self):
        if self.select_object_button.checked:
            self.set_state('select_object')
            MultiSceneManager.select_object(self)
        else:
            self.select_object_off()

    def select_vertices(self):
        if self.select_vertices_button.checked:
            self.set_state('select_vertices')
            self.current_object.select_vertices()
        else:
            self.current_object.select_vertices_off()

    def select_edges(self):
        if self.select_edges_button.checked:
            self.set_state('select_edges')
            self.current_object.select_edges()
        else:
            self.current_object.select_edges_off()

    def select_faces(self):
        if self.select_faces_button.checked:
            self.set_state('select_faces')
            self.current_object.select_faces()
        else:
            self.current_object.select_faces_off()

    def select_boundary_vertices(self):
        if self.select_boundary_vertices_button.checked:
            self.set_state('select_boundary_vertices')
            self.current_object.select_boundary_vertices()
        else:
            self.current_object.select_boundary_vertices_off()

    def select_all_vertices(self):
        if self.select_all_vertices_button.checked:

            self.set_state('select_all_vertices')
            self.current_object.select_all_vertices()
        else:
            self.current_object.select_vertices_off()

    def move_vertices(self):
        if self.move_vertices_button.checked:
            self.set_state('move_vertices')
            self.current_object.move_vertices(self.current_object.update_plot)
        else:
            self.current_object.move_vertices_off()


def geolab_gui():
    return GeolabGUI()
