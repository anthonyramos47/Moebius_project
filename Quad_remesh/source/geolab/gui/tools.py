#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from traits.api import HasTraits, Instance, Property, Enum, Button, String, \
    on_trait_change, Float, Bool, Int, Constant, ReadOnly, \
    List, Array, Range, File, observe

from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor, HGroup, \
    Group, ListEditor, Tabbed, VGroup, CheckListEditor, \
    ArrayEditor, Action, ToolBar, Separator, Controller, \
    FileEditor, RangeEditor

from pyface.image_resource import ImageResource

from geolab.utilities.stringutilities import make_filepath

'''check.py: an interactive checker for mesh connectivity and normals'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
#                                    Tool
# -----------------------------------------------------------------------------


class T_Handler(Controller):

    def close(self, info, is_ok):
        info.object._closed = True
        Controller.close(self, info, is_ok)
        return True


class Tool(HasTraits):

    def __init__(self):
        HasTraits.__init__(self)
        self._geolab_gui = None

    @property
    def current_object(self):
        return self._geolab_gui.current_object

    @property
    def geolab_gui(self):
        return self._geolab_gui

    @geolab_gui.setter
    def geolab_gui(self, geolab_gui):
        self._geolab_gui = geolab_gui

    def close(self):
        try:
            self._closed = True
            self._handler.info.ui.dispose()
        except:
            pass


# -----------------------------------------------------------------------------
#                                   Save
# -----------------------------------------------------------------------------


class S_Handler(Controller):

    def close(self, info, is_ok):
        info.object._closed = True
        if info.initialized:
            info.object.current_object.highlight_off()
        Controller.close(self, info, is_ok)
        return True


class SaveGeometry(Tool):
    _closed = Bool(True)

    _handler = S_Handler()

    label = String('opt')

    save_button = Button(label='Save')

    view = View(HGroup('label',
                       Item('save_button',
                            show_label=False),
                       show_border=True),
                title='Save geometry',
                handler=_handler,
                icon=ImageResource('../icons/save.png'))

    def start(self):
        if not self._closed:
            self._handler.info.ui.dispose()
        self.configure_traits()
        self._closed = False
        self.current_object.highlight()

    def close(self):
        try:
            self._closed = True
            self._handler.info.ui.dispose()
        except:
            pass

    @observe('save_button')
    def save_file(self, event):
        name = '{}_{}'.format(self.current_object.name, self.label)
        path = make_filepath(name, ext='obj', overwrite=False)
        self.current_object.save(path)
        self.geolab_gui.object_saved(path)
        self.current_object.highlight_off()
        self.close()


# -----------------------------------------------------------------------------
#                                   Save
# -----------------------------------------------------------------------------


class CornerTolerance(Tool):

    corner_tolerance = Range(0.0, 1.0, 0.3)

    _closed = Bool(True)

    _handler = T_Handler()

    view = View(Group(Item('corner_tolerance',
                           editor=RangeEditor(mode='slider'), ),
                      show_border=True),
                title='Set corner tolerance',
                handler=_handler,
                icon=ImageResource('../icons/corners.png'))

    @property
    def scenemanager(self):
        return self._geolab_gui

    @scenemanager.setter
    def scenemanager(self, scene_manager):
        self._geolab_gui = scene_manager
        ct = (scene_manager.current_object.corner_tolerance + 1) / 2
        self.corner_tolerance = ct

    def start(self):
        if not self._closed:
            self._handler.info.ui.dispose()
        self.configure_traits()
        self._closed = False
        self.plot_corners()
        self.scenemanager.current_object.add_plot_callback(self.plot_corners,
                                                           name='corners')
        self.scenemanager.current_object.update_plot()

    def close(self):
        try:
            self._closed = True
            self._handler.info.ui.dispose()
        except:
            pass

    @on_trait_change('corner_tolerance')
    def set_corner_tolerance(self):
        self.scenemanager.current_object.corner_tolerance = self.corner_tolerance
        self.scenemanager.current_object.update_plot()

    @on_trait_change('_closed')
    def hide_corners(self):
        self.scenemanager.current_object.hide_source('corners')
        self.scenemanager.current_object.remove_plot_callback('corners')

    def plot_corners(self):
        corners = self.scenemanager.current_object.corners()
        r = self.scenemanager.current_object.g * 2.52
        self.scenemanager.current_object.plot_glyph(vertex_indices=corners,
                                                    glyph_type='sphere',
                                                    radius=r,
                                                    color='g',
                                                    name='corners')
