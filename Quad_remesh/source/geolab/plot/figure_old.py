# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import os

import copy

import numpy as np

from tvtk.api import tvtk

from mayavi.api import Engine

from tvtk.pyface.tvtk_scene import TVTKScene

from traits.api import Button, Range, on_trait_change, Instance

from traitsui.api import View, Item, Group, VGroup, HGroup

from tvtk.pyface.scene_editor import SceneEditor

from pyface.image_resource import ImageResource

from mayavi.tools.mlab_scene_model import MlabSceneModel

# -----------------------------------------------------------------------------

from geolab.plot.geolabscene import GeolabScene

from geolab.plot.imageutilities import blur_shadow, add_shadow

from geolab.plot.plotmanager import SourcesManager, PlotManager

from geolab.plot.facesource import Faces

from geolab.geometry.intersect import points_plane_projection

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class Figure(SourcesManager):

    def __init__(self, **kwargs):
        SourcesManager.__init__(self)

        self._kwargs = kwargs

        self._size = None

        self._magnification = None

        self._shadow_rgba = None

        self._shadows = []

        self._vtk_scene = None

        self._path = None

        self._callback_args = None

        self._engine = Engine()

        self._read_position_file()

    # -------------------------------------------------------------------------

    @property
    def position(self):
        return self._kwargs.get('position', None)

    @position.setter
    def position(self, position):
        self._kwargs['position'] = position

    @property
    def shade_position(self):
        return self._kwargs.get('shade_position', None)

    @shade_position.setter
    def shade_position(self, shade_position):
        self._kwargs['shade_position'] = shade_position

    @property
    def background(self):
        return self._kwargs.get('background', (1, 1, 1))

    @background.setter
    def background(self, background):
        self._kwargs['background'] = background

    @property
    def overwrite(self):
        return self._kwargs.get('overwrite', True)

    @overwrite.setter
    def overwrite(self, overwrite):
        self._kwargs['overwrite'] = overwrite

    @property
    def transparent(self):
        return self._kwargs.get('transparent', False)

    @transparent.setter
    def transparent(self, transparent):
        self._kwargs['transparent'] = transparent

    @property
    def size(self):
        return self._kwargs.get('size', (2500, 1800))

    @size.setter
    def size(self, size):
        self._kwargs['size'] = size

    @property
    def shading_geometry(self):
        return self._kwargs.get('shading_geometry', None)

    @shading_geometry.setter
    def shading_geometry(self, shading_geometry):
        self._kwargs['shading_geometry'] = shading_geometry

    @property
    def shade_blur(self):
        return self._kwargs.get('shade_blur', 20)

    @shade_blur.setter
    def shade_blur(self, shade_blur):
        self._kwargs['shade_blur'] = shade_blur

    @property
    def shade_opacity(self):
        return self._kwargs.get('shade_opacity', 0.5)

    @shade_opacity.setter
    def shade_opacity(self, shade_opacity):
        self._kwargs['shade_opacity'] = shade_opacity

    @property
    def name(self):
        return self._kwargs.get('name', 'img')

    @name.setter
    def name(self, name):
        self._kwargs['name'] = name

    @property
    def path(self):
        if self._path is None:
            self._make_path(self._kwargs.get('path', None))
        return self._path

    @path.setter
    def path(self, path):
        self._make_path(path)

    @property
    def position_name(self):
        return self._kwargs.get('position_path', self.name)

    @position_name.setter
    def position_name(self, path):
        self._kwargs['position_path'] = path

    # -------------------------------------------------------------------------
    #                               Public API
    # -------------------------------------------------------------------------

    def save_figure(self):
        self._read_position_file()
        self._make_size()
        self._make_shadow()
        self._make_scene()
        self._save_figure()
        self._close_scene()
        return True

    def rgba(self):
        self._read_position_file()
        self._make_size()
        self._make_shadow()
        self._make_scene()
        rgba = self._rgba()
        self._close_scene()
        return rgba

    def set_view(self, path=None):
        if path is None:
            path = self.path
        self._make_size()
        set_view(self.sources_list(), shadows=self._shadows, size=self._size,
                 resizable=False, path=path, position_path=self.position_name)
        self._read_position_file()

    def make_animation(self, callback, *args, iterations=10, frame_rate=30,
                       delay=0.1, set_view=True):
        from shutil import copyfile
        if not callable(callback):
            raise TypeError('figure must be a callable')
        n = max(int(frame_rate * delay), 1)
        img = 0
        self.clear_figure()
        for i in range(iterations):
            self._make_size()
            if i == 0:
                self._callback_args = callback(*args)
                if set_view:
                    self.set_view(self._path)
            else:
                self._callback_args = callback(*self._callback_args)
            self._make_shadow()
            self._make_scene()
            prefix = len(str(int(iterations))) - len(str(img)) + 1
            sv = '{}_{}.png'.format(self.name, '0' * prefix + str(img))
            file_name = os.path.join(self._path, sv)
            self._save_figure(file_name)
            img += 1
            for j in range(n - 1):
                prefix = len(str(int(iterations))) - len(str(img)) + 1
                sv2 = 'img_{}.png'.format('0' * prefix + str(img))
                file_name_2 = os.path.join(self._path, sv2)
                copyfile(file_name, file_name_2)
                img += 1
            self._close_scene()
            self.clear_figure()

    def plot_shadow(self, vertices, connectivity, **kwargs):
        if hasattr(vertices, 'vertices'):
            vertices = vertices.vertices
        kwargs['connectivity'] = connectivity
        self._shadows.append((vertices, kwargs))

    def clear_figure(self):
        self._shadows = []
        self.clear_sources()

    # -------------------------------------------------------------------------
    #                                Path
    # -------------------------------------------------------------------------

    def _read_position_file(self):
        p = _read_position_file(self.path, self.position_name)
        self.position = p[0]
        self.shade_position = p[1]

    def _make_file_name(self):
        name = '{}.png'.format(self.name)
        file_name = os.path.join(self.path, name)
        return file_name

    def _make_path(self, path):
        if path is None:
            path = os.getcwd()
        path = '{}'.format(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self._path = path

    # -------------------------------------------------------------------------
    #                                Scene
    # -------------------------------------------------------------------------

    def _make_size(self):
        m = max(self.size[0] // 1200, self.size[1] // 900) + 1
        self._magnification = m
        self._size = (self.size[0] // m, self.size[1] // m)

    def _close_scene(self):
        for scene in copy.copy(self._engine.scenes):
            self._engine.close_scene(scene)
            self._vtk_scene.close()

    def _set_lights(self):
        self.scene.light_manager.lights[0].azimuth = 45 + self.azimuth
        self.scene.light_manager.lights[0].elevation = 45 + self.elevation
        self.scene.light_manager.lights[1].azimuth = -60 + self.azimuth
        self.scene.light_manager.lights[1].elevation = -30 + self.elevation
        self.scene.light_manager.lights[2].azimuth = 60 + self.azimuth
        self.scene.light_manager.lights[2].elevation = -30 + self.elevation
        self.scene.light_manager.lights[3].azimuth = self.azimuth
        self.scene.light_manager.lights[3].elevation = self.elevation

    def _make_scene(self, objects=None):
        scene = TVTKScene()
        scene.disable_render = True
        scene.set_size(self._size)
        scene.magnification = self._magnification
        scene.point_smoothing = True
        scene.line_smoothing = True
        scene.polygon_smoothing = True
        scene.anti_aliasing_frames = 8
        self._engine.add_scene(scene)
        if objects is None:
            objects = self.sources_list()
        for obj in objects:
            self._engine.add_source(obj.source)
        pos = self.position
        if pos is not None:
            scene.camera.position = np.array([pos[0], pos[1], pos[2]])
            scene.camera.view_up = np.array([pos[3], pos[4], pos[5]])
            scene.camera.focal_point = np.array([pos[6], pos[7], pos[8]])
            scene.camera.view_angle = float(pos[9])
            scene.camera.clipping_range = np.array([pos[10], pos[11]])
            scene.camera.parallel_projection = pos[12]
            self._set_lights()
        scene.background = self.background
        scene.disable_render = False
        scene.render()
        self._vtk_scene = scene

    def _rgba(self):
        w2if = tvtk.WindowToImageFilter()
        w2if.input = self._vtk_scene.render_window  # w2if.input = scene._renwin
        if hasattr(w2if, 'magnification'):
            w2if.magnification = self._magnification
        else:
            w2if.scale = self._magnification, self._magnification
        w2if.input_buffer_type = 'rgba'
        w2if.update()
        shape = w2if.output.dimensions
        img = w2if.output.point_data.scalars.to_array()
        img.shape = (shape[1], shape[0], 4)
        img = np.flipud(img) / 255
        return img

    def _make_shadow(self):
        shadows = _make_shadow(self._shadows, self.shade_position)
        self._make_scene(shadows)
        sh = self._rgba()
        sh = blur_shadow(sh, blur=self.shade_blur, opacity=self.shade_opacity)
        self._shadow_rgba = sh
        self._close_scene()

    def _save_figure(self, file_name=None):
        if file_name is None:
            file_name = self._make_file_name()
        if not self.transparent and len(self._shadows) == 0:
            self._vtk_scene.save(file_name, size=self._size)
        else:
            try:
                import matplotlib.pyplot as plt
                img = self._rgba()
                if len(self._shadows) > 0:
                    img = add_shadow(img, self._shadow_rgba)
                # plt.axis('off')
                plt.imsave(arr=img, fname=file_name)
            except ImportError:
                print("Transparency and shadows available only with matplotlib")
                self._vtk_scene.save(file_name, size=self._size)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class FigureViewer(PlotManager):
    scene = Instance(MlabSceneModel, ())

    editor = SceneEditor(scene_class=GeolabScene)

    gamma = Range(-180, 180, 90)

    delta = Range(-360, 360, 0)

    alpha = Range(-90, 90, 45)

    beta = Range(-360, 360, 315)

    offset = Range(-300, 300, 0)

    view_angle = Range(1, 160, 30)

    azimuth = Range(-180, 180, 0)

    elevation = Range(-180, 180, 0)

    save_view = Button()

    # -------------------------------------------------------------------------

    def __init__(self, sources, **kwargs):
        PlotManager.__init__(self)

        self.background = (1, 1, 1)

        self._size = kwargs.get('size', (800, 800))

        self._path = kwargs.get('path', None)

        self._position_path = kwargs.get('position_path', None)

        self._resizable = kwargs.get('resizable', True)

        self._shadows = kwargs.get('shadows', None)

        self.add_sources(sources)

    # -------------------------------------------------------------------------

    def _get_shadow_position(self):
        if self._shadows is None:
            f = 1
        else:
            m = -np.inf
            for s in self._shadows:
                m = max(m, np.max(np.abs(s[0])))
            f = - m / 100
        p = [np.deg2rad(self.alpha), np.deg2rad(self.beta),
             np.deg2rad(self.gamma), np.deg2rad(self.delta), self.offset * f]
        return p

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------

    def start(self):
        view = View(
            HGroup(Group(
                VGroup('alpha',
                       'beta',
                       'gamma',
                       'delta',
                       'offset',
                       label='Shadow',
                       show_border=True,
                       ),
                VGroup('view_angle',
                       label='Camera',
                       show_border=True,
                       ),
                VGroup('azimuth',
                       'elevation',
                       label='Lights',
                       show_border=True,
                       ),
                Item('save_view', show_label=False),
                show_border=True),
                Item('scene',
                     editor=SceneEditor(scene_class=GeolabScene),
                     show_label=False,
                     resizable=self._resizable,
                     height=self._size[1],
                     width=self._size[0],
                     ),
            ),
            resizable=self._resizable,
            title='GeoLab viewer',
            icon=ImageResource('../icons/geolab_logo.png'),
        )

        self._plot()
        self.configure_traits(view=view)

    @on_trait_change('alpha, beta, gamma, delta, offset, shadow')
    def _plot(self):
        if self._shadows is not None:
            self._make_shadow()
            self.update_plot()

    @on_trait_change('save_view')
    def _save_view_fired(self):
        print(self.get_position())
        p = [self.get_position(), self._get_shadow_position()]
        _make_position_file(p, self._path, self._position_path)

    @on_trait_change('view_angle')
    def focal_changed(self):
        p = self.get_position()
        p[9] = self.view_angle
        self.set_position(p)

    @on_trait_change('azimuth, elevation')
    def lights_changed(self):
        try:
            self.scene.scene_editor.light_manager.lights[0].azimuth = 45 + self.azimuth
            self.scene.scene_editor.light_manager.lights[0].elevation = 45 + self.elevation
            self.scene.scene_editor.light_manager.lights[1].azimuth = -60 + self.azimuth
            self.scene.scene_editor.light_manager.lights[1].elevation = -30 + self.elevation
            self.scene.scene_editor.light_manager.lights[2].azimuth = 60 + self.azimuth
            self.scene.scene_editor.light_manager.lights[2].elevation = -30 + self.elevation
            self.scene.scene_editor.light_manager.lights[3].azimuth = self.azimuth
            self.scene.scene_editor.light_manager.lights[3].elevation = self.elevation
        except:
            pass

    # @on_trait_change('scene:scene_editor:light_manager')
    # def muck_with_the_lights(self):
    # print(self.scene.light_manager.lights)

    def _make_shadow(self):
        if self._shadows is None:
            return
        position = self._get_shadow_position()
        i = 0
        for shadow in self._shadows:
            self.rename_sources = False
            if position is None:
                S = points_plane_projection(shadow[0])
            else:
                Nx = np.cos(position[2]) * np.cos(position[3])
                Ny = np.cos(position[2]) * np.sin(position[3])
                Nz = np.sin(position[2])
                Lx = np.cos(position[0]) * np.cos(position[1])
                Ly = np.cos(position[0]) * np.sin(position[1])
                Lz = np.sin(position[0])
                N = [[Nx, Ny, Nz]]
                L = [Lx, Ly, Lz]
                C = [[-position[4] * Lx, -position[4] * Ly, -position[4] * Lz]]
                S = points_plane_projection(shadow[0], plane_normal=N,
                                            plane_point=C, light_direction=L)
            kwargs = shadow[1]
            self.plot_faces(S, connectivity=kwargs.get('connectivity', None),
                            color='k', shading=False, glossy=0,
                            name='shad_{}'.format(i))
            i += 1

    @on_trait_change('scene.activated')
    def _scene_settings(self):
        if self.position is not None:
            self.set_position(self.position)
        if self.background is not None:
            self.scene.background = self.background
        if self.magnification is not None:
            self.scene.magnification = self.magnification
        self.initialize_plot()


# -----------------------------------------------------------------------------
#                                Functions
# -----------------------------------------------------------------------------


def figure(**kwargs):
    return Figure(**kwargs)


# -----------------------------------------------------------------------------
#                                Functions
# -----------------------------------------------------------------------------


def set_view(objects, **kwargs):
    viewer = FigureViewer(objects, **kwargs)
    viewer.start()


# -----------------------------------------------------------------------------


def save_figure(sources, **kwargs):
    figure = Figure(**kwargs)
    figure.add_sources(sources)
    figure.save_figure()


# -----------------------------------------------------------------------------


def rgba_image(sources, **kwargs):
    f = Figure(**kwargs)
    f.add_sources(sources)
    return f.rgba()


# -----------------------------------------------------------------------------


def _make_shadow(shadows, position=None):
    if len(shadows) == 0:
        return
    shadow_sources = []
    i = 0
    for shadow in shadows:
        if position is None:
            S = points_plane_projection(shadow[0])
        else:
            Nx = np.cos(position[2]) * np.cos(position[3])
            Ny = np.cos(position[2]) * np.sin(position[3])
            Nz = np.sin(position[2])
            Lx = np.cos(position[0]) * np.cos(position[1])
            Ly = np.cos(position[0]) * np.sin(position[1])
            Lz = np.sin(position[0])
            N = [[Nx, Ny, Nz]]
            L = [Lx, Ly, Lz]
            C = [[-position[4] * Lx, -position[4] * Ly, -position[4] * Lz]]
            S = points_plane_projection(shadow[0], plane_normal=N,
                                        plane_point=C, light_direction=L)
        kwargs = shadow[1]
        shadow_sources.append(Faces(S, connectivity=kwargs.get('connectivity', None),
                                    color='k', shading=False, glossy=0,
                                    name='shad_{}'.format(i)))
        i += 1
    return shadow_sources


# -----------------------------------------------------------------------------


def _make_position_file(position, path, position_path):
    name = '{}.position'.format(position_path)
    file_name = os.path.join(path, name)
    out = open(file_name, 'w')
    line = ''
    for p in position[0]:
        line += '{} '.format(p)
    line += '\n'
    for p in position[1]:
        line += '{} '.format(p)
    line += '\n'
    for p in position[2]:
        line += '{} '.format(p)
    out.write(line)
    out.close()


# -----------------------------------------------------------------------------


def _read_position_file(path, position_path):
    name = '{}.position'.format(position_path)
    file_name = os.path.join(path, name)
    if not os.path.exists(file_name):
        return [None, None]
    read = open(file_name, encoding='utf-8')
    position = []
    for line in read:
        p = []
        sp_line = line.split(' ')
        for n in sp_line:
            if n == 'False':
                p.append(False)
            elif n == 'True':
                p.append(True)
            else:
                try:
                    p.append(float(n))
                except ValueError:
                    pass
        position.append(p)
    return position
