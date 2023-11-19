# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from traits.api import Instance, on_trait_change, Button, Any, Range

from traitsui.api import View, Item, HGroup, VSplit, Controller

from tvtk.pyface.scene_editor import SceneEditor

from mayavi.tools.mlab_scene_model import MlabSceneModel

from pyface.image_resource import ImageResource

from pyface.timer.api import Timer

# -----------------------------------------------------------------------------

from geolab.plot.geolabscene import GeolabScene

from geolab.plot.plotmanager import PlotManager

# -----------------------------------------------------------------------------

'''viewer.py: The viewer for plot souce classes'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                    Viewer
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class Handler(Controller):

    def close(self, info, is_ok):
        info.object._closing()
        info.object._closed = True
        Controller.close(self, info, is_ok)
        return True


class Viewer(PlotManager):

    _handler = Handler()

    scene = Instance(MlabSceneModel, ())

    editor = SceneEditor(scene_class=GeolabScene)

    play = Button()

    pause = Button()

    timer = Any

    delay = Range(10, 100000, 50, desc='refresh rate')

    def __init__(self, **kwargs):
        PlotManager.__init__(self)

        self.size = kwargs.get('size', (800, 800))

        self.position = kwargs.get('position', None)

        self.background = kwargs.get('background', None)

        self.resizable = kwargs.get('resizable', True)

        self.magnification = kwargs.get('magnification', 4)

        self._callback = None

        self._callback_args = None

        self.iterations = kwargs.get('iterations', None)

    # -------------------------------------------------------------------------
    #                                 Methods
    # -------------------------------------------------------------------------

    def _update_attributes(self, **kwargs):
        self.delay = kwargs.get('delay', self.delay)
        self.size = kwargs.get('size', self.size)
        self.position = kwargs.get('position', self.position)
        self.background = kwargs.get('background', self.background)
        self.resizable = kwargs.get('resizable', self.resizable)
        self.magnification = kwargs.get('magnification', self.magnification)
        self.iterations = kwargs.get('iterations', self.iterations)

    def show(self, **kwargs):
        """Opens a window showing the plot."""
        self._update_attributes(**kwargs)
        win = View(Item('scene',
                        editor=SceneEditor(scene_class=GeolabScene),
                        show_label=False,
                        resizable=self.resizable,
                        height=self.size[1],
                        width=self.size[0],
                        ),
                   resizable=self.resizable,
                   title='GeoLab viewer',
                   icon=ImageResource('../icons/geolab_logo.png')
                   )
        self.update_plot()
        self.configure_traits(view=win)

    def animate(self, callback, args=None, **kwargs):
        """Create an animation.

        Parameters
        ----------
        callback : callable
            A function to be plot at each frame that returns its input
            parameters

        Optional Arguments
        ------------------
        input parameters of the callback
        """
        self._update_attributes(**kwargs)
        win = View(
            VSplit(Item('scene',
                        editor=SceneEditor(scene_class=GeolabScene),
                        show_label=False,
                        resizable=self.resizable,
                        height=self.size[1],
                        width=self.size[0],
                        ),
                   HGroup(Item('play', show_label=False),
                          Item('pause', show_label=False),
                          Item('delay', show_label=True),
                          show_border=True),
                   ),
            resizable=self.resizable,
            title='GeoLab animator',
            handler=self._handler,
            icon=ImageResource('../icons/geolab_logo.png'),
            )
        self.clear_all()
        self.rename_sources = False
        self._callback_args = args
        self._callback = callback
        if self._callback_args is not None:
            args = self._callback(*self._callback_args)
            if type(args) is not tuple:
                args = [args]
            self._callback_args = args
        else:
            self._callback()
        self.rename_sources = False
        self.configure_traits(view=win)

    def close(self):
        """Close the animator UI."""
        self._closing()
        if self.ui is not None:
            self.ui.dispose()

    def _closing(self):
        try:
            self.timer.Stop()
        except:
            pass

    # -------------------------------------------------------------------------
    #                              Traited
    # -------------------------------------------------------------------------

    def _play_fired(self):
        def __anim():
            i = 0
            while True:
                if self._callback_args is not None:
                    args = self._callback(*self._callback_args)
                    if type(args) is not tuple:
                        args = [args]
                    self._callback_args = args
                else:
                    self._callback()
                self.update_plot()
                i += 1
                if self.iterations is not None:
                    if i == self.iterations:
                        break
                yield
        anim = __anim()
        self.timer = Timer(self.delay, anim.__next__)
        self.timer.Start(self.delay)

    def _pause_fired(self):
        self.timer.Stop()

    def _delay_changed(self, value):
        t = self.timer
        if t is None:
            return
        if t.IsRunning():
            t.Stop()
            t.Start(value)

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
#                                 View Function
# -----------------------------------------------------------------------------


def plotter(**kwargs):
    """Creates a viewer instance.
    Provides plotting functions for meshes and points for display.

    Keyword Arguments
    -----------------
    size : tuple (W, H)
        A tuple with the width and height size of the image.
        Default is (800,800)
    background : tuple of floats (r,g,b)
        The color of the background, with value in the interval [0, 1]
         per channel.
    iterations : int
        The maximum number of iterations of an animation callback.
    position : list
    magnification : int
        The multiplication of image size when saving an image.

    Returns
    -------
    viewer : Viewer
    """
    return Viewer(**kwargs)

