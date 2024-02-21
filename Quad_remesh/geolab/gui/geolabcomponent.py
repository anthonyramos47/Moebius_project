

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

#------------------------------------------------------------------------------

from traits.api import HasTraits

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                  Component
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class GeolabComponent(HasTraits):

    name = 'component'

    def __init__(self):
        HasTraits.__init__(self)
        self.__geolab_gui = None

    @property
    def geolab(self):
        return self.__geolab_gui

    @geolab.setter
    def geolab(self, geolab):
        self.__geolab_gui = geolab
        self.__geolab_gui.add_update_callback('object_changing', self.geolab_object_change)
        self.__geolab_gui.add_update_callback('object_changed', self.geolab_object_changed)
        self.__geolab_gui.add_update_callback('object_saved', self.geolab_object_save)
        self.__geolab_gui.add_update_callback('object_added', self.geolab_object_added)
        self.__geolab_gui.add_start_callback(self.geolab_started)
        #self.__geolab.handler.add_state_callback(self.geolab_set_state)

    #@property
    #def handler(self):
        #return self.geolab.handler

    def geolab_settings(self):
        pass

    def initialize_plot(self):
        pass

    def geolab_set_state(self, name):
        pass

    def geolab_object_change(self):
        pass

    def geolab_object_changed(self):
        pass

    def geolab_object_save(self, file_name):
        pass

    def geolab_object_added(self):
        pass

    def geolab_started(self):
        pass
    
    def geolab_closing(self):
        pass


def geolab_component():
    return GeolabComponent()

