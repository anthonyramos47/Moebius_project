#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import sys

from traits.api import Button, Str, Float, Bool, Int, Range

from traits.api import on_trait_change

from traitsui.api import View, Item, HGroup, Group, VGroup

import numpy as np

#------------------------------------------------------------------------------

from geolab import geolab_component

from geopt.optimization.meshoptimizer import MeshOptimizer

#------------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                        InteractiveGuidedProjection
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class MeshOptimizerGUI(geolab_component):

    name = Str('Mesh Optimizer')

    iterations = Int(1)

    epsilon = Float(0.001, label='dumping')

    step = Range(0.0, 1.0, 1.0)

    fairness_reduction = Float(0)

    equilibrium = Float(0)

    geometric = Float(1)

    planarity = Float(0)

    mesh_fairness = Float(0.3)

    tangential_fairness = Float(0.3)

    boundary_fairness = Float(0.3)

    reference_closeness = Float(0.01)

    boundary_closeness = Float(0)

    gliding = Float(1)

    self_closeness = Float(0)

    reinitialize = Button(label='Initialize')

    fix_corners = Button()

    reset = Button(label='Reset')

    optimize = Button(label='Optimize')

    compression = Bool(False)

    geometric_error = Str('_', label='Geometric')

    planarity_error = Str('_', label='Planarity')

    equilibrium_error = Str('_', label='Equilibrium')

    interactive = Bool(False, label='Interactive')

    step_control = Bool(False)

    force_min = Float(0)

    force_equal = Float(0)

    airy_scale = Float(1)

    fix_boundary_normals = Bool(False, label='Fix b-normals')

    #--------------------------------------------------------------------------
    view = View(
                VGroup(
                       Group(#'iterations',
                             #'epsilon',
                             'mesh_fairness',
                             'tangential_fairness',
                             'boundary_fairness',
                             'planarity',
                             'equilibrium',
                             'geometric',
                             'reference_closeness',
                             'boundary_closeness',
                             'gliding',
                             'self_closeness',
                             'compression',
                             show_border=True,
                             label='settings'),
                       VGroup('geometric_error',
                              'equilibrium_error',
                              'planarity_error',
                              style='readonly',
                              label='errors [ mean | max ]',
                              show_border=True),
                       Group(Item('step', show_label=False),
                             show_border=True,
                             label='step'),
                       HGroup(Item('interactive',
                                   tooltip='Interactive',
                                   show_label=False,),
                              Item('_'),
                              'optimize',
                              'reinitialize',
                              show_labels=False,
                              show_border=True),
                       show_border=False,
                       show_labels=True,
                       ),
               resizable=False,
               width = 0.1,
               )

    # -------------------------------------------------------------------------
    #                                Initialize
    # -------------------------------------------------------------------------

    optimizer = MeshOptimizer()

    # -------------------------------------------------------------------------
    #                                Properties
    # -------------------------------------------------------------------------

    @property
    def mesh(self):
        return self.geolab.current_object.geometry

    @property
    def current_object(self):
        return self.geolab.current_object

    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------

    def geolab_settings(self):
        pass
        #self.geolab.add_scene('3D')
        #self.geolab.add_scene('2D')

    def geolab_object_added(self):
        obj = self.geolab.last_object
        obj.normalize()
        if obj.type == 'Mesh':
            obj.area_load = 1
            obj.beam_load = 0
            #self.initialize_mesh_plot_functions(obj)

    def geolab_object_change(self):
        pass

    def geolab_object_changed(self):
        self.optimizer.mesh = self.geolab.current_object
        self.set_settings()
        self.optimizer.initialization()
        #self.geolab.current_object.clear()
        self.initialize_mesh_plot_functions(self.geolab.current_object)
        self.print_error()

    def geolab_object_save(self, file_name):
        self.optimizer.save_report(file_name)

    def geolab_set_state(self, state):
        if state != 'sy_interactive':
            self.interactive = False

    # -------------------------------------------------------------------------
    #                                  Plot
    # -------------------------------------------------------------------------

    def initialize_mesh_plot_functions(self, obj):
        obj.face_plot = 'face planarity'
        obj.edge_plot = 'axial force'
        obj.show_supported_vertices = True
        obj.show_fixed_vertices = True
        obj.update_plot()

    # -------------------------------------------------------------------------
    #                              Optimization
    # -------------------------------------------------------------------------

    def optimization_step(self):
        if not self.interactive:
            self.geolab.set_state(None)
        self.set_settings()
        self.optimizer.optimize()
        self.print_error()
        self.mesh_fairness = self.mesh_fairness/(10**(self.fairness_reduction))
        self.tangential_fairness = self.tangential_fairness/(10**
                                    (self.fairness_reduction))
        self.boundary_fairness = self.boundary_fairness/(10**
                                (self.fairness_reduction))
        self.current_object.update_plot()

    def print_error(self):
        self.geometric_error = self.optimizer.geometric_error_string()
        self.equilibrium_error = self.optimizer.equilibrium_error_string()
        self.planarity_error = self.optimizer.planarity_error_string()

    @on_trait_change('optimize')
    def optimize_mesh(self):
        self.current_object.iterate(self.optimization_step, self.iterations)
        self.current_object.update_plot()

    @on_trait_change('interactive')
    def interactive_optimize_mesh(self):
        self.geolab.set_state('sy_interactive')
        if self.interactive:
            def start():
                self.mesh.handle = self.current_object.selected_vertices
            def interact():
                self.current_object.iterate(self.optimization_step,1)
            def end():
                self.current_object.iterate(self.optimization_step,5)
            self.current_object.move_vertices(interact,start,end)
        else:
            self.mesh.handle = None
            self.current_object.move_vertices_off()

    @on_trait_change('fix_corners')
    def fix_corners_fired(self):
        M = self.geolab.current_object.geometry
        l = M.vertex_ring_lengths()
        v = np.where(l == 2)[0]
        M.release()
        M.fix(v)
        M.constrain(v)
        self.geolab.update_plot()

    # -------------------------------------------------------------------------
    #                              Settings
    # -------------------------------------------------------------------------

    def set_settings(self):
        self.optimizer.make_residual = True
        self.optimizer.exclude_free_boundary = True
        self.optimizer.threshold = 1e-20
        self.optimizer.iterations = 1
        self.optimizer.epsilon = self.epsilon
        self.optimizer.step = self.step
        self.optimizer.fairness_reduction = self.fairness_reduction
        self.optimizer.compression = self.compression
        self.optimizer.set_weight('forces_minimization', self.force_min)
        self.optimizer.set_weight('laplacian_fairness', self.mesh_fairness)
        self.optimizer.set_weight('tangential_fairness', self.tangential_fairness)
        self.optimizer.set_weight('boundary_fairness', self.boundary_fairness)
        self.optimizer.set_weight('gliding', self.gliding)
        self.optimizer.set_weight('reference_closeness', self.reference_closeness)
        self.optimizer.set_weight('boundary_closeness', self.boundary_closeness)
        self.optimizer.set_weight('self_closeness', self.self_closeness)
        self.optimizer.set_weight('geometric', self.geometric)
        self.optimizer.set_weight('planarity', self.planarity)
        self.optimizer.set_weight('equilibrium',  self.equilibrium)

    # -------------------------------------------------------------------------
    #                                  Reset
    # -------------------------------------------------------------------------

    @on_trait_change('reinitialize')
    def reinitialize_optimizer(self):
        self.set_settings()
        self.optimizer.reinitialization()
        self.print_error()
        self.current_object.update_plot()

