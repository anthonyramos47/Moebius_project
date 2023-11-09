import os 
import sys

# Add hananLab to path
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

import vedo as vd 
import tkinter as tk
from tkinter import Entry, Label, Button, filedialog, Toplevel
import numpy as np
import igl
from hanan.geometry.mesh import Mesh
from hanan.geometry.utils import circle_3pts
from hanan.vedo_gui.Visualizer import Visualizer

from hanan.optimization.Optimizer import Optimizer
from hanan.optimization.LineCong import LineCong
from hanan.optimization.Torsal import Torsal
from hanan.optimization.Torsal_angle import Torsal_angle
from hanan.optimization.Sphere_angle import Sphere_angle


class Vis_Approx(Visualizer):

    def __init__(self) -> None:
        super().__init__()
        self.var_dic = None # Store variable dictionary
        self.bc = None

    
    def setup(self)-> None:
        """ Here we setup our initial conditions and get all the geometry of our mesh
        """

        # Load test mesh
        tv, tf = np.array(self.meshes[0][0]), np.array(self.meshes[0][1] )

        # Create dual mesh
        tmesh = Mesh()
        tmesh.make_mesh(tv,tf)

        # Get inner vertices
        inner_vertices = tmesh.inner_vertices()

        # Get vertex normals for test mesh
        e_i = igl.per_vertex_normals(tv, tf)

        # Fix normal directions
        signs = np.sign(np.sum(e_i * ([0,0,1]), axis=1))
        e_i = e_i * signs[:, None]

        # Compute circumcenters and axis vectors for each triangle
        p1, p2, p3 = tv[tf[:, 0]], tv[tf[:, 1]], tv[tf[:, 2]]

        bf, _, ncf = circle_3pts(p1, p2, p3)

        # store barycenters
        self.bc = bf

        # Dual topology 
        dual_tf = tmesh.vertex_ring_faces_list()

        nV = len(tv) # number of vertices

        nF = len(tf) # number of faces

        var_idx = {     "e"  : np.arange( 0            , 3*nV), 
                        "a1" : np.arange( 3*nV        , 3*nV +    nF),
                        "b1" : np.arange( 3*nV +    nF, 3*nV +  2*nF),
                        "nt1": np.arange( 3*nV +  2*nF, 3*nV +  5*nF),
                        "a2" : np.arange( 3*nV +  5*nF, 3*nV +  6*nF),
                        "b2" : np.arange( 3*nV +  6*nF, 3*nV +  7*nF),
                        "nt2": np.arange( 3*nV +  7*nF, 3*nV + 10*nF),
                        "df" : np.arange( 3*nV + 10*nF, 3*nV + 11*nF),
                        "c0" : np.arange( 3*nV + 11*nF, 3*nV + 12*nF),
                        "u"  : np.arange( 3*nV + 12*nF, 3*nV + 13*nF),
                }
        
        self.var_dic = var_idx

        # Init X 
        X = np.zeros(3*len(tv) + 13*len(tf))

        X[var_idx["e"]] = e_i.flatten()

        X[var_idx["df"]] = np.random.rand(len(var_idx["df"]))*2

        X[var_idx["c0"]] = np.sum(X[var_idx["nt1"]].reshape(-1,3)*X[var_idx["nt2"]].reshape(-1,3), axis=1)

        X[var_idx["u"]] = 1



        # # Init LineCong
        linecong = LineCong()
        linecong.initialize_constraint(X, var_idx, len(tv), bf, ncf, len(tf), dual_tf, inner_vertices)
        linecong.set_weigth(10)

        # #Init Torsal 
        torsal = Torsal()
        torsal.initialize_constraint(X, var_idx, tv, tf, bf, ncf)

        # Init Torsal angle
        tang = Torsal_angle()
        tang.initialize_constraint(X, var_idx, tv, tf)

        # Sphere angle
        sph_ang = Sphere_angle()
        sph_ang.initialize_constraint(X, var_idx, tv, tf, bf, ncf) 
        

        self.constraints = {"linecong" : linecong, 
                            "torsal"   : torsal,
                            "tang"     : tang,
                            "sph_ang"  : sph_ang}

        self.weights = {"linecong" : 1,
                        "torsal"   : 1,
                        "tang"     : 1,
                        "sph_ang"  : 1}

        # init optimizer
        optimizer = Optimizer()
        optimizer.initialize_optimizer(X, "LM", 0.8)

        self.optimizer = optimizer

        
    def update_scene(self, X):


        # Cross field
        t1 = self.constraints["torsal"].compute_t(X[self.var_dic["a1"]],X[self.var_dic["b1"]])
        t2 = self.constraints["torsal"].compute_t(X[self.var_dic["a2"]],X[self.var_dic["b2"]])
        
        t1 /= np.linalg.norm(t1,axis=1)[:,None]
        t2 /= np.linalg.norm(t2,axis=1)[:,None]
        # Draw line congruence
        bc = self.bc

        lt1 = vd.Lines(
                        bc-0.1*t1,
                        bc+0.1*t1,
                        res=1,
                        scale=0.4,
                        lw=1,
                        c='w',
                        alpha=1.0)


        
        # lt2 = vd.Lines(
        #                 bc-0.1*t2,
        #                 bc+0.1*t2,
        #                 res=1,
        #                 scale=0.4,
        #                 lw=1,
        #                 c='b',
        #                 alpha=1.0)
        
        # self.plotter.add(lt1)
        # self.plotter.add(lt2)

        self.plotter.render()

        
        