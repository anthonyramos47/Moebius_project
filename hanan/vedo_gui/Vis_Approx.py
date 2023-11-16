import os 
import sys

# Add hananLab to path
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
print(path)

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
        self.bc = None # Barycenters
        self.bf = None # Circumcenters
        self.ncf = None # Normal of the circumcenters
        self.radius = None # Radius of the spheres
        self.px0 = None # Planes x0 point per edge
        self.pn = None # Planes normal per edge

    # def spherical_panels(self):
    #     s = vd.Sphere([0,0,0], 2, c="r", alpha=1)

    #     s.cut_with_plane([-0.2,0,0], [1,0,0])
    #     s.cut_with_plane([0.2,0.2,0], [-1,-1,0])
    #     s.cut_with_plane([-0.2,-0.2,0], [1,1,0])

    #     self.add_to_scene("spheres", s)
    #     self.plotter.render()
        
    def spherical_panels(self):

        # Get variables
        X = self.optimizer.X

        # Get circumcenters distance to center
        df = X[self.var_dic["df"]]

        # Get circumcenters
        bf = self.bf

        # Get centers
        cf = bf + df[:,None]*self.ncf

        # Init Halfedge 
        mesh = Mesh()
        mesh.make_mesh(self.meshes[0][0], self.meshes[0][1])

        # Get inner edges
        inner_edges = mesh.inner_edges()

        # Get adjacent faces
        f1, f2 = mesh.edge_faces()

        # Get circumcenters
        c1 = cf[f1[inner_edges]]
        c2 = cf[f2[inner_edges]]

        v = np.array(self.meshes[0][0])
        f = np.array(self.meshes[0][1])

        r = np.linalg.norm(cf - v[f[:,0]] , axis=1)

        inner_faces = mesh.inner_faces()

        r1 = r[f1[inner_edges]]
        r2 = r[f2[inner_edges]]

        d = np.linalg.norm(c2-c1, axis=1)

        self.pn = (c2-c1)/d[:,None]

        self.px0 = c1 + ((r1**2 - r2**2 + d**2)/(2*d))[:,None]*self.pn

        # Draw spheres per each face
        spheres = []
        for c, r in zip(cf, r):
            spheres.append(vd.Sphere(c, r, c="b", alpha=1))

        # Cut spheres with planes at each edge
        for e in range(len(inner_edges)):
            # Get plane
            x0 = self.px0[e]
            n = self.pn[e]

            # Get spheres
            s1 = spheres[f1[e]]
            s2 = spheres[f2[e]]

            # Get baricenters f1, f2
            b1 = self.bc[f1[e]]
            b2 = self.bc[f2[e]]

            dir12 = (b2 - b1)/np.linalg.norm(b2-b1)
            dir21 = (b1 - b2)/np.linalg.norm(b1-b2)

            sig1 = np.sign(np.dot(dir12, n))
            sig2 = np.sign(np.dot(dir21, n))
            
            # Cut sphere
            s1.cut_with_plane(x0, -sig1*n)
            s2.cut_with_plane(x0, -sig2*n)

        #Draw spheres
        for i, s_id in enumerate(inner_faces):
            s = spheres[s_id]
            self.add_to_scene("spheres_"+str(i), s)


    
    def init_menu(self) -> None:
        # Draw spherical panels
        self.plotter.add_button(self.spherical_panels,
                                             pos=(0.2, 0.1), 
                                             states=["Draw Spherical Panels"],
                                             c="w",
                                             bc="k",
                                             font="Calco",
                                             )


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
        self.bc = (p1+p2+p3)/3

        self.bf = bf
        self.ncf = ncf

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

        X[var_idx["df"]] = 4

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
        optimizer.initialize_optimizer(X, "LM", 0.6)

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
                        bc-0.05*t1,
                        bc+0.05*t1,
                        res=1,
                        scale=1,
                        lw=1,
                        c='w',
                        alpha=1.0)


        
        lt2 = vd.Lines(
                        bc-0.05*t2,
                        bc+0.05*t2,
                        res=1,
                        scale=1,
                        lw=1,
                        c='b',
                        alpha=1.0)
        
        self.add_to_scene("lt1", lt1)
        self.add_to_scene("lt2", lt2)
        

        self.plotter.render()

        
        