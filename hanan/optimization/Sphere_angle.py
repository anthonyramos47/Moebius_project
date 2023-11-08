# Planarity constraint implementation

import numpy as np
from geometry.mesh import Mesh
from geometry.utils import vec_dot
from optimization.constraint import Constraint

class Sphere_angle(Constraint):

    def __init__(self) -> None:
        """ Sphere angle minimization
        E = \sum{adj(ci, cj) \in E} || (ci.cj - (ci + cj).v + v^2 - 1)/(ri * rj)  ||
        where, 
            ci, cj,- center of the spheres i and j that are adjacent to a given edge
            v.- is a random vertex in the choosen edge
            ri, rj.- sphere radius 
        """
        super().__init__()
        self.bc = None # Barycenters
        self.nc = None # Normal of the barycenters
        self.v = None # Vertices of the edges
        self.v0 = None # Vertex of the faces
        self.nE = None # Edges number
        self.radius = None # Radius of the spheres
        self.f1 = None # First face of the edge
        self.f2 = None # Second face of the edge

        

    
    def initialize_constraint(self, X, var_indices, V, F, bf, ncf) -> np.array:
        # Input
        # X: variables 
        # var_indices: indices of the variables
        # V: Vertices
        # F: Faces
        # bf: circumcenters of the faces
        # ncf: normals of the circumcenters
    
        # Set circumcenters
        self.bf = bf

        # Set circumcenters axis
        self.ncf = ncf

        # create mesh 
        mesh = Mesh()
        mesh.make_mesh(V, F)

        inner_edges = mesh.inner_edges()

        # Number of edges
        self.nE = len(inner_edges)

        # Number of constraints |IE| inner edges
        self.const = self.nE 

        # Get adjacent faces
        auxf1, auxf2 = mesh.edge_faces() 

        self.f1 = auxf1[inner_edges]
        self.f2 = auxf2[inner_edges]

        # Get vertices of the edges
        auxv, _ = mesh.edge_vertices()


        self.v = V[auxv[inner_edges]]

        # Define indices indices
        self.var_idx = var_indices
    
        self.const_idx = {  "ang"  : np.arange( 0                  , self.nE)
                        }
        
        # Number of variables
        self.var = len(X)

        df = self.uncurry_X(X, "df")

        # Get vertex from all faces
        self.v0 = V[F[:, 0]]
        # Compute radius
        c = bf + df[:,None]*ncf

        # Compute radius
        self.radius = np.linalg.norm(self.v0 - c, axis=1)

        

    def compute(self, X) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                F: Faces
        """
        # Get variables indices
        var_idx = self.var_idx

        # Get constraints indices
        c_idx = self.const_idx

        # Get distance from baricenters to center of spheres
        df = self.uncurry_X(X, "df")

        ri = self.radius[self.f1]
        rj = self.radius[self.f2]

        cj = self.bf[self.f2] + df[self.f2, None]*self.ncf[self.f2]
        ci = self.bf[self.f1] + df[self.f1, None]*self.ncf[self.f1] 

        nci = self.ncf[self.f1]
        ncj = self.ncf[self.f2]

        v = self.v

        # print("ri*rj", (ri*rj).shape)
        # print("nci", nci.shape)
        # print("cj", cj.shape)
        # print("v", v.shape)
        # print("nci.(cj - v)", vec_dot(nci, (cj - v) ).shape)
        # print("di", (vec_dot(nci, (cj - v) )/(ri*rj)[:, None]).shape)
        # E : || (ci.cj - (ci + cj).v + v^2 - 1)/(ri * rj)  ||^2
        # d di => nci.(cj - v)/ ri*rj
        self.add_derivatives(c_idx["ang"], var_idx["df"][self.f1], vec_dot(nci, (cj - v) )/(ri*rj))
        # d dj => ncj.(ci - v)/ ri*rj
        self.add_derivatives(c_idx["ang"], var_idx["df"][self.f2], vec_dot(ncj, (ci - v) )/(ri*rj))

        self.set_r(c_idx["ang"], (vec_dot(ci, cj) - vec_dot(ci + cj, v) + vec_dot(v, v))/(ri*rj) - 1)

        self.radius = np.linalg.norm(self.v0 - (self.bf + df[:,None]*self.ncf), axis=1)

        
        