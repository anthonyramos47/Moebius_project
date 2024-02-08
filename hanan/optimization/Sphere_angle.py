# Planarity constraint implementation

import numpy as np
from hanan.geometry.mesh import Mesh
from hanan.geometry.utils import vec_dot
from hanan.optimization.constraint import Constraint

class Sphere_angle(Constraint):

    def __init__(self) -> None:
        """ Sphere angle minimization
        E = \sum{adj(ci, cj) \in E} || ri^2 + rj^2 - d^2 - 2*r1*r2 ||^2; d**2 =  <cj - ci, cj - ci>
        where, 
            ci, cj,- center of the spheres i and j that are adjacent to a given edge
            ri, rj.- sphere radius 
        """
        super().__init__()
        self.name = "Sphere_Angles" # Name of the constraint
        self.nE = None # Edges number
        self.f1 = None # First face of the edge
        self.f2 = None # Second face of the edge

    def initialize_constraint(self, X, var_indices, V, F) -> np.array:
        # Input
        # X: variables 
        # var_indices: indices of the variables
        # V: Vertices
        # F: Faces
    
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

        # Define indices indices
        self.var_idx = var_indices
    
        self.const_idx = {  "ang"  : np.arange( 0                  , self.nE)
                        }
        
        # Number of variables
        self.var = len(X)



        

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
        sph_c, sph_r = self.uncurry_X(X, "sph_c", "sph_r")

        sph_c = sph_c.reshape(-1, 3)

        # Get spheres per corresponding edge ij
        ri = sph_r[self.f1]
        rj = sph_r[self.f2]

        ci = sph_c[self.f1]
        cj = sph_c[self.f2]

        d = np.linalg.norm(cj - ci, axis=1)

        # Sphere indices
        sph_idx = var_idx["sph_c"]
        
        i =  3 * np.repeat(self.f1, 3) + np.tile(range(3), len(self.f1))
        j =  3 * np.repeat(self.f2, 3) + np.tile(range(3), len(self.f2))
        
        # E : || ri^2 + rj^2 - d^2 - 2r1r2||^2; d**2 =  <cj - ci, cj - ci>
        # d ci => 2*(cj - ci).flatten
        self.add_derivatives(c_idx["ang"].repeat(3), sph_idx[i], 2*(cj - ci).flatten())
        # d cj => -2*(cj - ci).flatten
        self.add_derivatives(c_idx["ang"].repeat(3), sph_idx[j], -2*(cj - ci).flatten())
        # d ri => 2*ri - 2*rj
        self.add_derivatives(c_idx["ang"], var_idx["sph_r"][self.f1], 2*ri - 2*rj)
        # d rj => 2*rj - 2*ri
        self.add_derivatives(c_idx["ang"], var_idx["sph_r"][self.f2], 2*rj - 2*ri)

        self.set_r(c_idx["ang"], ri**2 + rj**2 - d**2 - 2*ri*rj) 

        
        