# Planarity constraint implementation

import numpy as np
from hanan.geometry.mesh import Mesh
from hanan.geometry.utils import vec_dot, unit
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
        self.ne = None # Normal of the edge

    def initialize_constraint(self, X, var_indices, ne, evi, evj, inner_edges, f1, f2) -> np.array:
        # Input
        # X: variables 
        # var_indices: indices of the variables
        # V: Vertices
        # F: Faces
    
        # Number of edges
        self.nE = len(inner_edges)
        
        # Set edges normals
        self.ne = ne 

        # Set direction of the edges
        self.dire = unit(evj - evi)

        
        self.f1 = f1
        self.f2 = f2
        
        self.add_constraint("ang", self.nE)
        



        

    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                F: Faces
        """
        # Get constraints indices
        c_idx = self.const_idx

        # Get distance from baricenters to center of spheres
        sph_c, sph_r = self.uncurry_X(X, var_idx, "sph_c", "sph_r")

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
        # d ci 
        self.add_derivatives(c_idx["ang"].repeat(3),     sph_idx[i],  (4*(ri**2 + rj**2 - d**2)[:,None]*(cj - ci)).flatten())
        # d cj 
        self.add_derivatives(c_idx["ang"].repeat(3),     sph_idx[j],  (-4*(ri**2 + rj**2 - d**2)[:,None]*(cj - ci)).flatten())
        # d ri 
        self.add_derivatives(c_idx["ang"], var_idx["sph_r"][self.f1], (4*(ri**2 + rj**2 - d**2)*ri - 8 * ri * rj**2) )
        # d rj
        self.add_derivatives(c_idx["ang"], var_idx["sph_r"][self.f2], (4*(ri**2 + rj**2 - d**2)*rj - 8 * ri**2 * rj) )

        self.set_r(c_idx["ang"], ((ri**2 + rj**2 - d**2)**2 - 4*ri**2*rj**2) ) 


        # E : || ne.(cj - ci)  + dire.(cj - ci) ||^2

        # print("ne", self.ne)
        # print("dire", self.dire)
        
        # # d ci
        # self.add_derivatives(c_idx["reg"].repeat(3), sph_idx[i], (- self.ne - self.dire).flatten())
        # # d cj
        # self.add_derivatives(c_idx["reg"].repeat(3), sph_idx[j], (self.ne + self.dire).flatten())

        # self.set_r(c_idx["reg"], (vec_dot(self.ne, (cj - ci)) + vec_dot(self.dire, (cj - ci))) )



        
        