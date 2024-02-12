# Planarity constraint implementation
import numpy as np
from hanan.geometry.utils import vec_dot
from hanan.optimization.constraint import Constraint

class Torsal_angle(Constraint):

    def __init__(self) -> None:
        """ Torsal directions constraint energy
        E = \sum{f \in F} ||nt1.nt2 - cos(60) + mu^2||^2  + || nt1.nt2 - nu^2 ||^2 
        where, 
            n_t .- normal of torsal plane.
            t   .- torsal direction in the face of triangular mesh T.
            tt  .- torsal direction in the face of triangular mesh \tilde{T} [second envelope]
            e_c .- direction of the line congruence at the barycenter of the face.
        """
        super().__init__()
        self.name = "Torsal_Angles" 
        self.nV = None # Number of vertices
        self.nF = None # Number of faces
        self.t_ang = None # Torsal angle target
    
    def initialize_constraint(self, X, var_indices, V, F, t_ang, w=1 ) -> np.array:
        # Input
        # X: variables 
        # var_indices: dictionary of indices of variables
        # V: Vertices
        # F: Faces
        # bf: circumcenters of the faces
        # ncf: normals of the circumcenters

        # Number of vertices
        self.nV = len(V)
        
        # Number of faces
        self.nF = len(F)

        # Number of constraints 4*|F|
        self.const = 2*self.nF

        # Torsal angle target
        self.t_ang = t_ang

        # E1 = || nt1.nt2^2  - cos(60) + u^2 ||^2  <=> nt1.nt2 <= cos(60)
        self.add_constraint("E1", self.nF)

       

    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                F: Faces
        """
        
        # indices vars
        v_idx = var_idx
        # indices constraints
        c_idx = self.const_idx

        # Get variables of interest
        nt1, nt2, u = self.uncurry_X(X, var_idx, "nt1", "nt2", "alpha")

        # Unflatten nt1, nt2
        nt1uf = nt1.reshape(-1, 3)
        nt2uf = nt2.reshape(-1, 3)

        dotnt1_nt2= vec_dot(nt1uf, nt2uf)

        # d nt1 (E1) = d nt1(nt1.nt2 - cos(60) + mu^2) = nt2
        self.add_derivatives(c_idx["E1"].repeat(3), v_idx["nt1"], (2*dotnt1_nt2[:,None]*nt2uf).flatten())

        # d nt2 (E1) = d nt2(nt1.nt2 - cos(60) + mu^2) = nt1
        self.add_derivatives(c_idx["E1"].repeat(3), v_idx["nt2"], (2*dotnt1_nt2[:,None]*nt1uf).flatten())

        # d u (E1) = d nt2(nt1.nt2 - cos(60) + u^2) = 2u
        #self.add_derivatives(c_idx["E1"], v_idx["alpha"], 2*u)
        #self.set_r(c_idx["E1"], dotnt1_nt2**2 - np.cos(65*np.pi/180)**2 + u**2 )
        self.set_r(c_idx["E1"], dotnt1_nt2**2 - np.cos(self.t_ang*np.pi/180)**2  )

