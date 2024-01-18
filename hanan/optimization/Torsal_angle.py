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
        self.nV = None # Number of vertices
        self.nF = None # Number of faces
        self.var_idx = None # Dictionary of indices of variables
        self.const_idx = None # Dictionary of indices of constraints


    
    def initialize_constraint(self, X, var_indices, V, F, w=1 ) -> np.array:
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

        # Define indices indices
        self.var_idx = var_indices

        # E1 = || nt1.nt2^2  - cos(60) + u^2 ||^2  <=> nt1.nt2 <= cos(60)
        # E2 = || nt1.nt2  - v^2           ||^2  <=> nt1.nt2 >= cos(90)
        self.const_idx = {  "E1"  : np.arange( 0                  , self.nF)
                            #"E2"  : np.arange( self.nF            , 2*self.nF),
                    }
        
        # Number of variables
        self.var = len(X)

       

    def compute(self, X) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                F: Faces
        """
        
        # indices vars
        v_idx = self.var_idx
        # indices constraints
        c_idx = self.const_idx

        # Get variables of interest
        nt1, nt2, u = self.uncurry_X(X, "nt1", "nt2", "u")

        # Unflatten nt1, nt2
        nt1uf = nt1.reshape(-1, 3)
        nt2uf = nt2.reshape(-1, 3)

        dotnt1_nt2= vec_dot(nt1uf, nt2uf)

        # d nt1 (E1) = d nt1(nt1.nt2 - cos(60) + mu^2) = nt2
        self.add_derivatives(c_idx["E1"].repeat(3), v_idx["nt1"], (2*dotnt1_nt2[:,None]*nt2uf).flatten())

        # d nt2 (E1) = d nt2(nt1.nt2 - cos(60) + mu^2) = nt1
        self.add_derivatives(c_idx["E1"].repeat(3), v_idx["nt2"], (2*dotnt1_nt2[:,None]*nt1uf).flatten())

        # d u (E1) = d nt2(nt1.nt2 - cos(60) + u^2) = 2u
        self.add_derivatives(c_idx["E1"], v_idx["u"], 2*u)

        self.set_r(c_idx["E1"], dotnt1_nt2**2 - np.cos(60*np.pi/180)**2 + u**2 )


        # # d nt1 (E2) = d nt1(nt1.nt2 - v^2) = nt2
        # self.add_derivatives(c_idx["E2"].repeat(3), v_idx["nt1"], (2*vec_dot(nt1uf, nt2uf)[:,None]*nt2uf).flatten())

        # # d nt2 (E2) = d nt2(nt1.nt2 - v^2) = nt1
        # self.add_derivatives(c_idx["E2"].repeat(3), v_idx["nt2"], (2*vec_dot(nt1uf, nt2uf)[:,None]*nt1uf).flatten())

        # # d v (E2) = d v(nt1.nt2 - v^2) = -2v
        # self.add_derivatives(c_idx["E2"], v_idx["v"], -2*v)

        # self.set_r(c_idx["E2"], vec_dot(nt1uf, nt2uf)**2 - - np.cos(90*np.pi/180)**2 - v**2 )