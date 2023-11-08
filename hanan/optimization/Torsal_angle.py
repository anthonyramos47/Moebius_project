# Planarity constraint implementation

import numpy as np
import geometry as geo
from optimization.constraint import Constraint

class Torsal_angle(Constraint):

    def __init__(self) -> None:
        """ Torsal directions constraint energy
        E = \sum{f \in F} || n_t1 . e_c ||^2 + || n_t1 . t1 ||^2 + || n_f1 . tt1 ||^2 + || n_t1 . n_t1 - 1 ||^2
            +             || n_t2 . e_c ||^2 + || n_t2 . t2 ||^2 + || n_f2 . tt2 ||^2 + || n_t2 . n_t2 - 1 ||^2
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
        # X: variables [ e   | a | b | n_t  | d_i ] 
        # X  size      [ 3*V | F | F | 3*F  | F   ]
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
        # E1 = || c0^2 - 0.7 + u^2 ||^2
        # E2 = || c0 - nt1.nt2 ||^2
        self.const_idx = {  "E1"  : np.arange( 0                  , self.nF),
                            "E2"  : np.arange( self.nF            , 2*self.nF),
                    }
        
        # Number of variables
        self.var = len(X)

       

    def compute(self, X, F) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                F: Faces
        """
        
        # indices vars
        v_idx = self.var_idx
        c_idx = self.const_idx

        nt1, nt2, c0, u = self.uncurry_X(X, "nt1", "nt2", "c0", "u")

        # Unflatten nt1, nt2
        nt1uf = nt1.reshape(-1, 3)
        nt2uf = nt2.reshape(-1, 3)

        # dc0( E1)  =  dc0 ( c0^2 - 0.7 + u^2 ) = 2 c0
        #           J[c_idx["E1"], v_idx["c0"]] = 2*c0
        self.add_derivatives(c_idx["E1"], v_idx["c0"], 2*c0)

        # du( E1)   =  du ( c0^2 - 0.7 + u^2 ) = 2 u
        #           J[c_idx["E1"], v_idx["u"]] = 2*u
        self.add_derivatives(c_idx["E1"], v_idx["u"], 2*u)

        # dc0( E2)  =  dc0 ( c0 - nt1.nt2 ) = 1
        #           J[c_idx["E2"], v_idx["c0"]] = 1
        self.add_derivatives(c_idx["E2"], v_idx["c0"], np.ones(len(c_idx["E2"])))

        # dnt1( E2) =  dnt1 ( c0 - nt1.nt2 ) = -nt2
        #           J[c_idx["E2"].repeat(3), v_idx["nt1"]] = -nt2.flatten()
        self.add_derivatives(c_idx["E2"].repeat(3), v_idx["nt1"], -nt2)

        # dnt2( E2) =  dnt2 ( c0 - nt1.nt2 ) = -nt1
        #           J[c_idx["E2"].repeat(3), v_idx["nt2"]] = -nt1.flatten()
        self.add_derivatives(c_idx["E2"].repeat(3), v_idx["nt2"], -nt1)

        # r of E1 
        self.set_r(c_idx["E1"], c0**2 - 0.5 + u**2)

        # r of E2
        self.set_r(c_idx["E2"], c0 - np.sum(nt1uf*nt2uf, axis=1))