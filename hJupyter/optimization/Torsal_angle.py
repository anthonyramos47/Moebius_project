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


    
    def initialize_constraint(self, X, V, F, w=1 ) -> np.array:
        # Input
        # X: variables [ e   | a | b | n_t  | d_i ] 
        # X  size      [ 3*V | F | F | 3*F  | F   ]
        # V: Vertices
        # F: Faces
        # bf: circumcenters of the faces
        # ncf: normals of the circumcenters
        # w: weight

        # Set weight
        self.w = w

        # Number of vertices
        self.nV = len(V)
        
        # Number of faces
        self.nF = len(F)

        # Number of constraints 4*|F|
        self.const = 2*self.nF

        # Define indices indices
        self.var_idx = {    
                            "nt1": np.arange( 3*self.nV + 2*self.nF , 3*self.nV + 5*self.nF),
                            "nt2": np.arange( 3*self.nV + 7*self.nF , 3*self.nV + 10*self.nF),
                            "c0" : np.arange( 3*self.nV + 11*self.nF, 3*self.nV + 12*self.nF),
                            "u"  : np.arange( 3*self.nV + 12*self.nF, 3*self.nV + 13*self.nF),
                    }

        # E1 = || c0^2 - 0.7 + u^2 ||^2
        # E2 = || c0 - nt1.nt2 ||^2
        self.const_idx = {  "E1"  : np.arange( 0                  , self.nF),
                            "E2"  : np.arange( self.nF            , 2*self.nF),
                    }
        
        # Number of variables
        self.var = len(X)

       

    def compute(self, X, F) -> None:

        # Init J
        J = np.zeros((self.const, self.var), dtype=np.float64)

        # Init r
        r = np.zeros(self.const, dtype=np.float64)

        J, r = self.fill_J(X, J, r, F)

        # Update J
        self.J =  J
        # Update r
        self.r =  r

    def fill_J(self, X, J, r, F):

        # indices vars
        v_idx = self.var_idx
        c_idx = self.const_idx

        nt1, nt2, c0, u = self.uncurry_variables(X)

        # dc0( E1)  =  dc0 ( c0^2 - 0.7 + u^2 ) = 2 c0
        J[c_idx["E1"], v_idx["c0"]] = 2*c0 
        # du( E1)   =  du ( c0^2 - 0.7 + u^2 ) = 2 u
        J[c_idx["E1"], v_idx["u"]] = 2*u

        # r of E1 
        r[c_idx["E1"]] = c0**2 - 0.5 + u**2

        # dc0( E2)  =  dc0 ( c0 - nt1.nt2 ) = 1
        J[c_idx["E2"], v_idx["c0"]] = 1
        # dnt1( E2) =  dnt1 ( c0 - nt1.nt2 ) = -nt2
        J[c_idx["E2"].repeat(3), v_idx["nt1"]] = -nt2.flatten()
        # dnt2( E2) =  dnt2 ( c0 - nt1.nt2 ) = -nt1
        J[c_idx["E2"].repeat(3), v_idx["nt2"]] = -nt1.flatten()


        # r of E2
        r[c_idx["E2"]] = c0 - np.sum(nt1*nt2, axis=1)

        
        return J, r

    def uncurry_variables(self, X):

        v_idx = self.var_idx

        # Get n_t
        nt1 = X[v_idx["nt1"]].reshape(-1, 3)

        # Get n_t2
        nt2 = X[v_idx["nt2"]].reshape(-1, 3)

        # Get c0
        c0 = X[v_idx["c0"]]

        # Get u
        u = X[v_idx["u"]]

        return nt1, nt2, c0, u 
   

        
       