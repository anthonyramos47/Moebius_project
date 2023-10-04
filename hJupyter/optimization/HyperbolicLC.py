# Planarity constraint implementation

import numpy as np
import geometry as geo
from optimization.constraint import Constraint

class HyperbolicLC(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.nV = None # Number of vertices
        self.nF = None # Number of faces
        self.fvij = None # List of the edge vectors per each face
        self.ec = None # List of the directions at the barycenters
        self.nc = None # List of the norms of ec
        
        
        
        

    def initialize_constraint(self, X, V, F, e_i ) -> None:
        # Input
        # X: variables [ e| A | delta]
        # V: Vertices
        # F: Faces        

        # Number of vertices
        self.nV = len(V)
        
        # Number of faces
        self.nF = len(F)

        # Number of constraints
        self.const = self.nF 

        # Number of variables
        self.var = self.nV*3 + self.nF*2
    
        # Compute the directions at the barycenters
        self.ec = np.sum( e_i[F], axis = 1)

        # Compute the norms of ec
        self.nc = np.linalg.norm(self.ec, axis=1)

        # Compute the edge vectors per each face
        vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]

        # Compute the edge vectors per each face
        self.fvij = np.empty((self.nF, 2, 3), dtype=float)

        self.fvij[:,0] = vi - vj
        self.fvij[:,1] = vk - vj
            

    def compute(self, X, F) -> None:

        # Init J
        J = np.zeros((self.const, self.var), dtype=np.float64)

        # Init r
        r = np.zeros(self.const, dtype=np.float64)

        # Loop over faces
        for i_f in range(self.nF):

            vij = self.fvij[i_f, 0]


        
       