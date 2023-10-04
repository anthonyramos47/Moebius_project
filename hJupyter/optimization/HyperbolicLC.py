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
        self.nc = None # List of the norms of ec
        self.e = None # List of the directions per each vertex
        
        
        
        

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
        self.const = 2*self.nF 

        # Number of variables
        self.var = self.nV*3 + self.nF*2

        # set directions
        self.e = e_i 
    
        # Compute the directions at the barycenters
        ec = np.sum( e_i[F], axis = 1)

        # Compute the norms of ec
        self.nc = np.linalg.norm(ec, axis=1)

        # Compute the edge vectors per each face
        vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]

        # Compute the edge vectors per each face
        self.fvij = np.empty((self.nF, 2, 3), dtype=float)

        # Compute the edge vectors per each face
        self.fvij[:,0] = vi - vj
        self.fvij[:,1] = vk - vj

        # Set up X 
        X[:3*self.nV] = V.flatten()

        eij = e_i[F[:,1]] - e_i[F[:,0]]
        eik = e_i[F[:,2]] - e_i[F[:,0]]
    
        # A = [vij, eik, ec] + [eij, vik, ec], where [ , , ] denotes determinant
        # A = det1, det2
        
        eikXec = np.cross(eik, ec)
        vikXec = np.cross(self.fvij[:,1], ec)

        det1 = np.sum(self.fvij[:,0]*eikXec, axis=1)
        det2 = np.sum(eij*vikXec, axis=1)
        
        # Set up A
        X[3*self.nV:3*self.nV + self.nF] = det1 + det2 

        # Set up delta
        X[3*self.nV + self.nF:] = 1

            

    def compute(self, X, F) -> None:

        # Init J
        J = np.zeros((self.const, self.var), dtype=np.float64)

        # Init r
        r = np.zeros(self.const, dtype=np.float64)

        # Get e_i directions
        e = X[:3*self.nV].reshape(self.nV, 3)

        # Get A 
        A = X[3*self.nV:3*self.nV + self.nF]

        # Get delta
        delta = X[3*self.nV + self.nF:]

        # Compute e_c
        ec = np.sum( e[F], axis = 1)

    
        # Compute the edge vectors per each face
        vij = self.fvij[:, 0]
        vik = self.fvij[:, 1]

        # Compute the directions per vertex in the faces       
        eij = e[F[:,1]] - e[F[:,0]]
        eik = e[F[:,2]] - e[F[:,0]]

        # Compute the norms of ec
        nc = self.nc

        # eik x ec
        eikXec = np.cross(eik, ec)

        # ones x ec
        idXec = np.cross(np.ones_like(ec), ec)

        # eik x dec
        eikXdec = np.cross(eik, np.ones_like(eik))/nc[:, None]

        # vik x dec
        vikXdec = np.cross(vik, np.ones_like(vik))/nc[:, None]

        # [vij, vik, ec] = det1 
        vikXec = np.cross(vik, ec)
        det1 = np.sum(vij*vikXec, axis=1)

        # [eij, eik, ec] = det2
        eikXec = np.cross(eik, ec)
        det2 = np.sum(eij*eikXec, axis=1)

        # second term
        sc_term = (vij[i_f]*vikXdec[i_f])*det2

        for i_f in range(self.nF):

            # d ei 
            # J[i_f, 3*F[i_f,0]]   = -4* ((-eikXdec[i_f][0] + eij[i_f, 0]*(- idXec[i_f][0] + eikXdec[i_f][0]))*det1 + (vij[i_f, 0]*vikXdec[i_f][0])*det2)
            # J[i_f, 3*F[i_f,0]+1] = -4* ((-eikXdec[i_f][1] + eij[i_f, 1]*(- idXec[i_f][1] + eikXdec[i_f][1]))*det1 + (vij[i_f, 1]*vikXdec[i_f][1])*det2)
            # J[i_f, 3*F[i_f,0]+2] = -4* ((-eikXdec[i_f][2] + eij[i_f, 2]*(- idXec[i_f][2] + eikXdec[i_f][2]))*det1 + (vij[i_f, 2]*vikXdec[i_f][2])*det2)

            # Vectorized version
            J[i_f, 3*F[i_f,0]:3*F[i_f,0]+3] = -4* ((-eikXdec[i_f] + eij[i_f]*(- idXec[i_f] + eikXdec[i_f]))*det1 + sc_term)

            # d ej 
            J[i_f, 3*F[i_f,1]:3*F[i_f,1]+3] = -4* (( eikXdec[i_f] + eij[i_f]*(  idXec[i_f] + eikXdec[i_f]))*det1 + sc_term)

            # d ek
            J[i_f, 3*F[i_f,2]:3*F[i_f,2]+3] = -4* (  eij[i_f]*(idXec[i_f] + eikXdec[i_f])*det1 + sc_term)

            # d A
            J[i_f, 3*self.nV + i_f] = A  

            # d delta
            J[i_f, 3*self.nV + self.nF + i_f] = delta 



        
        
    






        
       