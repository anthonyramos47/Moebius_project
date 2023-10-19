# Planarity constraint implementation

import numpy as np
import geometry as geo
from optimization.constraint import Constraint

class HyperbolicLC(Constraint):

    def __init__(self) -> None:
        """ Hyperbolic constraint energy
        E = \sum{f \in F} || A^2 - 4[vij, vik, ec][eij, eik, ec] - delta^2 ||^2 + \sum_{f \in F} || A - [vij, eik, ec] - [eij, vik, ec] ||^2
        """
        super().__init__()
        self.nV = None # Number of vertices
        self.nF = None # Number of faces
        self.fvij = None # List of the edge vectors per each face
        self.nc = None # List of the norms of ec
        self.derx = None # Derivative matrix coordinate x
        self.dery = None # Derivative matrix coordinate y
        self.derz = None # Derivative matrix coordinate z
    
    def cross_id(self, v, coord, type=0):
        """ Function tha retunr the cross product with coordinate vectors like [1, 0, 0], [0, 1, 0], [0, 0, 1]
            type 0
            V x [1,0,0]
            type 1
            [1,0,0] x V

        """

        if coord=='x':
            cross = v[:, [0, 2, 1]] 
            cross[:, 0] = 0
            cross[:, 2] *= -1 
        elif coord=='y':
            cross = v[:, [2, 1, 0]]
            cross[:, 0] *= -1
            cross[:, 1] = 0
        elif coord=='z':
            cross = v[:, [1, 0, 2]]
            cross[:, 1] *= -1
            cross[:, 2] = 0
        
        if type==1:
            cross *= -1

        return cross

        
        

    def initialize_constraint(self, X, V, F, e_i, delta, w=1 ) -> np.array:
        # Input
        # X: variables [ e| A | delta]
        # V: Vertices
        # F: Faces        
        self.w = w
        # Number of vertices
        self.nV = len(V)
        
        # Number of faces
        self.nF = len(F)

        # Number of constraints
        self.const = 2*self.nF
        
        # Number of variables
        self.var = len(X)
        
        # Compute the directions at the barycenters
        ec = np.sum( e_i[F], axis = 1)/3

        # Compute the edge vectors per each face
        vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]

        # Compute the edge vectors per each face
        self.fvij = np.empty((self.nF, 2, 3), dtype=float)

        # Compute the edge vectors per each face
        self.fvij[:,0] = vj - vi
        self.fvij[:,1] = vk - vi

        # Set up X 
        eij = e_i[F[:,1]] - e_i[F[:,0]]
        eik = e_i[F[:,2]] - e_i[F[:,0]]
    
        # A = [vij, eik, ec] + [eij, vik, ec], where [ , , ] denotes determinant
        # A = det1 +  det2
        eikXec = np.cross(eik, ec)
        vikXec = np.cross(self.fvij[:,1], ec)

        det1 = np.sum(self.fvij[:,0]*eikXec, axis=1)
        det2 = np.sum(eij*vikXec, axis=1)
        
        # Set up A
        X[3*self.nV: 3*self.nV + self.nF] = det1 + det2 

        # Set up delta
        X[3*self.nV + self.nF:] = delta

        return X 

        
    def _compute(self, X, F) -> None:

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
        ec = np.sum( e[F], axis = 1)/3

        # Compute the edge vectors per each face
        vij = self.fvij[:, 0]
        vik = self.fvij[:, 1]

        # Compute the directions per vertex in the faces       
        eij = e[F[:,1]] - e[F[:,0]]
        eik = e[F[:,2]] - e[F[:,0]]

        # eik x ec
        eikXec = np.cross(eik, ec)
        # eij x ec 
        #eijXec = np.cross(eij, ec)

        # ones x ec 
        idXec_x = self.cross_id(ec, 'x', 1)
        idXec_y = self.cross_id(ec, 'y', 1)
        idXec_z = self.cross_id(ec, 'z', 1)

        # eik x dec = eik x (1, 0, 0)/3
        eikXdec_x = self.cross_id(eik/3, 'x')
        eikXdec_y = self.cross_id(eik/3, 'y')
        eikXdec_z = self.cross_id(eik/3, 'z')
        
        # eij x dec 
        eijXdec_x = self.cross_id(eij/3, 'x')
        eijXdec_y = self.cross_id(eij/3, 'y')
        eijXdec_z = self.cross_id(eij/3, 'z')

        # vik x dec 
        vikXdec_x = self.cross_id(vik/3, 'x')
        vikXdec_y = self.cross_id(vik/3, 'y')
        vikXdec_z = self.cross_id(vik/3, 'z')

        # [vij, vik, ec] = det1 
        vikXec = np.cross(vik, ec)
        det1 = np.sum(vij*vikXec, axis=1)

        # [eij, eik, ec] = det2
        eikXec = np.cross(eik, ec)
        det2 = np.sum(eij*eikXec, axis=1)

        # second term
        sc_term_x = ( np.sum(vij*vikXdec_x, axis=1))*det2
        sc_term_y = ( np.sum(vij*vikXdec_y, axis=1))*det2
        sc_term_z = ( np.sum(vij*vikXdec_z, axis=1))*det2
        
        # d ei
        J[range(self.nF), 3*F[:,0]    ] = -4* ((-eikXec[:,0] + np.sum( eij*(- idXec_x + eikXdec_x), axis=1) )*det1 + sc_term_x)
        J[range(self.nF), 3*F[:,0] + 1] = -4* ((-eikXec[:,1] + np.sum( eij*(- idXec_y + eikXdec_y), axis=1) )*det1 + sc_term_y)
        J[range(self.nF), 3*F[:,0] + 2] = -4* ((-eikXec[:,2] + np.sum( eij*(- idXec_z + eikXdec_z), axis=1) )*det1 + sc_term_z)

        # d ej
        J[range(self.nF), 3*F[:,1]    ] = -4* (( eikXec[:,0] + np.sum( eij*( eikXdec_x), axis=1) )*det1 + sc_term_x)
        J[range(self.nF), 3*F[:,1] + 1] = -4* (( eikXec[:,1] + np.sum( eij*( eikXdec_y), axis=1) )*det1 + sc_term_y)
        J[range(self.nF), 3*F[:,1] + 2] = -4* (( eikXec[:,2] + np.sum( eij*( eikXdec_z), axis=1) )*det1 + sc_term_z)

        # d ek
        J[range(self.nF), 3*F[:,2]    ] = -4* (  np.sum( eij*(idXec_x + eikXdec_x), axis=1 )*det1 + sc_term_x)
        J[range(self.nF), 3*F[:,2] + 1] = -4* (  np.sum( eij*(idXec_y + eikXdec_y), axis=1 )*det1 + sc_term_y)
        J[range(self.nF), 3*F[:,2] + 2] = -4* (  np.sum( eij*(idXec_z + eikXdec_z), axis=1 )*det1 + sc_term_z)
        
        # d A
        J[:self.nF, 3*self.nV: 3*self.nV + self.nF] = np.diag(2 * A)
        # d delta
        J[:self.nF, 3*self.nV + self.nF : 3*self.nV + 2*self.nF] = np.diag(-2*delta)
        
        # r 
        r[:self.nF] = A**2 - 4*det1*det2 -delta**2
    
        
        # d ei 
        J[range(self.nF, 2*self.nF), 3*F[:, 0]    ] = -( np.sum(vij*( - idXec_x + eikXdec_x), axis=1)  - np.sum(vik*(-idXec_x + eijXdec_x), axis=1 ))
        J[range(self.nF, 2*self.nF), 3*F[:, 0] + 1] = -(np.sum(vij*( - idXec_y + eikXdec_y), axis=1)  - np.sum(vik*(-idXec_y + eijXdec_y), axis=1 ))
        J[range(self.nF, 2*self.nF), 3*F[:, 0] + 2] = -(np.sum(vij*( - idXec_z + eikXdec_z), axis=1)  - np.sum(vik*(-idXec_z + eijXdec_z), axis=1 ))

        # d eij
        J[range(self.nF, 2*self.nF), 3*F[:, 1]    ] = -(np.sum(vij*( eikXdec_x), axis=1)  - np.sum(vik*(idXec_x + eijXdec_x), axis=1 ))
        J[range(self.nF, 2*self.nF), 3*F[:, 1] + 1] = -(np.sum(vij*( eikXdec_y), axis=1)  - np.sum(vik*(idXec_y + eijXdec_y), axis=1 ))
        J[range(self.nF, 2*self.nF), 3*F[:, 1] + 2] = -(np.sum(vij*( eikXdec_z), axis=1)  - np.sum(vik*(idXec_z + eijXdec_z), axis=1 ))

        # d eik
        J[range(self.nF, 2*self.nF), 3*F[:, 2]    ] = -(np.sum(vij*( idXec_x + eikXdec_x), axis=1)  - np.sum(vik*(eijXdec_x), axis=1 ))
        J[range(self.nF, 2*self.nF), 3*F[:, 2] + 1] = -(np.sum(vij*( idXec_y + eikXdec_y), axis=1)  - np.sum(vik*(eijXdec_y), axis=1 ))
        J[range(self.nF, 2*self.nF), 3*F[:, 2] + 2] = -(np.sum(vij*( idXec_z + eikXdec_z), axis=1)  - np.sum(vik*(eijXdec_z), axis=1 ))

        # # # dA 
        J[self.nF: 2*self.nF, 3*self.nV : 3*self.nV + self.nF] = np.eye(self.nF)
        r[self.nF: 2*self.nF] = A - np.sum(vij*eikXec, axis=1) - np.sum(eij*vikXec, axis=1) 

        # Update J
        self.J =  J
        # Update r
        self.r =  r
    






        
       