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
        self.e = None # List of the directions per each vertex
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

        
        

    def initialize_constraint(self, X, V, F, e_i, delta ) -> np.array:
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
        
        #  2*self.nF 
        # Number of variables
        self.var = len(X)
        # self.nV*3 + self.nF*2

        # set directions
        self.e = e_i 

        # Compute the directions at the barycenters
        ec = np.sum( e_i[F], axis = 1)/3

        # Compute the norms of ec
        #self.nc = np.linalg.norm(ec, axis=1)

        # Compute the edge vectors per each face
        vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]

        # Compute the edge vectors per each face
        self.fvij = np.empty((self.nF, 2, 3), dtype=float)

        # Compute the edge vectors per each face
        self.fvij[:,0] = vj - vi
        self.fvij[:,1] = vk - vi

        # Set up X 
        #X[:3*self.nV] = V.flatten()

        eij = e_i[F[:,1]] - e_i[F[:,0]]
        eik = e_i[F[:,2]] - e_i[F[:,0]]
    
        # A = [vij, eik, ec] + [eij, vik, ec], where [ , , ] denotes determinant
        # A = det1 +  det2
        eikXec = np.cross(eik, ec)
        vikXec = np.cross(self.fvij[:,1], ec)

        det1 = np.sum(self.fvij[:,0]*eikXec, axis=1)
        det2 = np.sum(eij*vikXec, axis=1)

        # Define matrix derivatives
        self.derx = np.zeros_like(eik)
        self.dery = np.zeros_like(eik)
        self.derz = np.zeros_like(eik)

        self.derx[:, 0] = 1 
        self.dery[:, 1] = 1
        self.derz[:, 2] = 1
        
        # Set up A
        X[3*self.nV: 3*self.nV + self.nF] = det1 + det2 

        # Set up delta
        X[3*self.nV + self.nF:] = delta

        return X 

            

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
        ec = np.sum( e[F], axis = 1)/3

    
        # Compute the edge vectors per each face
        vij = self.fvij[:, 0]
        vik = self.fvij[:, 1]

        # Compute the directions per vertex in the faces       
        eij = e[F[:,1]] - e[F[:,0]]
        eik = e[F[:,2]] - e[F[:,0]]

        # Compute the norms of ec
        #nc = self.nc
        # Unnormalized
        nc = np.ones(self.nF)

        # eik x ec
        eikXec = np.cross(eik, ec)
        # eij x ec 
        eijXec = np.cross(eij, ec)

        # ones x ec 
        idXec_x = self.cross_id(ec, 'x', 1)
        idXec_y = self.cross_id(ec, 'y', 1)
        idXec_z = self.cross_id(ec, 'z', 1)


        # eik x dec x      
        eikXdec_x = self.cross_id(eik, 'x')
        eikXdec_x = eikXdec_x/nc[:, None]

        # eik x dec y
        eikXdec_y = self.cross_id(eik, 'y')
        eikXdec_y = eikXdec_y/nc[:, None]

        # eik x dec z
        eikXdec_z = self.cross_id(eik, 'z')
        eikXdec_z = eikXdec_z/nc[:, None]


        # eij x dec 
        eijXdec_x = self.cross_id(eij, 'x')/nc[:, None]
        eijXdec_y = self.cross_id(eij, 'y')/nc[:, None]
        eijXdec_z = self.cross_id(eij, 'z')/nc[:, None]

        # vik x dec 
        vikXdec_x = self.cross_id(vik, 'x')/nc[:, None]
        vikXdec_y = self.cross_id(vik, 'y')/nc[:, None]
        vikXdec_z = self.cross_id(vik, 'z')/nc[:, None]


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

        for i_f in range(self.nF):

            # d ei x
            J[i_f, 3*F[i_f,0]    ] = -4* ((-eikXec[i_f][0] + eij[i_f]@(- idXec_x[i_f] + eikXdec_x[i_f]))*det1[i_f] + sc_term_x[i_f])
            # d ei y
            J[i_f, 3*F[i_f,0] + 1] = -4* ((-eikXec[i_f][1] + eij[i_f]@(- idXec_y[i_f] + eikXdec_y[i_f]))*det1[i_f] + sc_term_y[i_f])
            # d ei z
            J[i_f, 3*F[i_f,0] + 2] = -4* ((-eikXec[i_f][2] + eij[i_f]@(- idXec_z[i_f] + eikXdec_z[i_f]))*det1[i_f] + sc_term_z[i_f])

            # d ej x
            J[i_f, 3*F[i_f,1]    ] = -4* (( eikXec[i_f][0] + eij[i_f]@( eikXdec_x[i_f]))*det1[i_f] + sc_term_x[i_f])
            # d ej y
            J[i_f, 3*F[i_f,1] + 1] = -4* (( eikXec[i_f][1] + eij[i_f]@( eikXdec_y[i_f]))*det1[i_f] + sc_term_y[i_f])
            # d ej z
            J[i_f, 3*F[i_f,1] + 2] = -4* (( eikXec[i_f][2] + eij[i_f]@( eikXdec_z[i_f]))*det1[i_f] + sc_term_z[i_f])

            # d ek x
            J[i_f, 3*F[i_f,2]    ] = -4* (  eij[i_f]@(idXec_x[i_f] + eikXdec_x[i_f])*det1[i_f] + sc_term_x[i_f])
            # d ek y
            J[i_f, 3*F[i_f,2] + 1] = -4* (  eij[i_f]@(idXec_y[i_f] + eikXdec_y[i_f])*det1[i_f] + sc_term_y[i_f])
            # d ek z
            J[i_f, 3*F[i_f,2] + 2] = -4* (  eij[i_f]@(idXec_z[i_f] + eikXdec_z[i_f])*det1[i_f] + sc_term_z[i_f])

            # r
            r[i_f] = A[i_f]**2 - 4*det1[i_f]*det2[i_f] - delta[i_f]**2 


        # d A
        J[:self.nF, 3*self.nV: 3*self.nV + self.nF] = 2 * A  

        # d delta
        J[:self.nF, 3*self.nV + self.nF : 3*self.nV + 2*self.nF] = - 2 * delta 


        # A = [vij, eik, ec] + [eij, vik, ec]
        for i_f in range(self.nF):

            jidx = i_f + self.nF

            # d ei x
            J[jidx, 3*F[i_f, 0]    ] = -(vij[i_f]@( - idXec_x[i_f] + eikXdec_x[i_f])  - vik[i_f]@(-idXec_x[i_f] + eijXdec_x[i_f]))
            # d ei y
            J[jidx, 3*F[i_f, 0] + 1] = -(vij[i_f]@( - idXec_y[i_f] + eikXdec_y[i_f])  - vik[i_f]@(-idXec_y[i_f] + eijXdec_y[i_f])) 
            # d ei z
            J[jidx, 3*F[i_f, 0] + 2] = -(vij[i_f]@( - idXec_z[i_f] + eikXdec_z[i_f])  - vik[i_f]@(-idXec_z[i_f] + eijXdec_z[i_f]))

            # d eij x
            J[jidx, 3*F[i_f, 1]    ] = -(vij[i_f]@( eikXdec_x[i_f])  - vik[i_f]@(idXec_x[i_f] + eijXdec_x[i_f]))
            # d eij y
            J[jidx, 3*F[i_f, 1] + 1] = -(vij[i_f]@( eikXdec_y[i_f])  - vik[i_f]@(idXec_y[i_f] + eijXdec_y[i_f]))
            # d eij z
            J[jidx, 3*F[i_f, 1] + 2] = -(vij[i_f]@( eikXdec_z[i_f])  - vik[i_f]@(idXec_z[i_f] + eijXdec_z[i_f]))


            # d eik x
            J[jidx, 3*F[i_f, 2]    ] = -(vij[i_f]@( idXec_x[i_f] + eikXdec_x[i_f])  - vik[i_f]@(eijXdec_x[i_f]))
            # d eik y
            J[jidx, 3*F[i_f, 2] + 1] = -(vij[i_f]@( idXec_y[i_f] + eikXdec_y[i_f])  - vik[i_f]@(eijXdec_y[i_f]))
            # d eik z
            J[jidx, 3*F[i_f, 2] + 2] = -(vij[i_f]@( idXec_z[i_f] + eikXdec_z[i_f])  - vik[i_f]@(eijXdec_z[i_f]))

            # r 
            r[jidx] = A[i_f] - vij[i_f]@eikXec[i_f] - eij[i_f]@vikXec[i_f]
        
        # d A
        J[self.nF: 2*self.nF, 3*self.nV : 3*self.nV + self.nF] = 1

        #self.nc = np.linalg.norm(ec, axis=1)
        
        # Update J
        self.J = J
        # Update r
        self.r = r
    






        
       