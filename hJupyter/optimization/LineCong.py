# Planarity constraint implementation

import numpy as np
import geometry as geo
from optimization.constraint import Constraint

class LineCong(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.ei_dim = None # Number of edges per face
        self.num_edge_const = None # Number of edge constraints
        self.bij = [] # List to store vertices distance
        self.nt = [] # List to store normals
        self.norms = [] # List to store norms of cicj
        self.initLi = None # Initial column for li variables
        

    def initialize_constraint(self, X, ei_dim, bt, nbt, num_faces, dual_faces, inner_vertices, w=1) -> None:
        # Input
        # bt: list of barycenters
        # nbt: list of Circum circles axis
        # cf: list of faces of central mesh
        # X: variables [ei | A | delta | li]

        # Initialize constraint \sum_{f \in F} \sum_{cj,ci \in E(f)} || e_f (cj - ci)  ||^2 ; ci = bi + li * nbi

        # Weight
        self.w = w

        # Vector dimension
        self.ei_dim = ei_dim

        # Set normals
        self.nt = nbt

        # Set initial column for li variables
        self.initLi = len(X) - len(X[-num_faces:])

        # Get number of edges per face
        edge_num = 0
        for f in inner_vertices:
            edge_num += len(dual_faces[f])

        self.num_edge_const = edge_num

        li = X[num_faces:]

        # Compute Jacobian
        # Row index
        i = 0
        # Loop over faces
        for idx_f in range(len(inner_vertices)):
            
            # Face 
            f = inner_vertices[idx_f]

            # Get face
            face = dual_faces[f]
            faceroll = np.roll(face, -1, axis=0)

            # Get vertices
            bi = bt[face]
            bj = bt[faceroll]

            # Get normals
            ni = nbt[face]
            nj = nbt[faceroll]

            # Get li 
            lbi = li[face]
            lbj = li[faceroll]

            # Store vertices distance
            self.bij.append(bj - bi)

            # Define direction
            cicj = (bj - bi) + (lbj[:, None]*nj - lbi[:, None]*ni)
            
            self.norms.append(np.linalg.norm(cicj, axis=1)[:, None]) 

            # Update row index
            i += len(face)

            

    def compute(self, X, inner_vertices, cf) -> None:
        
        # Get directions
        ei = X[:3*self.ei_dim].reshape(self.ei_dim, 3)

        print(self.initLi)
        # Get li 
        li = X[self.initLi:]

        en = self.num_edge_const

        # Init Jacobian and residual vector
        J = np.zeros((en + len(ei), len(X)), dtype=np.float64)
        r = np.zeros( en + len(ei), dtype=np.float64)

        # Compute Jacobian
        i = 0
        for idx_f in range(len(inner_vertices)):

            f = inner_vertices[idx_f]
            # Get face
            face = cf[f]
            faceroll = np.roll(face, -1, axis=0) 

            # Get normals
            ni = self.nt[face]
            nj = self.nt[faceroll]

            # Get li
            lbi = li[face]
            lbj = li[faceroll]

            # Define direction
            cicj = (self.bij[idx_f] + (lbj[:, None]*nj - lbi[:, None]*ni) )

            norms = self.norms[idx_f]

            self.norms[idx_f] = np.linalg.norm(cicj, axis=1)[:, None] 

            # Define Jacobian
            cicj /= np.array(norms)

            # d ei
            J[i:i + len(face), 3*f: 3*f + 3 ] = cicj

                 
            # Indices for I and J derivative
            ii = self.initLi + np.array(face)
            jj = self.initLi + np.array(faceroll)
            # d li
            J[range(i,i + len(face)), ii] = -np.sum( ei[f]*ni, axis=1)/self.norms[idx_f].flatten()
            #d lj
            J[range(i,i + len(face)), jj] = np.sum( ei[f]*nj, axis=1)/self.norms[idx_f].flatten()

            # Define residual
            r[i:i + len(face)] = np.sum(cicj*ei[f], axis=1)

            assert np.allclose(np.dot(cicj, ei[f]), np.sum(cicj*ei[f], axis=1))

            # Update row index
            i += len(face)

        
        # Update Jacobian
        for f in range(len(cf)):
            J[self.num_edge_const+f, f*3:f*3+3 ] = ei[f]
        
        # Update residual
        r[self.num_edge_const:] = np.sum ( ei*ei,  axis=1) - 1

        self.J = J
        self.r = r
