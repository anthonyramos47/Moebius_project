# Planarity constraint implementation

import numpy as np
import geometry as geo
from optimization.constraint import Constraint

class LineCong(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.ei_dim = None
        self.num_edge_const = None
        self.cij = []
        

    def initialize_constraint(self, X, ei_dim, ct, dual_faces, inner_vertices, w=1) -> None:
        # Input
        # ct: list of vertices of central mesh
        # cf: list of faces of central mesh
        # X: variables

        # Initialize constraint \sum_{f \in F} \sum_{cj,ci \in E(f)} || e_f (cj - ci)  ||^2

        # Weight
        self.w = w

        # Vector dimension
        self.ei_dim = ei_dim

        # Get number of edges per face
        edge_num = 0
        for f in inner_vertices:
            edge_num += len(dual_faces[f])

        self.num_edge_const = edge_num

        # Get directions
        ei = X[:3*self.ei_dim].reshape(self.ei_dim, 3)

        # Init Jacobian and residual vector
        J = np.zeros((edge_num + len(ei), len(X)), dtype=np.float64)
        r = np.zeros( edge_num + len(ei), dtype=np.float64)

        # Compute Jacobian
        # Row index
        i = 0
        # Loop over faces
        for idx_f in range(len(inner_vertices)):
            
            # Face 
            f = inner_vertices[idx_f]

            # Get face
            face = dual_faces[f]

            # Get vertices
            v0 = ct[face]
            v1 = np.roll(ct[face], -1, axis=0)

            # Define direction
            cicj = (v1 - v0) / np.linalg.norm(v1 - v0, axis=1)[:, None]

            # Define Jacobian
            J[i:i + len(face), 3*f:3*f + 3] = cicj

            # Store cicj because it is a constant value
            self.cij.append(cicj)

            # Define residual
            r[i:i + len(face)] = np.dot(self.cij[idx_f], ei[f])

            # Update row index
            i += len(face)


        # Define Jacobian for the auxiliary variable
        for f in range(len(dual_faces)):
            J[edge_num + f, f*3:f*3+3 ] = ei[f]

        
        r[edge_num:] =np.sum ( ei*ei,  axis=1) - 1

        self.J = J
        self.r = r

            

    def compute(self, X, inner_vertices, cf) -> None:
        
        # Get directions
        ei = X[:3*self.ei_dim].reshape(self.ei_dim, 3)

        # Compute Jacobian
        i = 0
        for idx_f in range(len(inner_vertices)):

            f = inner_vertices[idx_f]
            # Get face
            face = cf[f]

            # Define residual
            self.r[i:i + len(face)] = np.dot(self.cij[idx_f], ei[f])
            
            # Update row index
            i += len(face)

        
        # Update Jacobian
        for f in range(len(cf)):
            self.J[self.num_edge_const+f, f*3:f*3+3 ] = ei[f]

        # Update residual
        self.r[self.num_edge_const:] = np.sum ( ei*ei,  axis=1) - 1