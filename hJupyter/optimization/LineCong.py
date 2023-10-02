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
        

    def initialize_constraint(self, ct, cf, ei_dim, X) -> None:
        # Input
        # ct: list of vertices of central mesh
        # cf: list of faces of central mesh
        # X: variables

        # Initialize constraint \sum_{f \in F} \sum_{cj,ci \in E(f)} || e_f (cj - ci)  ||^2

        # Vector dimension
        self.ei_dim = ei_dim

        # Get number of edges per face
        count = 0
        for f in range(len(cf)):
            count += len(cf[f])

        self.num_edge_const = count

        # Get directions
        ei = X.reshape(self.ei_dim, 3)

        # Init Jacobian and residual vector
        J = np.zeros((count + 3*len(ei), len(X)), dtype=np.float64)
        r = np.zeros(count + 3*len(ei), dtype=np.float64)

        # Compute Jacobian
        # Row index
        i = 0
        # Loop over faces
        for f in range(len(cf)):
            # Get face
            face = cf[f]

            # Get vertices
            v0 = ct[face]
            v1 = np.roll(ct[face], -1, axis=0)

            # Define direction
            cicj = (v1 - v0) / np.linalg.norm(v1 - v0, axis=1)[:, None]

            # Define Jacobian
            J[i:i + len(face), 3*f:3*f + 3] = cicj

            # Store cicj because it is a constant value
            self.cij.extend(cicj)

            # Define residual
            r[i:i + len(face)] = np.dot(cicj, ei[f])

            # Update row index
            i += len(face)


        # Define Jacobian for the auxiliary variable
        for f in range(len(cf)):
            J[count+f, f*3:f*3+3 ] = ei[f]

            r[count+f] = ei[f]@ei[f] - 1

        self.J = J
        self.r = r

            

    def compute(self, ct, cf, X) -> None:
        # Get directions
        ei = X.reshape(self.ei_dim, 3)

        # Compute Jacobian
        i = 0
        for f in range(len(cf)):

            face = cf[f]
            for id in range(len(face)):
               
                # Define residual
                self.r[i] = self.cij[i]@ei[f]

                i += 1

        # Define Jacobian for the auxiliary variable
        for f in range(len(cf)):
            self.J[self.num_edge_const+f, f*3:f*3+3 ] = ei[f]

            self.r[self.num_edge_const+f] = ei[f]@ei[f] - 1