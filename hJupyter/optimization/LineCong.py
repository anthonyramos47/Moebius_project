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

        self.ei_dim = ei_dim

        # Get number of edges per face
        count = 0
        for f in range(len(cf)):
            count += len(cf[f])

        self.num_edge_const = count

        # Get directions
        ei = X.reshape(self.ei_dim, 3)

        # Comppute constant Jacobian
        J = np.zeros((count + 3*len(ei), len(X)), dtype=np.float64)

        r = np.zeros(count + 3*len(ei), dtype=np.float64)

        print(f"ei : {len(ei)}")
        print(f"cf : {len(cf)}")

        # Compute Jacobian
        i = 0
        for f in range(len(cf)):

            face = cf[f]

            for id in range(len(face)):
                # Get vertices
                v0 = ct[face[id]]
                v1 = ct[face[(id+1)%len(face)]]

                # Define direction
                cicj = (v1 - v0)/ np.linalg.norm(v1 - v0)

                # Define Jacobian
                J[i, 3*f: 3*f + 3] = cicj
                
                self.cij.append(cicj)
                # Define residual
                r[i] = cicj@ei[f]

                i += 1

        # Define Jacobian for the auxiliary variable
        for f in range(len(cf)):
            J[count+f, f*3:f*3+3 ] = ei[f]

            r[count+f] = ei[f]@ei[f] - 1

        self.J = J
        self.r = r

            

    def compute(self, ct, cf, X) -> None:
        # Get center mesh 
        

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