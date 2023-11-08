# Planarity constraint implementation

import numpy as np
import geometry as geo
from optimization.constraint import Constraint

class Planarity(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.normals = None

    def initialize_constraint(self, mesh, X) -> None:
        # Init normals
        m = mesh
        v = X[:3*mesh.V].reshape(m.V, 3)

        # Get list of indices 
        iv0, iv1, iv2, iv3 = m.faces()[:,0], m.faces()[:,1], m.faces()[:,2], m.faces()[:,3]

        # v0, v1, v2, v3 = v[iv0], v[iv1], v[iv2], v[iv3]
        diag1 = v[iv2] - v[iv0]
        diag2 = v[iv3] - v[iv1]

        # # Compute normals
        X[3*mesh.V:] = np.cross(diag1, diag2).flatten()

        self.normals = np.cross(diag1, diag2)
        

    def compute(self, mesh, X) -> None:

        # Get mesh
        m = mesh
        
        v = X[:3*mesh.V].reshape(m.V, 3)

        normals = X[3*mesh.V:].reshape(m.F, 3)
        # Planarity
        # Constraint  n . (vi - vj) per each edge (vi, vj) in F, so we have 4 constraints per each face
        # Our variarles are the vertices of the mesh, so we have 3n variables plus the auxiliary variable
        # n the normals of the faces which result in 3*n + 3*F variables
        J = np.zeros((m.F*5, m.V*3+ 3*m.F), dtype=np.float64)

        r = np.zeros(m.F*5, dtype=np.float64)
        # Compute sums of vertices diferences
        #face_edge_sum = (v1-v0) + (v2-v1) + (v3-v2) + (v0-v3)
        for f in range(m.F):

            iv0, iv1, iv2, iv3 = m.faces()[f]    

            v0, v1, v2, v3 = v[iv0], v[iv1], v[iv2], v[iv3]

            # Edge constraint nf(v1 - v0) = 0
            J[        f, 3*iv0: 3*iv0 + 3] = - normals[f]
            J[        f, 3*iv1: 3*iv1 + 3] =   normals[f]

            # Edge constraint nf(v2 - v1) = 0
            J[  m.F + f, 3*iv1: 3*iv1 + 3] = - normals[f]
            J[  m.F + f, 3*iv2: 3*iv2 + 3] =   normals[f]

            # Edge constraint nf(v3 - v2) = 0
            J[2*m.F + f, 3*iv2: 3*iv2 + 3] = - normals[f]
            J[2*m.F + f, 3*iv3: 3*iv3 + 3] =   normals[f]

            # Edge constraint nf(v0 - v3) = 0
            J[3*m.F + f, 3*iv3: 3*iv3 + 3] = - normals[f]
            J[3*m.F + f, 3*iv0: 3*iv0 + 3] =   normals[f]

            # Edge constraint nf(vi - vj) = 0 normal derivative
            J[        f, 3*m.V + 3*f: 3*m.V + 3*f+3] = (v1 - v0)
            J[  m.F + f, 3*m.V + 3*f: 3*m.V + 3*f+3] = (v2 - v1)
            J[2*m.F + f, 3*m.V + 3*f: 3*m.V + 3*f+3] = (v3 - v2)
            J[3*m.F + f, 3*m.V + 3*f: 3*m.V + 3*f+3] = (v0 - v3)

        
            r[        f] = np.dot(normals[f], v1 - v0)
            r[  m.F + f] = np.dot(normals[f], v2 - v1)
            r[2*m.F + f] = np.dot(normals[f], v3 - v2)
            r[3*m.F + f] = np.dot(normals[f], v0 - v3)

            # Unit normal constraint
            J[4*m.F + f, 3*m.V + 3*f: 3*m.V + 3*f+3] = normals[f]
            r[4*m.F + f] = normals[f]@normals[f] - 1
        
        self.J = J
        self.r = r