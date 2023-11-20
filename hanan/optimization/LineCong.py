# Planarity constraint implementation
import numpy as np
from hanan.optimization.constraint import Constraint

class LineCong(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.ei_dim = None # Number of edges per face
        self.num_edge_const = None # Number of edge constraints
        self.bij = [] # List to store vertices distance
        self.nt = [] # List to store normals
        self.norms = [] # List to store norms of cicj
        self.inner_vertices = None # List of inner vertices
        self.dual_faces = None # List of dual faces
        

    def initialize_constraint(self, X, var_indices, ei_dim, bt, nbt, num_faces, dual_faces, inner_vertices) -> None:
        # Input
        # X : Variables
        # var_indices: Dictionary with the indices of the variables
        # ei_dim: Number of edges per face
        # bt: Circumcenters of the faces
        # nbt: Normals at the circumcenters
        # num_faces: Number of faces
        # dual_faces: List of dual faces
        # inner_vertices: List of inner vertices


        # Initialize constraint \sum_{f \in F} \sum_{cj,ci \in E(f)} || e_f (cj - ci)/|| cj - ci||  ||^2 ; ci = bi + df * nbi
        # 

        self.var_idx = var_indices

        # Vector dimension
        self.ei_dim = ei_dim

        # Set normals
        self.nt = nbt

        # Set inner vertices
        self.inner_vertices = inner_vertices

        # Get number of edges per face
        edge_num = 0
        for f in inner_vertices:
            edge_num += len(dual_faces[f])

        self.num_edge_const = edge_num

        
        self.const = self.num_edge_const
        self.var = len(X)

        e, df = self.uncurry_X(X, "e", "df")

        e = e.reshape(-1, 3)

        self.const_idx = {"E":[]
                          }

        self.dual_faces = dual_faces
        # Compute Jacobian
        # Row index
        i = 0
        # Loop over faces
        for idx_f in range(len(inner_vertices)):
            
            # Face index in dual
            f = inner_vertices[idx_f]

            # Get dual face
            face = dual_faces[f]
            # Get rolled indices
            faceroll = np.roll(face, -1, axis=0)

            # Get vertices
            bi = bt[face]
            bj = bt[faceroll]

            # Get normals
            ni = nbt[face]
            nj = nbt[faceroll]

            # Get df 
            lbi = df[face]
            lbj = df[faceroll]

            # Store vertices distance
            self.bij.append(bj - bi)

            # Define direction
            cicj = (bj - bi) + (lbj[:, None]*nj - lbi[:, None]*ni)
            
            self.norms.append(np.linalg.norm(cicj, axis=1)[:, None]) 

            self.const_idx["E"].append(np.arange(i, i + len(face)))

            # Update row index
            i += len(face)

            

    def compute(self, X) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
        """
        
        # Get inner vertices
        inner_vertices = self.inner_vertices

        # Get variables
        ei, df = self.uncurry_X(X, "e", "df")

        ei = ei.reshape(-1, 3)

        e_idx = self.var_idx["e"]

        # Compute Jacobian for || e_f (cj - ci)/|| cj - ci||  ||^2 ; ci = bi + df * nbi
        for idx_f in range(len(inner_vertices)):

            f = inner_vertices[idx_f]

            # Get face
            face     = self.dual_faces[f]
            faceroll = np.roll(face, -1, axis=0) 

            # Get normals
            ni = self.nt[face]
            nj = self.nt[faceroll]

            # Get df
            lbi = df[face]
            lbj = df[faceroll]

            # Define direction
            cicj = (self.bij[idx_f] + (lbj[:, None]*nj - lbi[:, None]*ni) )

            norms = self.norms[idx_f]
                 
            # Define Jacobian
            cicjnor = cicj/np.array(norms)

            # d ei
            #J[i:i + len(face), 3*f: 3*f + 3 ] = cicj
            row_indices = self.const_idx["E"][idx_f].repeat(3)
            col_indices = e_idx[np.tile(np.arange(3*f, 3*f + 3), len(face))]

            # print(f" f : {f} \t 3*f : {3*f}")
            # print("row indices:",row_indices)
            # print(f"col indices: {col_indices}")
            # print("cicjnor:", cicjnor)
            # print("\n\n")
            self.add_derivatives(row_indices, col_indices, cicjnor.flatten() )

            # Indices for I and J derivative
            dfi = self.var_idx["df"][face]
            dfj = self.var_idx["df"][faceroll]
            
            # d dfi || e_f (cj - ci)/|| cj - ci||  ||^2 =>  - ef.ni
            #J[range(i,i + len(face)), ii] = -np.sum( ei[f]*ni, axis=1)/self.norms[idx_f].flatten()
            #self.add_derivatives(self.const_idx["E"][idx_f], dfi, -np.sum( ei[f]*ni, axis=1)/self.norms[idx_f].flatten())
            
            #d dfj
            #J[range(i,i + len(face)), jj] = np.sum( ei[f]*nj, axis=1)/self.norms[idx_f].flatten()
            #self.add_derivatives(self.const_idx["E"][idx_f], dfj, np.sum( ei[f]*nj, axis=1)/self.norms[idx_f].flatten())

            # Define residual
            self.set_r(self.const_idx["E"][idx_f], np.sum(cicjnor*ei[f], axis=1) )
            #self.r[self.const_idx["E"][idx_f]] = np.sum(cicj*ei[f], axis=1)


            self.norms[idx_f] = np.linalg.norm(cicj, axis=1)[:, None] 

