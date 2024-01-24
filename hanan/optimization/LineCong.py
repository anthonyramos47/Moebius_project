# Planarity constraint implementation
import numpy as np
from hanan.optimization.constraint import Constraint
from hanan.geometry.utils import vec_dot

class LineCong(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.ei_dim = None # Number of edges per face
        self.ei_norms = None # Norms of the edges
        self.num_edge_const = None # Number of edge constraints
        self.cij_norms = [] # List to store norms of cicj
        self.inner_vertices = None # List of inner vertices
        

    def initialize_constraint(self, X, var_indices, ei_dim, dual_faces, inner_vertices) -> None:
        # Input
        # X : Variables
        # var_indices: Dictionary with the indices of the variables
        # ei_dim: Number e_i lines per vertex
        # dual_faces: List of dual faces
        # inner_vertices: List of inner vertices


        self.var_idx = var_indices

        # Vector dimension
        self.ei_dim = ei_dim

        # Set inner vertices
        self.inner_vertices = inner_vertices

        # Get number of edges per face
        edge_num = 0
        for f in inner_vertices:
            edge_num += len(dual_faces[f])

        self.num_edge_const = edge_num

        
        self.const = self.num_edge_const
        self.var = len(X)

        c, e= self.uncurry_X(X, "sph_c", "e")

        e = e.reshape(-1, 3)
        c = c.reshape(-1, 3)

        self.ei_norms = np.linalg.norm(e, axis=1)

        self.const_idx = {"E":[]
                          }

        self.dual_faces = dual_faces
        # Compute Jacobian
        # Row index
        i = 0
        # Loop over faces
        for idx_v in range(len(inner_vertices)):
            
            # Face index in dual
            v = inner_vertices[idx_v]

            # Get indices of neighbor faces to idx_v; faces to vertex v 
            face = dual_faces[v]
            faceroll = np.roll(face, -1, axis=0)
            
            # Get vertices
            ci = c[face]
            cj = c[faceroll]
            
            self.cij_norms.append(np.linalg.norm(cj-ci, axis=1)[:, None]) 

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
        c, e= self.uncurry_X(X, "sph_c", "e")

        e = e.reshape(-1, 3)
        c = c.reshape(-1, 3)

        cij_norms = self.cij_norms

        en = e/self.ei_norms[:, None]

        e_idx = self.var_idx["e"]        

        # Compute Jacobian for || e_f/||e_f|| . (cj - ci)/|| cj - ci||  ||^2 ;
        for idx_f in range(len(inner_vertices)):

            f = inner_vertices[idx_f]

            # Get face
            face     = self.dual_faces[f]
            faceroll = np.roll(face, -1, axis=0) 

            i_idx = 3 * np.repeat(face, 3) + np.tile(range(3), len(face))
            j_idx = 3 * np.repeat(faceroll, 3) + np.tile(range(3), len(faceroll))
        

            ci = c[face]
            cj = c[faceroll]

            # Define Jacobian
            cicjnor = (cj - ci)/np.array(cij_norms[idx_f])

   
            # Indices for I and J derivative
            idx_c_i = self.var_idx["sph_c"][i_idx]
            idx_c_j = self.var_idx["sph_c"][j_idx]

                       
            # d dci || e_f (cj - ci)/|| cj - ci|||e_f|  ||^2 =>  - ef.ni
            #J[range(i,i + len(face)), ii] = -np.sum( ei[f]*ni, axis=1)/self.norms[idx_f].flatten()
            self.add_derivatives(self.const_idx["E"][idx_f].repeat(3), idx_c_i, -(en[f] / np.array(cij_norms[idx_f])).flatten())
            
            #d dcj
            #J[range(i,i + len(face)), jj] = np.sum( ei[f]*nj, axis=1)/self.norms[idx_f].flatten()
            self.add_derivatives(self.const_idx["E"][idx_f].repeat(3), idx_c_j, (en[f] / np.array(cij_norms[idx_f])).flatten())

            # Get norms of ei                    
            e_norm = self.ei_norms[f]

            # d ei
            row_indices = self.const_idx["E"][idx_f].repeat(3)
            col_indices = e_idx[np.tile(np.arange(3*f, 3*f + 3), len(face))]

            self.add_derivatives(row_indices, col_indices, (cicjnor/e_norm).flatten() )


            # Define residual
            self.set_r(self.const_idx["E"][idx_f], vec_dot(en[f], cicjnor) )

        self.ei_norms = np.linalg.norm(e, axis=1)
