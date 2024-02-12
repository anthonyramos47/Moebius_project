# Planarity constraint implementation
import numpy as np
from hanan.optimization.constraint import Constraint
from hanan.geometry.utils import vec_dot

class LineCong(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.name = "LineCong" # Name of the constraint
        self.ei_dim = None # Number of edges per face
        self.ei_norms = None # Norms of the edges
        self.num_edge_const = None # Number of edge constraints
        self.cij_norms = [] # List to store norms of cicj
        self.ci_idx = None # Indices of ci
        self.cj_idx = None # Indices of cj
        self.i_idx = None # Indices of i
        self.j_idx = None # Indices of j
        self.inner_vertices = None # List of inner vertices
        self.normals = None # List of normals
        

    def initialize_constraint(self, X, var_indices, ei_dim, dual_faces, inner_vertices, normals) -> None:
        # Input
        # X : Variables
        # var_indices: Dictionary with the indices of the variables
        # ei_dim: Number e_i lines per vertex
        # dual_faces: List of dual faces
        # inner_vertices: List of inner vertices
        # normals: List of normals

        self.normals = normals

        # Vector dimension
        self.ei_dim = ei_dim

        # Set inner vertices
        self.inner_vertices = inner_vertices

        # Get number of edges per face
        edge_num = 0
        for f in inner_vertices:
            edge_num += len(dual_faces[f])

        self.num_edge_const = edge_num


        c, e= self.uncurry_X(X, var_indices, "sph_c", "e")

        e = e.reshape(-1, 3)
        c = c.reshape(-1, 3)

        self.ei_norms = np.linalg.norm(e, axis=1)


        # Init ci, cj indices
        self.ci_idx = []
        self.cj_idx = []

        # Init i, j indices
        self.i_idx = []
        self.j_idx = []

        # Loop over faces
        for idx_v in range(len(inner_vertices)):
            
            # Face index in dual
            v = inner_vertices[idx_v]

            # Get indices of neighbor faces to idx_v; faces to vertex v 
            face = dual_faces[v]
            self.ci_idx.append(face)

            faceroll = np.roll(face, -1, axis=0)
            self.cj_idx.append(faceroll)

            self.i_idx.append(3 * np.repeat(face, 3) + np.tile(range(3), len(face)))
            self.j_idx.append(3 * np.repeat(faceroll, 3) + np.tile(range(3), len(faceroll)))
            
            # Get vertices
            ci = c[face]
            cj = c[faceroll]
            
            self.cij_norms.append(np.linalg.norm(cj-ci, axis=1)[:, None]) 

            self.add_constraint("Orth_"+str(idx_v), len(face))
   
        self.add_constraint("Length", len(e))
        self.add_constraint("Surf_Orth", len(e))
        

            

    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
        """
        
        # Get inner vertices
        inner_vertices = self.inner_vertices

        # Get variables
        c, e = self.uncurry_X(X, var_idx, "sph_c", "e")

        e = e.reshape(-1, 3)
        c = c.reshape(-1, 3)

        cij_norms = self.cij_norms

        en = e/self.ei_norms[:, None]

        e_idx = var_idx["e"]        

        # Compute Jacobian for || e_f/||e_f|| . (cj - ci)/|| cj - ci||  ||^2 ;
        for idx_f in range(len(inner_vertices)):

            # Get vertex index ef 
            f = inner_vertices[idx_f]

            # Get face
            face     = self.ci_idx[idx_f]
            faceroll = self.cj_idx[idx_f]

            i_idx = self.i_idx[idx_f]
            j_idx = self.j_idx[idx_f]
        

            ci = c[face]
            cj = c[faceroll]

            # Define Jacobian
            cicjnor = (cj - ci)/np.array(cij_norms[idx_f])

            # Indices for I and J derivative
            idx_c_i = var_idx["sph_c"][i_idx]
            idx_c_j = var_idx["sph_c"][j_idx]

            # row indices
            row_indices = self.const_idx["Orth_"+str(idx_f)].repeat(3)
           
            # d ci || e_f (cj - ci)/|| cj - ci|||e_f|  ||^2 =>  - ef/|| cj - ci|||e_f| 
            self.add_derivatives(row_indices, idx_c_i, -(en[f] / np.array(cij_norms[idx_f])).flatten())
            
            #d cj => ef
            self.add_derivatives(row_indices, idx_c_j, (en[f] / np.array(cij_norms[idx_f])).flatten())

            # Get norms of ei                    
            e_norm = self.ei_norms[f]

            # d ef = > (cicjnor/e_norm)
            col_indices = e_idx[np.tile(np.arange(3*f, 3*f + 3), len(face))]

            self.add_derivatives(row_indices, col_indices, (cicjnor/e_norm).flatten() )

            # Define residual
            self.set_r(self.const_idx["Orth_"+str(idx_f)], vec_dot(en[f], cicjnor) )

        #print("Orth Energy:", np.sum(np.linalg.norm(self.r[self.const_idx["Orth_"+str(idx_f)]],axis=1)**2) )

        # Length constraint || e.e - le^2 ||^2
            
        # d e => 2e    
        # self.add_derivatives(self.const_idx["Length"].repeat(3), e_idx, 2*e.flatten())
        # # d le => -2le
        # self.add_derivatives(self.const_idx["Length"], self.var_idx["le"], -2*le)
        # self.set_r(self.const_idx["Length"], np.linalg.norm(e, axis=1)**2 - 2**2 - le**2)

        # print("Length Energy:", np.linalg.norm(self.r[self.const_idx["Length"]])**2 )

        ow = 0.7
        # Orthogonality constraint || (e.n)**2/||e||^2 - 1 ||^2
        e2 = self.ei_norms**2
        # d e => n/||e||
        self.add_derivatives(self.const_idx["Surf_Orth"].repeat(3), e_idx, ow*(2*((vec_dot(self.normals, e)/e2)[:,None]* self.normals) ).flatten())
        self.set_r(self.const_idx["Surf_Orth"], ow*((1/e2)*vec_dot(e, self.normals)**2 - 1) )

        # print("Orth Energy:", np.linalg.norm(self.r[self.const_idx["Orth"]])**2 )
        # print("Total LC Energy:", self.r@self.r)
        # print("\n")
        self.ei_norms = np.linalg.norm(e, axis=1)



        

       

        
