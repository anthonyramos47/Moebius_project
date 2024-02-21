# Planarity constraint implementation
import numpy as np
from hanan.optimization.constraint import Constraint
from hanan.geometry.utils import vec_dot

class LineCong_Fair(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.name = "LineCong_Fair" # Name of the constraint
        self.inner_vertices = None # List of inner vertices
        self.vertices_neighbors = None # List of vertices neighbors
        self.nt = None # Normal of the tangent plane
        self.e_norms = None # Norms of the edges

        
    def initialize_constraint(self, X, var_indices, vertex_neigh, inner_vertices, nt) -> None:
        # Input
        # X : Variables
        # var_indices: Dictionary with the indices of the variables
        # inner_vertices: List of inner vertices

        # Initialize constraint \sum_{f \in F} \sum_{cj,ci \in E(f)} || e_f (cj - ci)/|| cj - ci||  ||^2 ; ci = bi + df * nbi
        # 
        self.vertices_neighbors = vertex_neigh
   
        # Set inner vertices
        self.inner_vertices = inner_vertices

        self.nt = nt
        
        e = self.uncurry_X(X, var_indices, "e")
        
        # length of e flattened
        e_flat = len(e)

        e = e.reshape(-1, 3)

        self.e_norms = np.linalg.norm(e, axis=1)

        self.add_constraint("Fair", e_flat)

                

    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
        """

        
        # Get variables
        e = self.uncurry_X(X,var_idx, "e")
        e = e.reshape(-1, 3)

        # Get indices
        e_idx = var_idx["e"]

        
        # d ei
        self.add_derivatives(self.const_idx["Fair"], e_idx, np.ones(3*len(e))) 
        
        for i in range(len(e)):

            # Get neighbors
            neigh = self.vertices_neighbors[i] 

            # Get number of neighbors
            n_neigh = len(neigh)

            # Get edges
            ej = e[neigh]

            indices = e_idx[3 * np.repeat(neigh, 3) + np.tile(range(3), len(neigh))]

            # Assert three values are the same
            assert len(np.tile(self.const_idx["Fair"][3*i: 3*i+3], n_neigh)) == len(indices), "Row != Cols"

            assert len(indices) == len(-ej.flatten()/n_neigh), "Cols != Vals"

            # d ej 
            self.add_derivatives(np.tile(self.const_idx["Fair"][3*i: 3*i+3], n_neigh), indices, -np.ones_like(ej).flatten()/n_neigh)
        
            # Assert columnsa and values same size
            assert len(self.const_idx["Fair"][3*i: 3*i+3]) == len(e[i] - np.sum(ej, axis=0)/n_neigh), "r: Cols != Values"
           
            self.set_r(self.const_idx["Fair"][3*i: 3*i+3], e[i] - np.sum(ej, axis=0)/n_neigh) 
        
        #print("E fair:", self.r[self.const_idx["Fair"]]@self.r[self.const_idx["Fair"]])
    
        # print("nt norm", np.sum(np.linalg.norm(self.nt, axis=1))/len(self.nt))
        # # # Orthogonality constraint ||(e.nt)/|e| - 1||^2
        # # # d e => nt
        # # self.add_derivatives(self.const_idx["Orth"].repeat(3), e_idx, (self.nt/self.e_norms[:,None]).flatten())
        
        # # self.set_r(self.const_idx["Orth"], vec_dot(e, self.nt)/self.e_norms - 1)

        # print("E orht:", self.r[self.const_idx["Orth"]]@self.r[self.const_idx["Orth"]]) 

        # self.e_norms = np.linalg.norm(e, axis=1)

     





        
