# Planarity constraint implementation
import numpy as np
from hanan.optimization.constraint import Constraint

class LineCong_Fair(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.inner_vertices = None # List of inner vertices
        self.vertices_neighbors = None # List of vertices neighbors

        

    def initialize_constraint(self, X, var_indices, vertex_neigh, inner_vertices) -> None:
        # Input
        # X : Variables
        # var_indices: Dictionary with the indices of the variables
        # inner_vertices: List of inner vertices

        # Initialize constraint \sum_{f \in F} \sum_{cj,ci \in E(f)} || e_f (cj - ci)/|| cj - ci||  ||^2 ; ci = bi + df * nbi
        # 

        self.var_idx = var_indices

        self.vertices_neighbors = vertex_neigh
   
        # Set inner vertices
        self.inner_vertices = inner_vertices

        self.const = 3*len(inner_vertices)
        self.var = len(X)

        e = self.uncurry_X(X, "e")

        e = e.reshape(-1, 3)

        self.const_idx = {"Fair": np.arange(0, self.const)
                          }

            

    def compute(self, X) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
        """
        
        # Get inner vertices
        inner_vertices = self.inner_vertices

        # Get variables
        e = self.uncurry_X(X, "e")

        e = e.reshape(-1, 3)

        e_idx = self.var_idx["e"]

        indices = 3 * np.repeat(e_idx[inner_vertices], 3) + np.tile(range(3), len(e_idx[inner_vertices]))
        
        # d ei
        self.add_derivatives(self.const_idx["Fair"], indices, np.ones(len(indices))) 
        
        for i in range(len(inner_vertices)):

            idx = inner_vertices[i]

            # Get neighbors
            neigh = e_idx[self.vertices_neighbors[idx]]

            # Get number of neighbors
            n_neigh = len(neigh)

            # Get edges
            ej = e[neigh]

            indices = 3 * np.repeat(neigh, 3) + np.tile(range(3), len(neigh))

            # d ej 
            self.add_derivatives(np.tile(self.const_idx["Fair"][3*i: 3*i+3], n_neigh), indices, -ej.flatten()/n_neigh)
            self.set_r(self.const_idx["Fair"][3*i: 3*i+3], e[idx] - np.sum(ej, axis=0)/n_neigh) 


        
