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

        
        self.var = len(X)

        e = self.uncurry_X(X, "e")
        
        self.const = len(e)

        e = e.reshape(-1, 3)

        self.const_idx = {"Fair": np.arange(0, self.const)
                          }

            

    def compute(self, X) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
        """

        
        # Get variables
        e = self.uncurry_X(X, "e")

        e = e.reshape(-1, 3)

        e_idx = self.var_idx["e"]

        #indices = 3 * np.repeat(e_idx, 3) + np.tile(range(3), len(e_idx))
        
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


        
