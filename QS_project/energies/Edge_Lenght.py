# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np
from geometry.utils import indices_flatten_dim


class Edge_L(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy the edge length of the mesh.
        E_{EL} = \sum_{eij in E} ( ||vi - vj||^2 - l0 - mu^2 )^2 
        """
        super().__init__()
        self.name = "Edge" # Name of the constraint
        self.l0 = None # Initial length of the edges
        self.weights = None # Weights per edge
        self.aux_idx = None # Auxiliar variable index
        self.vi_idx = None # Vertex i index
        self.vj_idx = None # Vertex j index
        self.dummy_idx = None # Dummy variable index

      
    def initialize_constraint(self, X, var_idx, var_name, edges, dim) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            var_name    : Name of the variable
            edges       : Edges indices
        """

        # Get the nodes
        v = self.uncurry_X(X, var_idx, var_name)
        v = v.reshape(-1, dim)

    
        # Get the indices of the vertices
        self.vi_idx = var_idx[var_name][indices_flatten_dim(edges[:, 0], dim)]
        self.vj_idx = var_idx[var_name][indices_flatten_dim(edges[:, 1], dim)]

        # Get the dummy variable
        self.dummy_idx = var_idx["mu"]

        # Compute initial length of the edges
        self.l0 = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)**2

        # Define initial weights
        self.weights = np.ones(len(edges))

        # Add constraints
        # E1 = ((vj - vi)^2 - l0^2 - mu^2)^2
        self.add_constraint("E1", len(edges))

        #X[var_idx[var_name]] += 0.2*np.random.rand()

        
    
    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """ 

        # Get the nodes
        vi = X[self.vi_idx].reshape(-1, 3)
        vj = X[self.vj_idx].reshape(-1, 3)

        # Get the dummy variable
        mu = X[self.dummy_idx]

        # Compute the length of the edges
        l = np.linalg.norm(vj - vi, axis=1)**2
        vij = (vj - vi) 

        # E1 = ((vj - vi)^2 - l0^2 - mu^2)^2

        # dvi E1 = - 2(vj -vi)
        dv_E1 = (2 * vij).flatten()
        self.add_derivatives(self.const_idx["E1"].repeat(3), 
                             self.vi_idx, 
                            - dv_E1)

        #  dvj E1 =  2(vj -vi) 
        self.add_derivatives(self.const_idx["E1"].repeat(3), 
                             self.vj_idx, 
                             dv_E1)
        
        # dmu_E1 = - 2 * mu * self.weights
        dmu_E1 = - 2 * mu
        self.add_derivatives(self.const_idx["E1"], 
                             self.dummy_idx, 
                             dmu_E1)

        
        # residual E1
        self.set_r(self.const_idx["E1"], (l - self.l0**2 - mu**2))

        #self.weights = 1/( l + 0.001)

        







        
        

        
                

        


