# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np


class Laplacian(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{supp} = \sum_{f\in Mesh Dual} \sum_{vi in f} || (vi - cf)^2 - rf^2 ||^2
        """
        super().__init__()
        self.name = "Laplacian" # Name of the constraint
        self.neigh_idx = None # Neighbors indices

        # Cont for weight decrease
        self.cont = 0


    def initialize_constraint(self, X, var_idx, adj_v, var_name, dim) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            inn_v       : list of inner vertices indices
            adj_v       : list of adjacent vertices indices

        """

        self.name = "Laplacian_"+var_name

        self.neigh_idx = []
        self.v_k = []
        self.row_i = []
        for i in range(len(adj_v)):
            n = len(adj_v[i])
            indices_vertices = var_idx[var_name][3 * np.repeat(adj_v[i], dim) + np.tile(range(dim), n)]
            self.neigh_idx.append(indices_vertices)
            self.v_k.append(var_idx[var_name][3 * i: 3 * i + dim])
            self.row_i.append(np.arange(3*i, 3*i+dim))

        self.add_constraint("Lap", len(adj_v)*3)

        
        
    def compute(self, X, var_idx):
       """ Energy computation of Laplacian
       """
        
       for i in range(len(self.neigh_idx)):
            X_k = X[self.v_k[i]]
            X_j = X[self.neigh_idx[i]].reshape(-1, 3)

            N = len(self.neigh_idx[i])

            # d_vj E = 1 / |N(v_i)| 
            d_vj = np.ones(len(self.neigh_idx[i])) / N

            # d_vk E = - 1
            d_vk = - np.ones(3)

            self.add_derivatives(self.row_i[i], 
                                 self.v_k[i], 
                                 d_vk)
            self.add_derivatives(self.row_i[i].repeat(len(X_j)), 
                                 self.neigh_idx[i], 
                                 d_vj)

            res = np.sum(X_j, axis=0) / N - X_k

            self.set_r(self.row_i[i], res)

            





            
            