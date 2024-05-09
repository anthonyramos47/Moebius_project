# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np


class Lap_Fairness(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{supp} = \sum_{f\in Mesh Dual} \sum_{vi in f} || lk - 1/Nlk sum(lj)||^2
        """
        super().__init__()
        self.name = "Lap_Fairness" # Name of the constraint
        
        self.vk = None # vertices
        self.vk_n = None # Neighbors of valence n!=4 vertices

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

        self.name = "Lap_Fairness_"+var_name


        num_rows_L = 0 # Rows for Laplacian
        

        # Valence = 4 vertices  Laplacian
        vk = []
        vk_n = []

        

        for i, _ in enumerate(adj_v):


            
            num_rows_L += 1

            vk_n.append(adj_v[i])
            vk.extend([i])
                
        
    

        if dim > 1:

            self.vk = var_idx[var_name][3 * np.repeat(vk, dim) + np.tile(range(dim), len(vk))]

            for i in range(len(vk_n)):
                vk_n[i] = var_idx[var_name][3 * np.repeat(vk_n[i], dim) + np.tile(range(dim), len(vk_n[i]))]

            self.vk_n = vk_n
        
        self.add_constraint("Lap_Fair", 3*num_rows_L)

        
        
    def compute(self, X, var_idx):
        
        # Laplacian
        vk = X[self.vk].reshape(-1, 3)
        vk_n = [X[i] for i in self.vk_n]

        vk_indices = self.vk.reshape(-1, 3)

        for i in range(len(vk_n)):
            
            # Get the neighbors
            vn = vk_n[i].reshape(-1, 3)

            # Get the number of neighbors
            Nk = len(vn)

            # Indices
            vk_n_idx = self.vk_n[i]

            # L = (vk - 1/n \sum_{vj in N(vk)} vj)^2
            # d_vk = 1 
            self.add_derivatives(self.const_idx["Lap_Fair"][3*i: 3*(i+1)], vk_indices[i], np.ones(3))

            # d_vj = -1/n
            self.add_derivatives(self.const_idx["Lap_Fair"][3*i: 3*(i+ 1)].repeat(Nk), vk_n_idx, -np.ones(len(vk_n_idx))/Nk)

            # Compute residuals
            res = (vk[i] - np.mean(vn, axis=0)).flatten()

            self.set_r(self.const_idx["Lap_Fair"][3*i: 3*(i+1)], res)


        self.cont+=1

        if self.cont %8 == 0:
            self.w *= 0.8
        #if self.cont %25 == 0:
        #    self.w = 0
            #print(f"Weight decrease to {self.w}")
        





            
            