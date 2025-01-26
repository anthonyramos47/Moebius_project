# Call parent class
from optimization.constraint import Constraint
import numpy as np


class Fairness(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{supp} = \sum_{f\in Mesh Dual} \sum_{vi in f} || (vi - cf)^2 - rf^2 ||^2
        """
        super().__init__()
        self.name = None # Name of the constraint
        self.du_i = None # Direction  u
        self.du_j = None # Direction  u
        self.dv_i = None # Direction v
        self.dv_j = None # Direction v

        self.vk   = None # Valence n!=4 vertices
        self.vk_n = None # Neighbors of valence n!=4 vertices
        
        self.row_du = None # Row indices du
        self.row_dv = None # Row indices dv

        self.dec_fac  = 0.5 # Decrease factor
        self.dec_step = 5  # Steps to decrease weight

        # Cont for weight decrease
        self.cont = 0


      
    def initialize_constraint(self, X, var_idx, var_name, adj_v, dim) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            var_name    : name of variable
            adj_v       : list of adjacent vertices indices
            dim         : dimension of the variable
        """

        self.name = "Fairness_"+var_name

        num_rows_Q = 0 # Rows for quadrilateral vertices
        num_rows_L = 0 # Rows for Laplacian

        # Valence != 4 vertices  Laplacian
        vk = []
        vk_n = []

        # Quadrilateral vertices indices
        du_i = []
        du_j = []
        dv_i = []
        dv_j = []
        
        # Non-quad vertices indices
        dv_k = []

        for i, _ in enumerate(adj_v):

            if len(adj_v[i]) == 4:
                num_rows_Q += 1

                dv_k.extend([i])
                du_i.extend([adj_v[i][0]])
                du_j.extend([adj_v[i][2]])

                dv_i.extend([adj_v[i][1]])
                dv_j.extend([adj_v[i][3]])

            elif len(adj_v[i]) != 4:
                num_rows_L += 1

                vk_n.append(adj_v[i])
                vk.extend([i])
                
        # Tansform to numpy arrays
        # Quadrilateral vertices
        du_i = np.array(du_i)
        du_j = np.array(du_j)
        dv_i = np.array(dv_i)
        dv_j = np.array(dv_j)
        dv_k = np.array(dv_k)


        if dim > 1:
            self.du_j = var_idx[var_name][3 * np.repeat(du_j, dim) + np.tile(range(dim), len(du_j))]
            self.du_i = var_idx[var_name][3 * np.repeat(du_i, dim) + np.tile(range(dim), len(du_i))]
            self.dv_j = var_idx[var_name][3 * np.repeat(dv_j, dim) + np.tile(range(dim), len(dv_j))]
            self.dv_i = var_idx[var_name][3 * np.repeat(dv_i, dim) + np.tile(range(dim), len(dv_i))]
            self.dv_k = var_idx[var_name][3 * np.repeat(dv_k, dim) + np.tile(range(dim), len(dv_k))]

            self.vk = var_idx[var_name][3 * np.repeat(vk, dim) + np.tile(range(dim), len(vk))]

            for i in range(len(vk_n)):
                vk_n[i] = var_idx[var_name][3 * np.repeat(vk_n[i], dim) + np.tile(range(dim), len(vk_n[i]))]

            self.vk_n = vk_n

        else:
            self.du_j = var_idx[var_name][du_j]
            self.du_i = var_idx[var_name][du_i]
            self.dv_j = var_idx[var_name][dv_j]
            self.dv_i = var_idx[var_name][dv_i]
            self.dv_k = var_idx[var_name][dv_k]

        # Quad fairness
        self.add_constraint("Du_Fair", 3*num_rows_Q)
        self.add_constraint("Dv_Fair", 3*num_rows_Q)
        
        # Laplacian fairness
        self.add_constraint("Lap_Fair", 3*num_rows_L)

        # Row indices
        self.row_du_Q =  self.const_idx["Du_Fair"]
        self.row_dv_Q =  self.const_idx["Dv_Fair"]

        
    def compute(self, X, var_idx):
        
        # Get variables
        du = X[self.du_i] +  X[self.du_j] - 2*X[self.dv_k]
        dv = X[self.dv_i] +  X[self.dv_j] - 2*X[self.dv_k]

        # Compute derivatives d_vi,j (du) = 1, d_vk = -2
        d_v  =  np.ones(len(self.du_i))
        d_vk = -2*np.ones(len(self.du_i))

        # Compute derivatives
        self.add_derivatives(self.row_du_Q, self.du_i, d_v)
        self.add_derivatives(self.row_du_Q, self.du_j, d_v)
        self.add_derivatives(self.row_du_Q, self.dv_k, d_vk)

        # Compute derivatives d_vi,j (dv) = 1, d_vk = -2
        self.add_derivatives(self.row_dv_Q, self.dv_i, d_v)
        self.add_derivatives(self.row_dv_Q, self.dv_j, d_v)
        self.add_derivatives(self.row_dv_Q, self.dv_k, d_vk)

        # Compute residuals
        self.set_r(self.const_idx["Du_Fair"], du)
        self.set_r(self.const_idx["Dv_Fair"], dv)

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

        if self.cont % self.dec_setp == 0:
            self.w *= self.dec_fac
        #if self.cont %25 == 0:
        #    self.w = 0
            #print(f"Weight decrease to {self.w}")
        





            
            