# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np


class QM_Fairness(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{supp} = \sum_{f\in Mesh Dual} \sum_{vi in f} || (vi - cf)^2 - rf^2 ||^2
        """
        super().__init__()
        self.name = "QM_Fairness" # Name of the constraint
        self.du_i = None # Direction  u
        self.du_j = None # Direction  u
        self.dv_i = None # Direction v
        self.dv_j = None # Direction v

        self.row_du = None # Row indices du
        self.row_dv = None # Row indices dv

        # Cont for weight decrease
        self.cont = 0

        

        

      
    def initialize_constraint(self, X, var_idx, inn_v, adj_v, var_name, dim) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            inn_v       : list of inner vertices indices
            adj_v       : list of adjacent vertices indices

        """

        self.name = "QM_Fairness_"+var_name

        num_rows = 0 
        du_i = []
        du_j = []

        dv_i = []
        dv_j = []

        dv_k = []
        for i, vi in enumerate(inn_v):
            
            if len(adj_v[vi]) == 4:
                num_rows += 1

                dv_k.extend([vi])
                du_i.extend([adj_v[vi][0]])
                du_j.extend([adj_v[vi][2]])

                dv_i.extend([adj_v[vi][1]])
                dv_j.extend([adj_v[vi][3]])
        

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
        else:
            self.du_j = var_idx[var_name][du_j]
            self.du_i = var_idx[var_name][du_i]
            self.dv_j = var_idx[var_name][dv_j]
            self.dv_i = var_idx[var_name][dv_i]
            self.dv_k = var_idx[var_name][dv_k]

        self.add_constraint("Du_Fair", 3*num_rows)
        self.add_constraint("Dv_Fair", 3*num_rows)

        # Row indices
        self.row_du =  self.const_idx["Du_Fair"]
        self.row_dv =  self.const_idx["Dv_Fair"]


        # # Filter only 4 vertices
        # new_adj_v = []
        # for v_n in adj_v:
        #     if len(v_n) == 4:
        #         new_adj_v.append(v_n)
        
        # adj_v = np.array(new_adj_v)

        # # Get directions of the vertices of the mesh for fairness du,dv = [vi, vj]
        # self.du = adj_v[inn_v, 0], adj_v[inn_v, 2]
        # self.dv = adj_v[inn_v, 1], adj_v[inn_v, 3]

        # # Col indices
        # self.du_vi = self.du[0].flatten()
        # self.du_vi = 3* np.repeat(self.d_vi, 3) + np.tile(range(3), len(self.d_vi))

        # self.du_vj = self.du[1].flatten()
        # self.du_vj = 3* np.repeat(self.d_vj, 3) + np.tile(range(3), len(self.d_vj))

        # # Col indices dv
        # self.dv_vi = self.dv[0].flatten()
        # self.dv_vi = 3* np.repeat(self.d_vi, 3) + np.tile(range(3), len(self.d_vi))

        # self.dv_vj = self.dv[1].flatten()
        # self.dv_vj = 3* np.repeat(self.d_vj, 3) + np.tile(range(3), len(self.d_vj))
        
        # self.d_vk = 3* np.repeat(inn_v, 3) + np.tile(range(3), len(inn_v))

        # # add constraints
        # self.add_constraint("Du_Fair", len(inn_v))
        # self.add_constraint("Dv_Fair", len(inn_v))

        #     # Row indices
        # self.row_du =  np.repeat( self.const_idx["Du_Fair"], 3)

        # # Row indices
        # self.row_dv =  np.repeat( self.const_idx["Dv_Fair"], 3)
 
        
    def compute(self, X, var_idx):
        
        # Get variables
        du = X[self.du_i] +  X[self.du_j] - 2*X[self.dv_k]
        dv = X[self.dv_i] +  X[self.dv_j] - 2*X[self.dv_k]

        # Compute derivatives d_vi,j (du) = 1, d_vk = -2
        d_v  =  np.ones(len(self.du_i))
        d_vk = -2*np.ones(len(self.du_i))

        # Compute derivatives
        self.add_derivatives(self.row_du, self.du_i, d_v)
        self.add_derivatives(self.row_du, self.du_j, d_v)
        self.add_derivatives(self.row_du, self.dv_k, d_vk)

        # Compute derivatives d_vi,j (dv) = 1, d_vk = -2
        self.add_derivatives(self.row_dv, self.dv_i, d_v)
        self.add_derivatives(self.row_dv, self.dv_j, d_v)
        self.add_derivatives(self.row_dv, self.dv_k, d_vk)


        # Compute residuals
        self.set_r(self.const_idx["Du_Fair"], du)
        self.set_r(self.const_idx["Dv_Fair"], dv)

        self.cont+=1

        if self.cont %10 == 0:
            self.cont = 0
            self.w = self.w/2
            #print(f"Weight decrease to {self.w}")
        





            
            