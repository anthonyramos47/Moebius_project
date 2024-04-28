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

        self.vk_3 = None # Valence 3 vertices
        self.vi_3 = None # Valence 3 vertices
        self.vj_3 = None # Valence 3 vertices

        self.vk_2 = None # Valence 2 vertices
        self.vi_2 = None # Valence 2 vertices
        self.vj_2 = None # Valence 2 vertices

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

        num_rows_Q = 0 # Rows for quadrilateral vertices
        num_rows_B = 0 # Rows for boundary vertices
        num_rows_C = 0 # Rows for corner vertices

        # Valence 2 vertices vi - 2 vj + vk = 0
        vk_2 = []
        vi_2 = []
        vj_2 = []

        # Valence 3 vertices vi - 2 vj + vk = 0
        vk_3 = []
        vi_3 = []
        vj_3 = []

        # Quadrilateral vertices indices
        du_i = []
        du_j = []

        dv_i = []
        dv_j = []

        dv_k = []
        for i, _ in enumerate(adj_v):

            if len(adj_v[i]) == 4:
                num_rows_Q += 1

                dv_k.extend([i])
                du_i.extend([adj_v[i][0]])
                du_j.extend([adj_v[i][2]])

                dv_i.extend([adj_v[i][1]])
                dv_j.extend([adj_v[i][3]])
            elif len(adj_v[i]) == 3:
                num_rows_B += 1

                vk_3.extend([i])
                vi_3.extend([adj_v[i][0]])
                vj_3.extend([adj_v[i][2]])
            
            elif len(adj_v[i]) == 2:
                num_rows_C += 1

                vk_2.extend([i])
                vi_2.extend([adj_v[i][0]])
                vj_2.extend([adj_v[i][1]])

        # Tansform to numpy arrays
        # Quadrilateral vertices
        du_i = np.array(du_i)
        du_j = np.array(du_j)
        dv_i = np.array(dv_i)
        dv_j = np.array(dv_j)
        dv_k = np.array(dv_k)

        # Boundary vertices
        vk_3 = np.array(vk_3)
        vi_3 = np.array(vi_3)
        vj_3 = np.array(vj_3)

        # Corner vertices
        vk_2 = np.array(vk_2)
        vi_2 = np.array(vi_2)

        if dim > 1:
            self.du_j = var_idx[var_name][3 * np.repeat(du_j, dim) + np.tile(range(dim), len(du_j))]
            self.du_i = var_idx[var_name][3 * np.repeat(du_i, dim) + np.tile(range(dim), len(du_i))]
            self.dv_j = var_idx[var_name][3 * np.repeat(dv_j, dim) + np.tile(range(dim), len(dv_j))]
            self.dv_i = var_idx[var_name][3 * np.repeat(dv_i, dim) + np.tile(range(dim), len(dv_i))]
            self.dv_k = var_idx[var_name][3 * np.repeat(dv_k, dim) + np.tile(range(dim), len(dv_k))]

            self.vi_3 = var_idx[var_name][3 * np.repeat(vi_3, dim) + np.tile(range(dim), len(vi_3))]
            self.vj_3 = var_idx[var_name][3 * np.repeat(vj_3, dim) + np.tile(range(dim), len(vj_3))]
            self.vk_3 = var_idx[var_name][3 * np.repeat(vk_3, dim) + np.tile(range(dim), len(vk_3))]

            self.vi_2 = var_idx[var_name][3 * np.repeat(vi_2, dim) + np.tile(range(dim), len(vi_2))]
            self.vj_2 = var_idx[var_name][3 * np.repeat(vj_2, dim) + np.tile(range(dim), len(vj_2))]
            self.vk_2 = var_idx[var_name][3 * np.repeat(vk_2, dim) + np.tile(range(dim), len(vk_2))]
        else:
            self.du_j = var_idx[var_name][du_j]
            self.du_i = var_idx[var_name][du_i]
            self.dv_j = var_idx[var_name][dv_j]
            self.dv_i = var_idx[var_name][dv_i]
            self.dv_k = var_idx[var_name][dv_k]

            self.vi_3 = var_idx[var_name][vi_3]
            self.vj_3 = var_idx[var_name][vj_3]
            self.vk_3 = var_idx[var_name][vk_3]

            self.vi_2 = var_idx[var_name][vi_2]
            self.vj_2 = var_idx[var_name][vj_2]

        self.add_constraint("Du_Fair", 3*num_rows_Q)
        self.add_constraint("Dv_Fair", 3*num_rows_Q)
        
        self.add_constraint("V3_Fair", 3*num_rows_B)
        self.add_constraint("V2_Fair", 3*num_rows_C)

        # Row indices
        self.row_du_Q =  self.const_idx["Du_Fair"]
        self.row_dv_Q =  self.const_idx["Dv_Fair"]

        self.row_v_3 =  self.const_idx["V3_Fair"]
        self.row_v_2 =  self.const_idx["V2_Fair"]


        
        
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


        # # Fairnes boundary vertices
        # v3 = X[self.vi_3] +  X[self.vj_3] - 2*X[self.vk_3]

        # # Compute derivatives d_vi,j (v3) = 1, d_vk = -2
        # d_v  =  np.ones(len(self.vi_3))
        # d_vk = -2*np.ones(len(self.vi_3))

        # # Compute derivatives
        # self.add_derivatives(self.row_v_3, self.vi_3, d_v)
        # self.add_derivatives(self.row_v_3, self.vj_3, d_v)
        # self.add_derivatives(self.row_v_3, self.vk_3, d_vk)

        # # Compute residuals
        # self.set_r(self.const_idx["V3_Fair"], v3)

        # Fairnes corner vertices
        v2 = X[self.vi_2] +  X[self.vj_2] - 2*X[self.vk_2]

        # Compute derivatives d_vi,j (v2) = 1, d_vk = -2
        d_v  =  np.ones(len(self.vi_2))
        d_vk = -2*np.ones(len(self.vi_2))

        # Compute derivatives
        self.add_derivatives(self.row_v_2, self.vi_2, d_v)
        self.add_derivatives(self.row_v_2, self.vj_2, d_v)
        self.add_derivatives(self.row_v_2, self.vk_2, d_vk)

        # Compute residuals
        self.set_r(self.const_idx["V2_Fair"], v2)

        self.cont+=1

        if self.cont %5 == 0:
            self.w = self.w/2
        if self.cont %25 == 0:
            self.w = 0
            #print(f"Weight decrease to {self.w}")
        





            
            