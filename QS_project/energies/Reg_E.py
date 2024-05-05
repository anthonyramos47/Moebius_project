# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np
from geometry.utils import indices_flatten_dim


class Reg_E(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{Reg} = \sum_{e_f_f\inE} || (B1 A2 - A1 B2)(w - u)/||w-u|| ||^2
        """
        super().__init__()
        self.name = "Reg_E" # Name of the constraint

      
    def initialize_constraint(self, X, var_idx, e_f_f, e_v_v) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            vertex_sph  : list of vertices per face
        """


        fi = e_f_f[0]
        fj = e_f_f[1]

        w = X[var_idx["v"]][indices_flatten_dim(e_v_v[1], 3)].reshape(-1, 3)
        u = X[var_idx["v"]][indices_flatten_dim(e_v_v[0], 3)].reshape(-1, 3)

        # Indices of vertices per edge
        self.u_idx = var_idx["v"][indices_flatten_dim(e_v_v[0], n=3)]
        self.w_idx = var_idx["v"][indices_flatten_dim(e_v_v[1], n=3)]

        # Indcies of the line congruence at the vertices of the edge
        self.lu_idx = var_idx["nd"][indices_flatten_dim(e_v_v[0], n=3)]
        self.lw_idx = var_idx["nd"][indices_flatten_dim(e_v_v[1], n=3)]


        # Indices of the variables
        # A
        self.Ai_idx = var_idx["A"][fi]
        self.Aj_idx = var_idx["A"][fj]

        # B
        self.Bi_idx = var_idx["B"][indices_flatten_dim(fi, n=3)]
        self.Bj_idx = var_idx["B"][indices_flatten_dim(fj, n=3)]

        # Number of constraints
        self.add_constraint("E_uw", len(fi))
        self.add_constraint("E_lu", len(fi))
        self.add_constraint("E_lw", len(fi))

        # Row indices E_uw
        self.row_E_uw_3 = self.const_idx["E_uw"].repeat(3)
        self.row_E_uw = self.const_idx["E_uw"]

        # row indices E_lu
        self.row_E_lu = self.const_idx["E_lu"].repeat(3)

        # row indices E_lw
        self.row_E_lw = self.const_idx["E_lw"].repeat(3)


        self.wu = np.linalg.norm(w-u, axis=1)



        
        
    def compute(self, X, var_idx):
        """ Compute the constraint
        """

        # Get sphere centers
        A_i = X[self.Ai_idx]
        A_j = X[self.Aj_idx]
        B_i = X[self.Bi_idx].reshape(-1, 3)
        B_j = X[self.Bj_idx].reshape(-1, 3)

        # Get Vertices
        u = X[self.u_idx].reshape(-1, 3)
        w = X[self.w_idx].reshape(-1, 3)

        # Get line congruences
        lu = X[self.lu_idx].reshape(-1, 3)
        lw = X[self.lw_idx].reshape(-1, 3)

        # Compute edge vector
        w_u = w - u

        w_u_n = w_u/self.wu[:,None]

        #
        cij =  A_j[:,None]*B_i - A_i[:,None]*B_j

        # E_uw = || (B1 A2 - A1 B2)(w - u)/||w-u|| ||^2
        # d_B1 = A2 (w-u)/||w-u||
        d_B1 =   A_j[:,None]*w_u_n
        self.add_derivatives(self.row_E_uw_3, self.Bi_idx, d_B1.flatten())

        # d_B2 = -A1 nl
        d_B2 =  -A_i[:,None]*w_u_n
        self.add_derivatives(self.row_E_uw_3, self.Bj_idx, d_B2.flatten())

        # d_A1 = -B2 (w-u)/||w-u||
        d_A1 = - np.einsum('ij,ij->i', B_j, w_u_n)
        self.add_derivatives(self.row_E_uw, self.Ai_idx, d_A1)

        # d_A2 = B1 (w-u)/||w-u||
        d_A2 = np.einsum('ij,ij->i', B_i, w_u_n)
        self.add_derivatives(self.row_E_uw, self.Aj_idx, d_A2)

        # d_u = -(A2 B1 - A1 B2)/||w-u||
        d_u = - cij/self.wu[:,None]
        self.add_derivatives(self.row_E_uw_3, self.u_idx,  d_u.flatten())

        # d_w = (A2 B1 - A1 B2)/||w-u||
        self.add_derivatives(self.row_E_uw_3, self.w_idx, -d_u.flatten())
        
        # res = (A_j B_i - A_i B_j)(w - u)/||w-u||
        res = np.einsum('ij,ij->i', cij, w_u_n)
        self.set_r(self.row_E_uw, res)

        # # E_lu = l_u(B2 A1 - A2 B1)
        # # d_B2 = A1 l_u
        # d_B2 = A_i[:,None]*lu
        # self.add_derivatives(self.row_E_lu, self.Bj_idx, d_B2.flatten())

        # # d_B1 = -A2 l_u
        # d_B1 = -A_j[:,None]*lu
        # self.add_derivatives(self.row_E_lu, self.Bi_idx, d_B1.flatten())

        # # d_A1 = B2 l_u
        # d_A1 = np.einsum('ij,ij ->i',  B_j,lu)
        # self.add_derivatives(self.const_idx["E_lu"], self.Ai_idx, d_A1)

        # # d_A2 = -B1 l_u
        # d_A2 = -np.einsum('ij,ij ->i',  B_i,lu)
        # self.add_derivatives(self.const_idx["E_lu"], self.Aj_idx, d_A2)

        # # d_lu = (B2 A1 - A2 B1)
        # self.add_derivatives(self.row_E_lu, self.lu_idx, cij.flatten())

        # # res = l_u(B2 A1 - A2 B1)
        # res = np.einsum('ij,ij ->i', lu, cij)
        # self.set_r(self.const_idx["E_lu"], res)

        # # E_lw = l_w(B2 A1 - A2 B1)
        # # d_B2 = A1 l_w
        # d_B2 = A_i[:,None]*lw
        # self.add_derivatives(self.row_E_lw, self.Bj_idx, d_B2.flatten())

        # # d_B1 = -A2 l_w
        # d_B1 = -A_j[:,None]*lw
        # self.add_derivatives(self.row_E_lw, self.Bi_idx, d_B1.flatten())

        # # d_A1 = B2 l_w
        # d_A1 = np.einsum('ij,ij ->i',  B_j,lw)
        # self.add_derivatives(self.const_idx["E_lw"], self.Ai_idx, d_A1)

        # # d_A2 = -B1 l_w
        # d_A2 = -np.einsum('ij,ij ->i',  B_i,lw)
        # self.add_derivatives(self.const_idx["E_lw"], self.Aj_idx, d_A2)

        # # d_lw = (B2 A1 - A2 B1)
        # self.add_derivatives(self.row_E_lw, self.lw_idx, cij.flatten())

        # # res = l_w(B2 A1 - A2 B1)
        # res = np.einsum('ij,ij ->i', lw, cij)
        # self.set_r(self.const_idx["E_lw"], res)

        self.wu = np.linalg.norm(w_u, axis=1)
        

            
            