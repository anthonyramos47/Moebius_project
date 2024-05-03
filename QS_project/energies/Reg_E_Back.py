# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np
from geometry.utils import indices_flatten_dim


class Reg_E(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{Reg} = \sum_{e_f_f\inE} || (B1 A2 - A1 B2)nd - 2 A1 A2 ||^2
        """
        super().__init__()
        self.name = "Reg_E" # Name of the constraint

      
    def initialize_constraint(self, X, var_idx, e_f_f, aux) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            vertex_sph  : list of vertices per face
        """

        fi = e_f_f[0]
        fj = e_f_f[1]


        # Indices of the variables
        # A
        self.Ai_idx = var_idx["A"][fi]
        self.Aj_idx = var_idx["A"][fj]

        # B
        self.Bi_idx = var_idx["B"][indices_flatten_dim(fi, n=3)]
        self.Bj_idx = var_idx["B"][indices_flatten_dim(fj, n=3)]


        # Number of constraints
        self.add_constraint("reg", len(fi))

        # dbi row indices
        self.row_Bi = self.row_Bj = self.row_n_l  = np.repeat(np.arange(len(fi)), 3)
        self.row_Ai = self.row_Aj = np.arange(len(fi))



        
        
    def compute(self, X, var_idx):
        """ Compute the constraint
        """
        # Get the variables
        A_i = X[self.Ai_idx]
        A_j = X[self.Aj_idx]
        B_i = X[self.Bi_idx].reshape(-1, 3)
        B_j = X[self.Bj_idx].reshape(-1, 3)
        
        n_l = self.uncurry_X(X, var_idx, "n_l")
        n_l = n_l.reshape(-1, 3)

        # E = || (B1 A2 - A1 B2)nl - 2 A1 A2 ||^2
        # d_B1 = A2 nl
        d_B1 = A_j[:,None]*n_l
        self.add_derivatives(self.row_Bi, self.Bi_idx, d_B1.flatten())

        # d_B2 = -A1 nl
        d_B2 = -A_i[:,None]*n_l
        self.add_derivatives(self.row_Bj, self.Bj_idx, d_B2.flatten())

        # d_A1 = - B2 nl -2 A2
        d_A1 = - np.einsum('ij,ij->i',B_j, n_l) - 2*A_j
        self.add_derivatives(self.row_Ai, self.Ai_idx, d_A1)

        # d_A2 = -2 A1
        d_A2 = np.einsum('ij,ij->i', B_i, n_l) - 2*A_i
        self.add_derivatives(self.row_Aj, self.Aj_idx, d_A2)

        # d_nl = (B1 A2 - A1 B2)
        d_nl = (B_i*A_j[:,None] - A_i[:,None]*B_j)
        self.add_derivatives(self.row_n_l, var_idx["n_l"], d_nl.flatten())

        # res
        res = np.einsum('ij,ij->i', d_nl, n_l) - 2*A_i@A_j

        self.set_r(self.const_idx["reg"], res)


            
            