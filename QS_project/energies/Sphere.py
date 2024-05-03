# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np


class Sphere(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{supp} = \sum_{f\in Mesh Dual} \sum_{vi in f} || (vi - cf)^2 - rf^2 ||^2
        """
        super().__init__()
        self.name = "Sphere" # Name of the constraint

      
    def initialize_constraint(self, X, var_idx, vertex_sph, aux) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            vertex_sph  : list of vertices per face
        """

    
        # Define u_points and v_points
        self.vertex_sph = vertex_sph.flatten()

        # Row indices
        self.row_vk = self.row_B =  np.repeat( np.arange(len(self.vertex_sph)), 3)
        self.row_A  = self.row_C =  np.arange(len(self.vertex_sph))

        # Col indices
        # v_k 
        fix_idx_vertex_sph = 3* np.repeat(self.vertex_sph, 3) + np.tile(range(3), len(self.vertex_sph))
        self.vk_idx = var_idx["v"][fix_idx_vertex_sph]

        # A
        self.dA_idx = var_idx["A"].repeat(4)

        # B
        self.dB_idx = var_idx["B"].reshape(-1,3).repeat(4, axis=0).flatten()

        # C
        self.dC_idx = var_idx["C"].repeat(4)
    
        # Number of constraints
        self.add_constraint("sph", len(self.vertex_sph))
        
        
    def compute(self, X, var_idx):
        """ Compute the constraint
        """
        # Get the variables
        v_ij = X[self.vk_idx]
        A = X[self.dA_idx]
        B = X[self.dB_idx]
        C = X[self.dC_idx]

        # Compute values
        v_ij = v_ij.reshape(-1, 3)
        # vij^2
        v_ij_2 = np.linalg.norm(v_ij, axis=1)**2

        # B in vector form
        VB = B.reshape(-1, 3)

        # d_vij = 2*A*v_ij - B
        d_vij = (2*A)[:,None]*v_ij - VB
        self.add_derivatives(self.row_vk, self.vk_idx, d_vij.flatten())

        # d_A = v_ij^2 
        self.add_derivatives(self.row_A, self.dA_idx, v_ij_2)

        # d_B = -v_ij
        self.add_derivatives(self.row_B, self.dB_idx, -v_ij.flatten())

        # d_C = 1
        self.add_derivatives(self.row_C, self.dC_idx, np.ones(len(self.vertex_sph)))


        # res
        res = A*v_ij_2 - np.einsum('ij,ij->i', v_ij, VB) + C

        self.set_r(self.const_idx["sph"], res)

            
            