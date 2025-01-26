# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np


class Sph_Unit(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{supp} = \sum_{f\in Mesh Dual} \sum_{vi in f} || (vi - cf)^2 - rf^2 ||^2
        """
        super().__init__()
        self.name = "Sphere_unit" # Name of the constraint

      
    def initialize_constraint(self, X, var_idx) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
        """

        n = len(X[var_idx["A"]])

        self.rows_B = np.repeat(np.arange(n), 3)
        self.rows_A = self.rows_C = np.arange(n)

        self.add_constraint("sph_unit", n)

        
        
        
    def compute(self, X, var_idx):
        """ Compute the constraint
        """
        # Get the variables
        A = X[var_idx["A"]]
        B = X[var_idx["B"]]
        C = X[var_idx["C"]]

        B = B.reshape(-1, 3)

        # d_B = 2B
        d_B = 2*B
        self.add_derivatives(self.rows_B, var_idx["B"], d_B.flatten())

        # d_A = -4 C
        d_A = - 4*C
        self.add_derivatives(self.rows_A, var_idx["A"], d_A)

        # d_C = - 4 A
        d_C = - 4*A
        self.add_derivatives(self.rows_C, var_idx["C"], d_C)

        # res
        res = np.einsum('ij,ij ->i', B, B) - 4*A*C - 1

        self.set_r(self.const_idx["sph_unit"], res) 

            
            