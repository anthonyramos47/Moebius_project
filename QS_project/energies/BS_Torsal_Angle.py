# Call parent class
from optimization.constraint import Constraint
from scipy.interpolate import BSpline, bisplev, bisplrep
from geometry.utils import *
from geometry.bsplines_functions import *
import numpy as np
from time import time



class BS_Torsal_Angle(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that constraint torsal plane angles
        Energy :  \sum_{f \in Grid} || (nt1.nt2)^2 - cos(a)^2 + theta^2 ||^2
        where, 
            nt1, nt2 .- Torsal plane normals
            cos(a)   .- Cosine of the angle threshold
            theta    .- Dummy variable
        """
        super().__init__()
        self.name = "BS_Torsal_Angle" # Name of the constraint
        self.cos_a_2 = None # Cosine of the angle threshold
      
    def initialize_constraint(self, X, var_idx, angle, angle2) -> None:
        """ 
        Input:
            X : Variables
            var_idx : dictionary of indices of variables
            angle : Angle threshold in degrees
        """

        nt1 = self.uncurry_X(X, var_idx, "nt1")

        nt1 = nt1.reshape(-1, 3)

        F = len(nt1)

        self.cos_a_2 = np.cos(angle*np.pi/180)**2

        # Add constraints
        self.add_constraint("nt1_nt2", F)
        

    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """
        

        nt1, nt2, theta = self.uncurry_X(X, var_idx, "nt1", "nt2", "theta")

        nt1 = nt1.reshape(-1, 3)
        nt2 = nt2.reshape(-1, 3)

        dot_nt1_nt2 = vec_dot(nt1, nt2)
        cols_R3 = self.const_idx["nt1_nt2"].repeat(3)

        # E = || (nt1.nt2)^2 - cos(a)^2 + theta^2 ||^2
        # d_nt1_E = 2*(nt1.nt2)*nt2 
        d_nt1_E = (2*dot_nt1_nt2)[:,None]*nt2
        self.add_derivatives(
            cols_R3,
            var_idx["nt1"],
            d_nt1_E.flatten()
        )

        # d_nt2_E = 2*(nt1.nt2)*nt1
        d_nt2_E = (2*dot_nt1_nt2)[:,None]*nt1
        self.add_derivatives(
            cols_R3,
            var_idx["nt2"],
            d_nt2_E.flatten()
        )

        # d_theta_E = 2*theta
        d_theta_E = 2*theta
        self.add_derivatives(
            self.const_idx["nt1_nt2"],
            var_idx["theta"],
            d_theta_E
        )

        # r = (nt1.nt2)^2 - cos(a)^2 + theta^2
        r = vec_dot(nt1, nt2)**2 - self.cos_a_2 + theta**2
        self.set_r(self.const_idx["nt1_nt2"], r)
