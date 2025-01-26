# Planarity constraint implementation
import numpy as np
from optimization.constraint import Constraint

class Step_Control(Constraint):

    class Unit:
        """
        This class represent a unit vector constraint for desired variable of any dimention.
        """
        def __init__(self) -> None:
            """
            Initializes a new instance of the Unit class.
            """
            self.dim = None # Dimension of the variable
            self.v_name = None # Variable name
            self.prev = None
            super().__init__()
        
        

    def initialize_constraint(self, X, var_indices, var_name) -> None:
        """
        Initializes the constraint.
        Input:
            var_indices: Dictionary with the indices of the variables
            var_name: Name of the variable that we want to constraint
            dim: Dimension of the variable
        """
        # Variable name
        self.v_name = var_name

        # X0
        self.prev = X[var_indices[var_name]]

        # Setup constraints
        const_dim = len(var_indices[self.v_name]) # Number of constraints

        self.add_constraint("step", const_dim) # Add constraint (unit vector constraint
        
  

    def compute(self, X, var_idx) -> None:
        """ 
        Function that compute the gradients and the residuals of the constraint.
        Input:
            X: Variables
        """
        # Get variable
        xv = self.uncurry_X(X, var_idx, self.v_name) # flattened variable
    
        # Compute Jacobian for || v - v* ||^2
        self.add_derivatives(self.const_idx["step"], var_idx[self.v_name], np.ones(len(xv)))

        # Update residual  
        self.set_r(self.const_idx["step"], xv - self.prev)

        self.prev = xv

