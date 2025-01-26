# Planarity constraint implementation
import numpy as np
from optimization.constraint import Constraint

class Unit(Constraint):

    class Unit:
        """
        This class represent a unit vector constraint for desired variable of any dimention.
        """
        def __init__(self) -> None:
            """
            Initializes a new instance of the Unit class.
            """
            self.dim = None # Dimension of the variable
            self.J_constant = True # Constant Jacobian
            self.v_name = None # Variable name
            super().__init__()
            
        
    def initialize_constraint(self, X, var_indices, var_name, dim) -> None:
        """
        Initializes the constraint.
        Input:
            var_indices: Dictionary with the indices of the variables
            var_name: Name of the variable that we want to constraint
            dim: Dimension of the variable
        """   
        self.v_name = var_name
        self.dim = dim

        # Setup constraints
        const_dim = len(var_indices[self.v_name])//dim  # Number of constraints

        self.add_constraint("unit", const_dim) # Add constraint (unit vector constraint
        
  

    def compute(self, X, var_idx) -> None:
        """ 
        Function that compute the gradients and the residuals of the constraint.
        Input:
            X: Variables
        """
        
        # Get variable
        xv = self.uncurry_X(X, var_idx, self.v_name) # flattened variable
        xs = xv.reshape(-1, self.dim) # reshaped variable

        # Compute Jacobian for || e_f . e_f - 1||^2
        self.add_derivatives(self.const_idx["unit"].repeat(self.dim), var_idx[self.v_name], 2*xv)

        # Update residual  
        self.set_r(self.const_idx["unit"], np.sum ( xs*xs,  axis=1) - 1)

