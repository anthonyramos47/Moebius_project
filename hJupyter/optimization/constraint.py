# Creation of the constraint class
# This is meant to be a template to compute constraints
# The main idea is to pass a mesh and return a residual and a Jacobian
import numpy as np
import geometry as geo
from scipy.sparse import csc_matrix


class Constraint():
    def __init__(self) -> None:
        self.w = 1 # Weight
        self.J = None # Jacobian matrix
        self.J0 = None # Constant Jacobian
        self.i = [] # Row index
        self.j = [] # Column index
        self.values = [] # Values
        self.r = None # Residual vector
        self.const = None # Num Constraints
        self.var = None # Num Variables
        self.var_idx = None # Dic to store the index of the variables
        self.const_idx = None # Dic to store the index of the constraints

    def initialize_constraint(self) -> None:
        """
        Method to initialize the all initial data for the constraint
        """
        pass

    def _compute(self, X, *args) -> None:
        
        # Reset Jacobian
        self.reset()

        # Compute residual and Jacobian
        self.compute(X, *args)

        # Set Jacobian
        self.J = csc_matrix((self.values, (self.i, self.j)), shape=(self.const, self.var))

        pass

    def compute(self, X, *args) -> None:
        """ Function to compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                args: Arguments of the constraint
        """
        pass

    def add_derivatives(self, c_idx, v_idx, values):
        """ Function to fill a row of the Jacobian
            Input:
                c_idx: Constraint indices
                v_idx: Variable indices
                values: Values of J
        """
        
        # Fill row
        self.i.extend(c_idx)
        self.j.extend(v_idx)
        self.values.extend(values)

    def set_r(self, c_idx, values):
        """ Function to set the residual
            Input:
                c_idx: Constraint indices
                values: Values of r
        """
        self.r[c_idx] = values

    def set_weigth(self, w):
        """ Function to set the weight
            Input:
                w: Weight
        """
        self.w = w

    def uncurry_X(self, X, *v_idx):
        """ Function to uncurry the variables
            Input:
                X: Variables
                v_idx: Variable indices
        """

        if len(v_idx) == 1:
            return X[self.var_idx[v_idx[0]]]
        else:
            return [X[self.var_idx[k]] for k in v_idx]
        
    def set_X(self, X, var, values):

        X[self.var_idx[var]] = values

        

    def reset(self):
        """ Function to clear the Jacobian
        """
        self.i = []
        self.j = []
        self.values = [] 
        self.J = None
        self.r = np.zeros(self.const, dtype=np.float64)




        

    



        
