# Creation of the constraint class
# This is meant to be a template to compute constraints
# The main idea is to pass a mesh and return a residual and a Jacobian
import numpy as np
import geometry as geo
from scipy.sparse import coo_matrix
from time import time


class Constraint():
    def __init__(self) -> None:
        self.name = None # Name of the constraint
        self.w = 1 # Weight
        self._J = None # Jacobian matrix
        self.J_constant = None # Constant Jacobian
        self._J0 = None # Constant Jacobian
        self.J0_done = False # Jacobian computed
        self.idx_done = False # Indices computed
        self._i = [] # Row index
        self._j = [] # Column index
        self._values = [] # Values
        self._r = None # Residual vector
        self.const = 0 # Num Constraints
        self.var = None # Num Variables
        self.sparse = True # Sparse matrix
        self.sum_energy = False # Define if we are going to sum the contribution of the energy 
        self.const_idx = {} # Dic to store the index of the constraints
        self.first_compt = False # First computation

    def initialize_constraint(self, X, var_indices, *args) -> None:
        """ Method to initialize the constraint
            Input:
                X: Variables
                var_indices: Indices of the variables
                args: Arguments of the constraint
        """
        pass

    def add_constraint(self, name, dim):
        """ Method to add a constraint
            Input:
                name: Name of the constraint
                dim: Dimension of the constraint
        """
        self.const_idx[name] = np.arange(self.const, self.const + dim)
        self.const += dim

    def _initialize_constraint(self, X, var_indices, *args) -> None:
        """
        Method to initialize the all initial data for the constraint
        """

        # Initialization function
        self.initialize_constraint(X, var_indices, *args)

        # Compute the indices for rows and columns of the Jacobian, since those remain constant
    
        # Get the number of variables
        self.var = len(X)
        


    def _compute(self, X, var_idx) -> None:
        """
        Method to compute the residual and the Jacobian
        """
        self.reset()    
        self.compute(X, var_idx)
        
        # Fix values after the first computation
        if not self.first_compt:
            self.first_compt = True
            self.idx_done = True

        # If J is constant, we compute it only once
        if self.J_constant and not self.J0_done:
            self._J = coo_matrix((np.array(self._values), (self._i, self._j)), shape=(self.const, self.var))

            self._J.tocsr()

            self.H = self.w * self._J.T.dot(self._J)

            self.b = self.w * self._J.T.dot(self._r)

            self._J0 = self._J.copy()
            self.J0_done = True

        # We only compute the residual and the Jacobian if J is not constant
        elif self.J0_done:
            self.b = self.w * self._J0.T.dot(self._r)
    
        # If J is not constant, we compute it every time
        if not self.J_constant:
            self._J = coo_matrix((np.array(self._values), (self._i, self._j)), shape=(self.const, self.var))
            self._J.tocsr()
            self.H = self.w * self._J.T.dot(self._J)
            self.b = self.w * self._J.T.dot(self._r)

        # if self.sparse:
        #     self._J = csr_matrix((np.array(self._values), (self._i, self._j)), shape=(self.const, self.var))
        # else:
        #     self._J = csr_matrix(self._J)
        
        

    def compute(self, X) -> None:
        """ Function to compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                args: Arguments of the constraint
        """
        pass

    def define_rows_cols_J(self, c_idx, v_idx):
        """ Function to define the rows and columns of the Jacobian
            Input:
                c_idx: Constraint indices
                v_idx: Variable indices
        """
        self._i = c_idx
        self._j = v_idx

    def set_derivatives(self, values, c_idx=None, v_idx=None):
        """ Function to set the derivatives of the constraint
            Input:
                values: Values of the derivatives
                c_idx: Constraint indices
                v_idx: Variable indices
        """

        if c_idx is None or v_idx is None:
            self._values = values
        else:
            self._i = c_idx
            self._j = v_idx
            self._values = values

    def add_derivatives(self,  c_idx, v_idx, values):
        """ Function to add the derivatives of the constraint
            Input:
                values: Values of the derivatives
                c_idx: Constraint indices
                v_idx: Variable indices
        """
        if self.idx_done:
            self._values.extend(values)
        else:
            self._i.extend(c_idx)
            self._j.extend(v_idx)
            self._values.extend(values)

    def set_r(self, c_idx,  values):
        """ Function to set the residual
            Input:
                c_idx: Constraint indices
                values: Values of r
        """
        if c_idx is None:
            self._r = values
        else:
            self._r[c_idx] = values

    def set_weigth(self, w):
        """ Function to set the weight
            Input:
                w: Weight
        """
        self.w = w

    def uncurry_X(self, X, var_idx, *v_idx):
        """ Function to uncurry the variables
            Input:
                X: Variables
                v_idx: Variable indices
        """

        if len(v_idx) == 1:
            return X[var_idx[v_idx[0]]]
        else:
            return [X[var_idx[k]] for k in v_idx]
        
    def set_X(self, X, var, values):

        X[self.var_idx[var]] = values

    
    def reset(self):
        """ Function to clear the Jacobian
        """
        self._values = [] 
        self._J = None
        self._r = np.zeros(self.const, dtype=np.float64)

    def print_per_const_energy(self):

        print("\n")
        for const in self.const_idx:
            print(f"{const}: {np.sum(self._r[self.const_idx[const]]**2)}")
            
        

    



        
