# Creation of the constraint class
# This is meant to be a template to compute constraints
# The main idea is to pass a mesh and return a residual and a Jacobian
import numpy as np
import geometry as geo
from scipy.sparse import csc_matrix


class Constraint():
    def __init__(self) -> None:
        self.name = None # Name of the constraint
        self.w = 1 # Weight
        self._J = None # Jacobian matrix
        self._J0 = None # Constant Jacobian
        self._i = [] # Row index
        self._j = [] # Column index
        self._values = [] # Values
        self._r = None # Residual vector
        self.const = 0 # Num Constraints
        self.var = None # Num Variables
        self.sparse = True # Sparse matrix
        self.const_idx = {} # Dic to store the index of the constraints

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


        # print("shape i:", len(self._i)) 
        # print("shape j:", len(self._j))
        # print("shape values:", len(self._values))
        if self.sparse:
            self._J = csc_matrix((np.array(self._values), (self._i, self._j)), shape=(self.const, self.var))
        else:
            self._J = csc_matrix(self._J)

        #print("Jacobian", self.J.toarray())

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
        # Fill row
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
        self._i = []
        self._j = []
        self._values = [] 
        self._J = None
        self._r = np.zeros(self.const, dtype=np.float64)




    def print_per_const_energy(self):

        print("\n")
        for const in self.const_idx:
            print(f"{const}: {np.sum(self._r[self.const_idx[const]]**2)}")
            
        

    



        
