# Planarity constraint implementation
import numpy as np
from hanan.optimization.constraint import Constraint


class Template(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that minimize ..... 
        Energy :  <Write what it minimize>
        where, 
            <Specifications>
        """
        super().__init__()
        self.name = "Template" # Name of the constraint
        
        # Here define any other parameter constant parameter that you need
        # for the constraint

    def initialize_constraint(self, X, var_idx, args) -> None:
        """ 
        Input:
            These two are given by the optimizer class
            X : Variables
            var_idx : dictionary of indices of variables
            Arguments that are given by the user for the computation
            args:
                nV: Number of vertices
                nF: Number of faces
        """
        # Example:
        # self.nV = nV # Number of vertices
        # self.nF = nF # Number of faces
        
        # Add constraints using the add_constraint method
        # Example:
        # Name, dimension of the constraint or number of equations
        # self.add_constraint("E1", self.nF)

        # Add any other initialization that you need
        # Example:
        # self.t_ang = t_ang # Torsal angle target
        pass




    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """
        # Example:


        # Get variables
        # s_c, s_r, e = self.uncurry_X(X, var_idx, "sph_c", "sph_r", "e")      

        # Reshape variables
        # s_c = s_c.reshape(-1, 3)
        # e = e.reshape(-1, 3)

        # Example to add derivatives
        # d sph_c =>   2*(c_f - v_i)
        # self.add_derivatives(cols, rows, values):
        # self.add_derivatives(self.const_idx["Env1"].repeat(3), np.tile(var_idx["sph_c"],3), 2*cf_vi.flatten())
        
        # Example to set r
        # set_r(cols, values)
        # self.set_r(self.const_idx["Env1"], np.sum(cf_vi*cf_vi, axis=1) - np.tile(s_r*s_r, 3))

        pass
        
                

        


