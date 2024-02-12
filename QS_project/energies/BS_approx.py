# Call parent class
from optimization.constraint import Constraint
from scipy.interpolate import BSpline, bisplev, bisplrep
import numpy as np

# Import any other library that you need ...
# Example:
# import numpy as np

class BS_approx(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that minimize the distance between two BS surfaces
        Energy :  \sum_{pij \in Grid} || BS1(pij) - BS2(pij) ||^2
        where, 
            BS1, BS2 are the two BS surfaces
            pij is a point in the grid
        """
        super().__init__()
        self.name = "BS_approx" # Name of the constraint
        self.bs1 = None # BS1 (knots, control points, degree)
        self.bs2 = None # BS2 (knots, control points, degree)
        self.u_pts = None # U points
        self.v_pts = None # V points
        self.d_alpha = None # Derivative of the control points
        #self.grid_points = None # Grid points
        
      
    def initialize_constraint(self, X, var_idx, grid_intervals, grid_size, bs1, bs2) -> None:
        """ 
        Input:
            X : Variables
            var_idx : dictionary of indices of variables
            grid_intervals : Number of intervals in the grid [(umin, vmin), (umax, vmax)]
            grid_size : Size of the grid (n, m)
        """

        # Set BSpline surfaces
        self.bs1 = bs1
        self.bs2 = bs2

        # Create the grid
        u_int = np.linspace(grid_intervals[0][0], grid_intervals[0][1], grid_size[0])
        v_int = np.linspace(grid_intervals[1][0], grid_intervals[1][1], grid_size[1])
        u_vals, v_vals = np.meshgrid(u_int, v_int, indexing='ij')

        self.u_pts = u_vals
        self.v_pts = v_vals

        self.d_alpha = np.eye((len(bs2[2])))
        
    
        X[var_idx["cp"]] = bs2[2]

        #self.grid_points = list(zip(u_vals.ravel(), v_vals.ravel()))
        self.add_constraint("BS_approx", grid_size[0]*grid_size[1])


    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """

        # Copy bs2 data 
        aux_bs2 = self.bs2.copy()
        aux_bs2[2] = self.uncurry_X(X, var_idx, "cp") # cp control points

        # dai bs2
        dai_bs2 = self.bs2.copy()

        # Compute derivatives
        for i in range(len(aux_bs2[2])): 
            
            dai_bs2[2] = self.d_alpha[i] # Set the derivative of the i-th control point

            print("dai_cp", self.d_alpha[i])

            # Compute the derivative of the control points
            da_bs2 = bisplev(self.u_pts[:,0], self.v_pts[0,:], dai_bs2)

            self.add_derivatives(self.const_idx["BS_approx"], var_idx["cp"][i].repeat(len(self.const_idx["BS_approx"])), -da_bs2.flatten())

  
        # Compute the residual
        self.set_r(self.const_idx["BS_approx"], (bisplev(self.u_pts[:,0], self.v_pts[0,:], self.bs1) - bisplev(self.u_pts[:,0], self.v_pts[0,:], aux_bs2)).flatten() )






        
        

        
                

        


