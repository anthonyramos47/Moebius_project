# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np
from time import time
from utils.bsplines_functions import *

# Import any other library that you need ...
# Example:
# import numpy as np

class BS_Repair(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that minimize the distance between two BS surfaces
        Energy :  \sum_{pij \in Grid} || BS1(pij) - BS2(pij) ||^2
        where, 
            BS1, BS2 are the two BS surfaces
            pij is a point in the grid
        """
        super().__init__()
        self.name    = "BS_Repair" # Name of the constraint
        self.basis_u = None # u basis
        self.basis_v = None # v basis
        self.n       = None # Normals at grid points
        self.aij     = None # Control points
        self.u_pts   = None # U points
        self.v_pts   = None # V points
        #self.perv_H  = None # Mean curvature
        self.prev_ev = None # Previous evaluation   
        self.epsilon = None # epsilon
        self.J_constant = True
        

        
      
    def initialize_constraint(self, X, var_idx, order_u, order_v, knots_u, knots_v, control_points, u_vals, v_vals) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            surf     : BSpline surface 
            u_sample  : U sample points
            v_sample  : V sample points
        """

        # Define u_points and v_points
        self.u_pts, self.v_pts = u_vals, v_vals

        # Create the B-splines basis
        self.basis_u = sp.BSplineBasis(order_u, knots_u) 
        self.basis_v = sp.BSplineBasis(order_v, knots_v) 

        # Create the B-spline surface 
        bsp = sp.Surface(self.basis_u, self.basis_v, control_points)

        # compute normals
        self.n = bsp.normal(self.u_pts, self.v_pts)


        # Evaluate surface at control points
        #ev = 
        self.prev_ev = bsp(self.u_pts, self.v_pts).reshape(-1,3)

        # Set the lower limit for the mean curvature as the inverse of the diagonal length of the bbox of evaluation points
        self.epsilon = 1/(0.5*np.linalg.norm([np.min(control_points, axis=0), np.max(control_points, axis=0)]))

        #self.epsilon = 0

        print("Epsilon: ", self.epsilon)
        print("Max radius", 0.5*np.linalg.norm([np.min(control_points, axis=0), np.max(control_points, axis=0)]))

        # Repeat Identity matrix n times
        daij = np.eye(len(control_points)*3)

        # Initialize the row, column and values of the Jacobian
        self.row_idx = []
        self.col_idx = []
        self.vals    = []

        nu = len(u_vals)
        nv = len(v_vals)
        
        # Store the derivative of the B-spline surface with respect to the control points
        for aij in range(len(control_points)):
            aux_dbspline = sp.Surface(self.basis_u, self.basis_v, daij[3*aij].reshape(-1,3))
            # Derivative of the B-spline surface with respect to the control points
            self.vals.extend((aux_dbspline(self.u_pts, self.v_pts)[:,:,0]).repeat(3).flatten())

            self.col_idx.extend(np.tile(np.arange(3*var_idx["cp"][aij], var_idx["cp"][aij]*3+3),nu*nv).flatten())   


        self.row_idx = np.tile(np.arange(nu*nv*3),len(control_points))
                               

        self.add_constraint("BS_Repair", nu*nv*3)


    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """ 
            
        # Define the b-spline surface
        bsp = sp.Surface(self.basis_u, self.basis_v, X[var_idx["cp"]].reshape(-1,3))

        # Compute Mean curvature
        _, H, _ = curvatures_par(bsp, self.u_pts, self.v_pts)
        
        # Filter H values
        H = H.flatten()
        
        # Define number of positive and negative values
        num_pos =  len(np.where(H > self.epsilon)[0])
        num_neg =  len(np.where(H < - self.epsilon)[0])

        
        # # Define the regions with different H signs
        if num_pos > num_neg:
            idx = np.where(H < self.epsilon)[0]
            
        else:
            idx = np.where(H > -self.epsilon)[0]

        #idx = np.where(H < self.epsilon)[0]

        # Evaluate 
        s_uv = bsp(self.u_pts, self.v_pts)
        s_uv = s_uv.reshape(-1,3)

        # hn 
        #Hn = - (np.sign(H)[:,None]  * self.n.reshape(-1,3))
        Hn = -(( H + self.epsilon)[:,None]*self.n.reshape(-1,3))
        #Hn = -(H[:,None]*self.n.reshape(-1,3))

        # Indices on the residual vector
        idx_res = flat_array_variables(idx ,3)

        # Compute the residual
        res = np.zeros((len(s_uv)*3))
        res[idx_res] = (s_uv[idx] - self.prev_ev[idx] + Hn[idx]).flatten()

        # Compute the Jacobian
        self.add_derivatives(self.row_idx, self.col_idx, self.vals)

        # Set the residual
        self.set_r(self.const_idx["BS_Repair"], res)

        # Update prev evaluation
        self.prev_ev = s_uv










        
        

        
                

        


