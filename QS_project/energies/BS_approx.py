# Call parent class
from optimization.constraint import Constraint
import splipy as sp
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
        self.bs1 = None # BS1 evaluated points
        self.bs2 = None # BS2 (knots, control points, degree)
        self.u_pts = None # U points
        self.v_pts = None # V points
        
      
    def initialize_constraint(self, X, var_idx, surf1, surf2, u_sample, v_sample) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            surf1     : BSpline surface 1
            surf2     : BSpline surface 2
            u_sample  : U sample points
            v_sample  : V sample points
        """

        # Define u_points and v_points
        self.u_pts, self.v_pts = np.linspace(0, 1, u_sample), np.linspace(0, 1, v_sample)

        # Evaluate points on the first surface
        self.bs1    = surf1
    
        # Set BSpline surfaces to optimize
        self.bs2     = surf2

        X[var_idx["cp"]] = surf2["control_points"].flatten()

        #print("Control points Init:\n", surf2.controlpoints)

        self.add_constraint("BS_approx", u_sample*v_sample*3)


    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """ 
    
        # Get basis
        u2_basis, v2_basis = self.bs2["basis_u"], self.bs2["basis_v"]
        u1_basis, v1_basis = self.bs1["basis_u"], self.bs1["basis_v"]

        # Get degrees
        deg_u, deg_v = u2_basis.order, v2_basis.order

        # Get the control points for bs2
        cp = X[var_idx["cp"]].reshape(-1, 3)

        # Create surface for evaluations
        surf_2 = sp.Surface(u2_basis, v2_basis, cp)

        surf_1 = sp.Surface(u1_basis, v1_basis, self.bs1["control_points"])
 
        # Evaluate the points on the second surface
        bs2_pts = surf_2(self.u_pts, self.v_pts)

        bs1_pts = surf_1(self.u_pts, self.v_pts)

        # Evaluate basis functions
        N_u_i = u2_basis.evaluate(self.u_pts)
        M_v_i = v2_basis.evaluate(self.v_pts)

        num_ctrl_pts = len(self.bs2["control_points"]) 
        

        print(num_ctrl_pts)
      
        # Rethink this part carefully
        rows = []
        cols = []
        vals = []
        # Try smaller example
        # Control points cp[ 3*(u+ len(v)*v) + 1 ]
        for i in range(len(self.u_pts)):
            for j in range(len(self.v_pts)):

                # Grid point
                g_p = 3*(len(self.v_pts)*i + j)
                
                for u in range(deg_u):
                    for v in range(deg_v):
                        # Get indices control point
                        cp_id_x = 3*(deg_v*u + v)
                        cp_id_y = 3*(deg_v*u + v) + 1
                        cp_id_z = 3*(deg_v*u + v) + 2
                        rows.extend(self.const_idx["BS_approx"][[g_p, g_p + 1, g_p + 2]])
                        cols.extend(var_idx["cp"][[cp_id_x, cp_id_y, cp_id_z]])
                        vals.extend([-N_u_i[i][u]*M_v_i[j][v],-N_u_i[i][u]*M_v_i[j][v], -N_u_i[i][u]*M_v_i[j][v]])

        print("rows", rows[-1])
        self.add_derivatives(rows, cols, vals)
        self.set_r(self.const_idx["BS_approx"], (bs1_pts.flatten() - bs2_pts.flatten()))




        
        

        
                

        


