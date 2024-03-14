# Call parent class
from optimization.constraint import Constraint
from scipy.interpolate import BSpline, bisplev, bisplrep
from geometry.utils import *
from geometry.bsplines_functions import *
import numpy as np



class BS_LC(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that minimize the angle between the line congruence l(u,v) and the surface normal n(u,v)
        Energy :  \sum_{pij \in Grid} || (l.n)^2 - cos(alpha)^2 - u^2 ||^2 
        where, 
            l(u,v) .- Line congruence normalized
            n(u,v) .- Surface normal
            alpha .- Angle threshold between the line congruence and the surface normal
            u .- dummy variable
        """
        super().__init__()
        self.name = "BS_LC" # Name of the constraint
        self.nu = None # Normals derivative u dir evaluated at grid points
        self.nv = None # Normals derivative v dir evaluated at grid points
        self.su = None # Surface derivative u dir evaluated at grid points
        self.sv = None # Surface derivative v dir evaluated at grid points
        self.n  = None # Normals evaluated at grid points
        self.r_bs = None # B-spline surface (graph) r(u,v) \in R  
        # Derivatives matrices
        self.d_a_cu = None # Derivative of the mid mesh with respect to the control points
        self.d_a_cv = None 
        self.u_pts = None # U points
        self.v_pts = None # V points
        self.l_norm = None # Norm of the line congruence
        self.cos_alpha = None # Cosine of the angle threshold
        
      
    def initialize_constraint(self, X, var_idx, bs1, r_bsp, u_pts, v_pts, angle) -> None:
        """ 
        Input:
            X : Variables
            var_idx : dictionary of indices of variables
            bs1 : BSpline surface 1
            grid_size : Size of the grid (n, m)
        """

        
        # Get grid points
        self.u_pts, self.v_pts = u_pts, v_pts
        
        # Get surface derivatives
        self.su = bs1.derivative(self.u_pts, self.v_pts, d=(1,0))
        self.sv = bs1.derivative(self.u_pts, self.v_pts, d=(0,1))
        
        # Compute normal
        self.n = bs1.normal(self.u_pts, self.v_pts)

        # Set the B-spline surface r(u,v)
        self.r_bs = r_bsp

        # Get normal derivatives
        self.nu, self.nv = normal_derivatives_uv(bs1, self.u_pts, self.v_pts)

        # Compute derivatives of the line congruence with respect to the control points [a0, a1, a2 ... an]
        cp_len = len(self.r_bs[2])

        # Auxiliar identity matrix
        d_a = np.eye(cp_len)

        # Auxiliar b-spline surface to compute the derivatives
        cp_da = self.r_bs.copy()

        self.d_a_cu = np.zeros((cp_len, len(self.u_pts), len(self.v_pts), 3))
        self.d_a_cv = np.zeros((cp_len, len(self.u_pts), len(self.v_pts), 3))

        # Compute the values of the J 
        for i in range(cp_len):

            # Get cooresponding a_i derivative
            d_cp = d_a[i]

            # Modify control points
            cp_da[2] = d_cp

            # Compute the derivative of the control points
            da_r_bs = bisplev(self.u_pts, self.v_pts, cp_da)
            da_ru   = bisplev(self.u_pts, self.v_pts, cp_da, dx=1, dy=0)
            da_rv   = bisplev(self.u_pts, self.v_pts, cp_da, dx=0, dy=1)

            self.d_a_cu[i] = da_ru[:,:,None]*self.n + da_r_bs[:,:,None]*self.nu
            self.d_a_cv[i] = da_rv[:,:,None]*self.n + da_r_bs[:,:,None]*self.nv

        # Set angle threshold
        self.cos_alpha = np.cos(angle)**2

        # Add contraints
        self.add_constraint("orth", len(self.u_pts)*len(self.v_pts))


    def d_c_uv(self, cp):
        """ Compute the line congruence l(u,v)
            Output:
                l(u,v) : Line congruence
        """
        # Update control points of the B-spline surface
        self.r_bs[2] = cp

        # Evaluate r(u,v) at grid points
        r_uv = bisplev(self.u_pts, self.v_pts, self.r_bs)

        # Compute derivatives of r(u,v)
        ru = bisplev(self.u_pts, self.v_pts, self.r_bs, dx=1, dy=0)
        rv = bisplev(self.u_pts, self.v_pts, self.r_bs, dx=0, dy=1)

        # Compute the mid mesh
        cu = self.su + ru[:,:,None]*self.n + r_uv[:,:,None]*self.nu
        cv = self.sv + rv[:,:,None]*self.n + r_uv[:,:,None]*self.nv

        return cu, cv


    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """
        
        # Get new control points
        cp, l = self.uncurry_X(X, var_idx, "rij", "l")
        

        # Compute the line congruence
        cu, cv = self.l_uv(cp)

        # Prod l.n 
        l_n = np.sum(l*self.n, axis=2)


        # Compute the values of the J 
        for i in range(len(cp)):

            # Compute the derivative of the line congruence l = c_u x c_v
            # cu = su + r_bs*n_u + r_u n; cv = sv + r_bs*n_v + r_v n
            # => da_l = (da_ru * n + da_r_bs * nu) x cv + cu x (da_rv * n + da_r_bs * nv)
            da_l = np.cross(self.d_a_cu[i], cv) + np.cross(cu, self.d_a_cv[i])

            d_l = np.sum(da_l*self.n, axis=2)/(self.l_norm**2)

            d_ln = 2*l_n*d_l

         
            self.add_derivatives(self.const_idx["orth"], var_idx["rij"][i].repeat(len(self.const_idx["orth"])), d_ln.flatten())

        #values.extend(2*u)
        

        # Compute the residual
        #r = ((l_n**2/(self.l_norm**2)).flatten() - self.cos_alpha - u**2)
        r = ((l_n**2/(self.l_norm**2)) - 1).flatten()
        
        self.set_r(self.const_idx["orth"], r)

        # Compute the norm of the line congruence
        self.l_norm = np.linalg.norm(l, axis=2)
    





        
        

        
                

        


