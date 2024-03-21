# Call parent class
from optimization.constraint import Constraint
from scipy.interpolate import BSpline, bisplev, bisplrep
from geometry.utils import *
from geometry.bsplines_functions import *
import numpy as np



class BS_LC_Orth(Constraint):

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
        self.name = "BS_LC_orth" # Name of the constraint
        self.nu = None # Normals derivative u dir evaluated at grid points
        self.nv = None # Normals derivative v dir evaluated at grid points
        self.su = None # Surface derivative u dir evaluated at grid points
        self.sv = None # Surface derivative v dir evaluated at grid points
        self.n  = None # Normals evaluated at grid points
        self.r_bs = None # B-spline surface (graph) r(u,v) \in R  
        # Derivatives matrices
        self.d_a_cu = None # Derivative of the mid mesh with respect to the control points
        self.d_a_cv = None # Derivative of the mid mesh with respect to the control points
        self.u_pts = None # U points
        self.v_pts = None # V points
        self.cos_alpha = None # Cosine of the angle threshold
        
      
    def initialize_constraint(self, X, var_idx, bs1, r_bsp, u_pts, v_pts, orient, angle) -> None:
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
        self.n = orient*bs1.normal(self.u_pts, self.v_pts)

        # Set the B-spline surface r(u,v)
        self.r_bs = r_bsp

        # Get normal derivatives
        self.nu, self.nv = normal_derivatives_uv(bs1, self.u_pts, self.v_pts)

        # Set angle threshold
        self.cos_alpha = np.cos(angle*np.pi/180)**2

        # Add contraints
        self.add_constraint("orth", len(self.u_pts)*len(self.v_pts))


    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """
        
        # Get new control points
        l, u = self.uncurry_X(X, var_idx, "l", "mu")

        # Reshape l 
        l = l.reshape(len(self.u_pts), len(self.v_pts), 3)

        # compute l. n
        l_n = np.sum(l*self.n, axis=2)

        # E_orth = || (l.n)^2 - cos(a)^2 -  mu^2 ||^2
        # d_l E_orth = 2 (l.n) n
        d_l_EO = 2*l_n[:,:,None]*self.n
        self.add_derivatives(
            self.const_idx["orth"].repeat(3),
            var_idx["l"],
            d_l_EO.flatten()
        )

        # d_mu E_orth =  2 mu 
        self.add_derivatives(
            self.const_idx["orth"],
            var_idx["mu"],
            -2*u
        )

        # Set residual = (l.n)^2 - 1
        self.set_r(self.const_idx["orth"], (l_n.flatten())**2 - self.cos_alpha - u**2)
        




        
        

        
                

        


