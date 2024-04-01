# Call parent class
from optimization.constraint import Constraint
from scipy.interpolate import BSpline, bisplev, bisplrep
from geometry.utils import *
from utils.bsplines_functions import *
import numpy as np



class BS_LC(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that minimize the angle between the line congruence l(u,v) and the surface normal n(u,v)
        Energy :  \sum_{pij \in Grid} || l.cu/||cu|| ||^2  + || l.cv/||cv|| ||^2 = E_cu + E_cv
        where, 
            l(u,v) .- Line congruence normalized
            c_u    .- Mid mesh derivative u dir
            c_v    .- Mid mesh derivative v dir
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
        self.cu_norm = None # Norm of the line congruence
        self.cv_norm = None # Norm of the line congruence
        self.cos_alpha = None # Cosine of the angle threshold
        
      
    def initialize_constraint(self, X, var_idx, bs1, r_bsp, u_pts, v_pts) -> None:
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

        # Compute cu, cv 
        cu, cv = self.d_c_uv(self.r_bs[2])

        # l = np.cross(cu, cv)
        # l = l/np.linalg.norm(l, axis=2)[:,:,None]

        # # Init X 
        # X[var_idx["l"]] = l.flatten()

        # Compute length of cu and cv
        self.cu_norm = np.linalg.norm(cu, axis=2)
        self.cv_norm = np.linalg.norm(cv, axis=2)

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

        # Add contraints
        self.add_constraint("E_cu", len(self.u_pts)*len(self.v_pts))
        self.add_constraint("E_cv", len(self.u_pts)*len(self.v_pts))




    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """
        
        # Get control points and line congruence
        cp, l = self.uncurry_X(X, var_idx, "rij", "l")
        
        # Reshape line congruence to shape (u, v, 3)
        l = l.reshape(len(self.u_pts), len(self.v_pts), 3)

        # Compute the derivatives of the mid mesh with respect to u and v
        cu, cv = self.d_c_uv(cp)

        #signs = - np.sign(np.sum(self.n*np.cross(cu, cv), axis=2))

        #cu_2 = np.linalg.norm(cu, axis=2)**2
        #cv_2 = np.linalg.norm(cv, axis=2)**2

        d_l_E_cu = (cu/self.cu_norm[:,:, None])
        #d_l_E_cu = (2*np.sum(l*cu, axis=2)/cu_2)[:,:,None]*cu 
        # d_l E_cu = cu 
        self.add_derivatives(self.const_idx["E_cu"].repeat(3),              # Rows
                            var_idx["l"],                                   # Cols
                            d_l_E_cu.flatten() # Vals
            )

        d_l_E_cv = (cv/self.cv_norm[:,:, None])
        #d_l_E_cv = (2*np.sum(l*cv, axis=2)/cv_2)[:,:,None]*cv
        # d_l E_cv = cv
        self.add_derivatives(self.const_idx["E_cv"].repeat(3), 
                            var_idx["l"], 
                            d_l_E_cv.flatten()
            )

        # = Derivatives with respec to the control points
        for k in range(len(cp)):

            d_ak_E_cu = np.sum(self.d_a_cu[k]*l, axis=2)/self.cu_norm
            d_ak_E_cv = np.sum(self.d_a_cv[k]*l, axis=2)/self.cv_norm

            #d_ak_E_cu = signs*(2*np.sum(self.d_a_cu[k]*l, axis=2)/cu_2)*(np.sum(l*cu, axis=2))
            #d_ak_E_cv = signs*(2*np.sum(self.d_a_cv[k]*l, axis=2)/cv_2)*(np.sum(l*cv, axis=2))

            # d_ak E_cu = l.(d_ak cu)/||cu||
            self.add_derivatives(self.const_idx["E_cu"], 
                                    var_idx["rij"][k].repeat(len(self.const_idx["E_cu"])), 
                                    d_ak_E_cu.flatten()
                                )
            
            
            # d_ak E_cv = l.(d_ak cv)/||cv||
            self.add_derivatives(self.const_idx["E_cv"],
                                    var_idx["rij"][k].repeat(len(self.const_idx["E_cv"])), 
                                    d_ak_E_cv.flatten()
                                )  
                                

        # r_cu = l.cu/||cu||^2 
        r_cu = (np.sum(l*cu, axis=2)/self.cu_norm).flatten()    
        #r_cu = (np.sum(l*cu, axis=2)**2/cu_2).flatten()
        self.set_r(self.const_idx["E_cu"], r_cu)

        # r_cv = l.cv/||cv||^2
        r_cv = (np.sum(l*cv, axis=2)/self.cv_norm).flatten()
        #r_cv = (np.sum(l*cv, axis=2)**2/cv_2).flatten()
        self.set_r(self.const_idx["E_cv"], r_cv)


        self.cu_norm = np.linalg.norm(cu, axis=2)
        self.cv_norm = np.linalg.norm(cv, axis=2)
    

    def d_c_uv(self, cp):
        """ Compute the cu, cv derivatives of the mid mesh
            Inputs:
                cp : Control points of the B-spline surface
            Outputs:
                cu : Derivative of the mid mesh with respect to u
                cv : Derivative of the mid mesh with respect to v
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




        
        

        
                

        


