# Call parent class
from optimization.constraint import Constraint
from scipy.interpolate import BSpline, bisplev, bisplrep
from geometry.utils import *
from utils.bsplines_functions import *
import numpy as np
from time import time

class BS_Torsal(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that minimize the angle between the line congruence l(u,v) and the surface normal n(u,v)
        Energy :  \sum_{pij \in Grid} || lt.nt/||lt|| ||^2  + || t.nt ||^2  +  || t^2 - 1 ||^2 + || lc/||lc||.nt ||^2
        where, 
            l .- Line congruence normalized
            t .- torsal direction t = ut du + vt dv
            nt .- torsal plane normal
            lc .- line congruence at baricenter
            lt .- line congruence direction lt = ut lu + vt lv
        """
        super().__init__()
        self.name = "BS_Torsal" # Name of the constraint
        self.du = None # Derivative of the grid in the u direction
        self.dv = None # Derivative of the grid in the v direction
        self.lt1_norm = None # Norm of the line congruence direction
        self.lt2_norm = None # Norm of the line congruence direction
        self.lc_norm = None # Norm of the line congruence at baricenter
        self.i0 = None # Index of the first variable
        self.i1 = None # Index of the second variable
        self.i2 = None # Index of the third variable
        self.i3 = None # Index of the fourth variable
        self.i0_flat = None # Index of the first variable in flat array
        self.i1_flat = None # Index of the second variable in flat array
        self.i2_flat = None # Index of the third variable in flat array
        self.i3_flat = None # Index of the fourth variable in flat array
        self.u_sample = None 
        self.v_sample = None
        self.counter = 0

        
      
    def initialize_constraint(self, X, var_idx, bsp, u_pts, v_pts, n, sample) -> None:
        """ 
        Input:
            X : Variables
            var_idx : dictionary of indices of variables
            bsp: B-spline surface
            sample : Sample size of the grid (n, m)
        """

        indices = np.array([  (sample[1]*j) + np.array([ i, i+1, sample[1] + i + 1, sample[1] + i])   for j in range(sample[1] - 1)for i in range(sample[0] - 1)] )

        self.u_sample = sample[0]
        self.v_sample = sample[1]

        # Assign the indices of the grid
        self.i0, self.i1, self.i2, self.i3 = indices[:,0], indices[:,1], indices[:,2], indices[:,3]

        # Assign the indices of the grid in flat array
        self.i0_flat  = flat_array_variables(self.i0)
        self.i1_flat  = flat_array_variables(self.i1)
        self.i2_flat  = flat_array_variables(self.i2)
        self.i3_flat  = flat_array_variables(self.i3)

        # Num of faces
        F = len(indices)

        # Evaluate B-spline surface at grid points
        s_uv = bsp(u_pts, v_pts).reshape(-1, 3)

        # Get grid points
        v0, v1, v2, v3 = s_uv[self.i0], s_uv[self.i1], s_uv[self.i2], s_uv[self.i3]

        # Compute the derivatives of the grid
        self.du = (v2 - v0)
        self.dv = (v1 - v3)

        # Compute normal of BSpline surface
        n = bsp.normal(u_pts, v_pts)

        # Initialize the torsal directions
        l = self.uncurry_X(X, var_idx, "l")
        l = l.reshape(sample[0], sample[1], 3)

        # Reorient l 
        sign = np.sign(np.einsum('ijk,ijk->ij', l, n))
        l = l*sign[:,:,None]
        
        # Compute the line congruence at the baricenter and the line congruence directions
        lc, lu, lv = lc_info_at_grid_points(l)

        # Reshape line congruence and normals
        lc = lc.reshape(-1, 3)
        lu = lu.reshape(-1, 3)
        lv = lv.reshape(-1, 3)

        # Normalize the line congruence
        lc /= np.linalg.norm(lc, axis=1)[:, None]

        # Compute the torsal directions 
        t1, t2, ut1, vt1, ut2, vt2, _ = torsal_directions(lc, lu, lv, self.du, self.dv)

        #lt1 = unit(ut1[:, None]*lu + vt1[:, None]*lv)
        #lt2 = unit(ut2[:, None]*lu + vt2[:, None]*lv)

        # Compute the torsal plane normal
        nt1 = unit(np.cross(lc, t1))
        nt2 = unit(np.cross(lc, t2))
    
        # Init the torsal directions
        X[var_idx["u1"]] = ut1 
        X[var_idx["v1"]] = vt1 
        X[var_idx["u2"]] = ut2 
        X[var_idx["v2"]] = vt2 

        # Init the torsal plane normals
        X[var_idx["nt1"]] = nt1.flatten()
        X[var_idx["nt2"]] = nt2.flatten()

        # Copute lines in torsal directions
        lt1 = ut1[:, None]*lu + vt1[:, None]*lv
        lt2 = ut2[:, None]*lu + vt2[:, None]*lv
    
        # Store the norm of the line congruence 
        self.lc_norm  = np.linalg.norm(lc, axis=1)
        self.lt1_norm = np.linalg.norm(lt1, axis=1)
        self.lt2_norm = np.linalg.norm(lt2, axis=1)

        # Add constraints
        self.add_constraint("lt_nt1", F)
        self.add_constraint("t_nt1", F)
        self.add_constraint("lt_nt2", F)
        self.add_constraint("t_nt2", F)
        self.add_constraint("lc_nt1", F)
        self.add_constraint("lc_nt2", F)
        self.add_constraint("t1_unit", F)
        self.add_constraint("t2_unit", F)


    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """
        self.counter += 1
        
        # Get variables
        l, u1, v1, u2, v2, nt1, nt2 = self.uncurry_X(X, var_idx, "l", "u1", "v1", "u2", "v2", "nt1", "nt2")

        # Reshape line congruence and normals
        nt1 = nt1.reshape(-1, 3)
        nt2 = nt2.reshape(-1, 3)
        l   = l.reshape(self.u_sample, self.v_sample, 3)

        # Compute the line congruence at the baricenter and the line congruence directions
        lc, lu, lv = lc_info_at_grid_points(l)

        lu = lu.reshape(-1, 3)
        lv = lv.reshape(-1, 3)
        lc = lc.reshape(-1, 3)

        t1 = u1[:, None]*self.du + v1[:, None]*self.dv
        t2 = u2[:, None]*self.du + v2[:, None]*self.dv

        # init_t = time()
        # Add E_t_unit E = || t^2 - 1 ||^2
        self.E_t_unit(t1, t2, var_idx)

        # Add E_t_nt E = || t.nt ||^2
        self.E_t_nt(nt1, nt2, t1, t2, var_idx)

        # Add E_lt_nt E = || lt/||lt||.nt ||^2
        self.E_lt_nt(lu, lv, nt1, nt2, u1, v1, u2, v2, var_idx)

        # Add E_lc_nt E = || lc/||lc||.nt ||^2
        self.E_lc_nt(lc, nt1, nt2, var_idx)

        # if self.counter % 25 == 0 and self.w > 0.5:
        #     self.w *= 0.6

        #final_t = time()
        #self.print_per_const_energy()
        #print("Time to compute BTorsal:", final_t - init_t)
 
    def E_t_unit(self, t1, t2, var_idx):
        """
        Compute energy J and residual for energy:
        E_t_unit = || t^2 - 1 ||^2
        Input: 
            u1, v1, u2, v2 : Torsal directions
            var_idx : Dictionary of indices of variables
        """

        # E_t_unit1 = || t1^2 - 1 ||^2
        cols = self.const_idx["t1_unit"]

        # d u1
        d_u1_E_t_unit1 = 2*(np.einsum('ij,ij->i', self.du, t1))
        self.add_derivatives(
            cols, 
            var_idx["u1"],
            d_u1_E_t_unit1
        )

        # d v1
        d_v1_E_t_unit1 = 2*(np.einsum('ij,ij->i', self.dv, t1))
        self.add_derivatives(
            cols, 
            var_idx["v1"],
            d_v1_E_t_unit1
        )

        # Set residual
        self.set_r(cols, np.einsum('ij,ij->i', t1, t1)-1)

        # E_t_unit2 = || t2^2 - 1 ||^2
        cols = self.const_idx["t2_unit"]

        # d u2
        d_u2_E_t_unit2 = 2*(np.einsum('ij,ij->i', self.du, t2))
        self.add_derivatives(
            cols, 
            var_idx["u2"],
            d_u2_E_t_unit2
        )

        # d v2
        d_v2_E_t_unit2 = 2*(np.einsum('ij,ij->i', self.dv, t2))
        self.add_derivatives(
            cols, 
            var_idx["v2"],
            d_v2_E_t_unit2
        )

        # Set residual
        self.set_r(cols, np.einsum('ij,ij->i', t2, t2) - 1)


    def E_t_nt(self, nt1, nt2, t1, t2, var_idx):
        """
        Compute energy J and residual for energy:
        E_t_nt = || t.nt ||^2
        Input: 
            t : Torsal direction
            nt1 : Torsal plane normal 1
            nt2 : Torsal plane normal 2
            u1, v1 : Torsal direction 1
            u2, v2 : Torsal direction 2
            var_idx : Dictionary of indices of variables
        """

        # E_t_nt1 = || t1.nt1 ||^2
        cols = self.const_idx["t_nt1"]

        # d u1
        d_u1_E_t_nt1 = (np.einsum('ij,ij->i', self.du, nt1))
        self.add_derivatives(
            cols, 
            var_idx["u1"],
            d_u1_E_t_nt1
        )

        # d v1
        d_v1_E_t_nt1 = (np.einsum('ij,ij->i', self.dv, nt1))
        self.add_derivatives(
            cols, 
            var_idx["v1"],
            d_v1_E_t_nt1
        )

        # d nt1 = t1 
        d_nt1_E_t_nt1 = (t1).flatten()
        self.add_derivatives(
            cols.repeat(3),
            var_idx["nt1"],
            d_nt1_E_t_nt1
        )

        #assert np.einsum('ij,ij->i', t1, nt1) == np.sum(t1*nt1, axis=1), "Error"
        # Set residual  
        self.set_r(cols, np.einsum('ij,ij->i', t1, nt1))

        # E_t_nt2 = || t2.nt2 ||^2
        cols = self.const_idx["t_nt2"]

        # d u2
        d_u2_E_t_nt2 = (vec_dot(self.du, nt2))
        self.add_derivatives(
            cols, 
            var_idx["u2"],
            d_u2_E_t_nt2
        )

        # d v2
        d_v2_E_t_nt2 = (vec_dot(self.dv, nt2))
        self.add_derivatives(
            cols, 
            var_idx["v2"],
            d_v2_E_t_nt2
        )

        # dnt2
        d_nt2_E_t_nt2 = (t2).flatten()
        self.add_derivatives(
            cols.repeat(3),
            var_idx["nt2"],
            d_nt2_E_t_nt2
        )

        # Set residual
        self.set_r(cols, np.einsum('ij,ij->i', t2, nt2))


    def E_lt_nt(self, lu, lv, nt1, nt2, u1, v1, u2, v2, var_idx):
        """
        Compute energy J and residual for energy:
        E_lt_nt = || lt/||lt||.nt ||^2
        Input: 
            lt : Line congruence direction
            nt : Torsal plane normal
            var_idx : Dictionary of indices of variables
        """

        lt1 = u1[:, None]*lu + v1[:, None]*lv
        lt2 = u2[:, None]*lu + v2[:, None]*lv

        # E_lt_nt1 = || lt1/||lt1||.nt1 ||^2; lt1 = u1 lu + v1 lv
        # Cols E_lt_nt1 
        cols = self.const_idx["lt_nt1"].repeat(3)
        lt1_norm = self.lt1_norm[:, None]

        # d u1 => (lu).nt1/||lt1||
        d_u1_E_lt_nt1 = (vec_dot(lu, nt1)/ self.lt1_norm)
        self.add_derivatives(
            self.const_idx["lt_nt1"], 
            var_idx["u1"],
            d_u1_E_lt_nt1
        )

        # d v1 => (lv).nt1/||lt1||
        d_v1_E_lt_nt1 = (vec_dot(lv, nt1)/self.lt1_norm)
        self.add_derivatives(
            self.const_idx["lt_nt1"], 
            var_idx["v1"],
            d_v1_E_lt_nt1
        )

        # lu = l2 - l0
        # lv = l1 - l3
        # d l2 = - d l0 = u1 nt1
        d_l0_E_lt_nt1 = ((u1[:, None]*nt1)/self.lt1_norm[:,None]).flatten()
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i0_flat],
            - d_l0_E_lt_nt1
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i2_flat],
             d_l0_E_lt_nt1
        )

        # d l1 = - d l3 = v1 nt1
        d_l1_E_lt_nt1 = ((v1[:, None]*nt1)/self.lt1_norm[:,None]).flatten()
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i1_flat],
            d_l1_E_lt_nt1
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i3_flat],
            -d_l1_E_lt_nt1
        )

        # d nt1 = lt1/||lt1||
        d_nt1_E_lt_nt1 = (lt1/lt1_norm).flatten()
        self.add_derivatives(
            cols,
            var_idx["nt1"],
            d_nt1_E_lt_nt1
        )

        # Set residual
        self.set_r(self.const_idx["lt_nt1"], np.einsum('ij,ij->i', lt1, nt1)/self.lt1_norm)

        # E_lt_nt2 = || lt2/||lt2||.nt2 ||^2
        # Cols E_lt_nt2
        cols = self.const_idx["lt_nt2"].repeat(3)

        # d u2
        d_u2_E_lt_nt2 = (np.einsum('ij,ij->i', lu, nt2)/self.lt2_norm)
        self.add_derivatives(
            self.const_idx["lt_nt2"], 
            var_idx["u2"],
            d_u2_E_lt_nt2
        )
        
        # d v2
        d_v2_E_lt_nt2 = (np.einsum('ij,ij->i', lv, nt2)/self.lt2_norm)
        self.add_derivatives(
            self.const_idx["lt_nt2"], 
            var_idx["v2"],
            d_v2_E_lt_nt2
        )

        
        # d l2 = - d l0 = u2 nt2
        d_l0_E_lt_nt2 = ((u2[:, None]*nt2)/self.lt2_norm[:,None]).flatten()
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i0_flat],
            - d_l0_E_lt_nt2
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i2_flat],
            d_l0_E_lt_nt2
        )

        # d l1 = - d l3
        d_l1_E_lt_nt2 = ((v2[:, None]*nt2)/self.lt2_norm[:,None]).flatten()
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i1_flat],
            d_l1_E_lt_nt2
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i3_flat],
            - d_l1_E_lt_nt2
        )

        # d nt2 = lt2/||lt2||
        d_nt2_E_lt_nt2 = (lt2/self.lt2_norm[:, None]).flatten()
        self.add_derivatives(
            cols,
            var_idx["nt2"],
            d_nt2_E_lt_nt2
        )

        # Set residual
        self.set_r(self.const_idx["lt_nt2"], np.einsum('ij,ij->i', lt2, nt2)/self.lt2_norm)

        # Update norms
        self.lt1_norm = np.linalg.norm(lt1, axis=1)
        self.lt2_norm = np.linalg.norm(lt2, axis=1)


    
    def E_lc_nt(self, lc, nt1, nt2, var_idx):
        """ 
        Compute energy J and residual for energy:
        E_lc_nt = || lc/||lc||.nt ||^2
        """

        lcnorm = self.lc_norm[:, None]

        # Cols E_lc_nt1
        cols = self.const_idx["lc_nt1"].repeat(3)
     
        # E lc_nt1 
        # dli E_lc_nt = nt/||lc||
        dli_E_lc_nt = (nt1/(4*lcnorm)).flatten()

        self.add_derivatives(
            cols, 
            var_idx["l"][self.i0_flat],
            dli_E_lc_nt
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i1_flat],
            dli_E_lc_nt
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i2_flat],
            dli_E_lc_nt
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i3_flat],
            dli_E_lc_nt
        )

        # d nt E_lc_nt = lc/||lc||
        d_nt_E_lc_nt = (lc/lcnorm).flatten()
        self.add_derivatives(
            cols, 
            var_idx["nt1"],
            d_nt_E_lc_nt
        )

        # Set residual
        self.set_r(self.const_idx["lc_nt1"], np.einsum('ij,ij->i', lc, nt1)/self.lc_norm)

        # E lc_nt2
        cols = self.const_idx["lc_nt2"].repeat(3)

        # dli E_lc_nt = nt/||lc||
        dli_E_lc_nt = (nt2/(4*lcnorm)).flatten()
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i0_flat],
            dli_E_lc_nt
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i1_flat],
            dli_E_lc_nt
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i2_flat],
            dli_E_lc_nt
        )
        self.add_derivatives(
            cols, 
            var_idx["l"][self.i3_flat],
            dli_E_lc_nt
        )

        # d nt E_lc_nt = lc/||lc||
        self.add_derivatives(
            cols, 
            var_idx["nt2"],
            d_nt_E_lc_nt
        )

        # Set residual
        self.set_r(self.const_idx["lc_nt2"], np.einsum('ij,ij->i', lc, nt2)/self.lc_norm)

        # Update lc norm 
        self.lc_norm = np.linalg.norm(lc, axis=1)
