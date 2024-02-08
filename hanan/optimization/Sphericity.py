# Planarity constraint implementation
import numpy as np
from hanan.optimization.constraint import Constraint


class Sphericity(Constraint):

    def __init__(self) -> None:
        """ Sphericity constraint
        Energy that penalizes the deviation of both envelopes from a sphere
        Energy : \sum_{f \in F} || \sum_{v \in V} [(c_f - v_i)^2 - r_f^2]  \sum_{j \in U} [(c_f - v_j)^2 - r_f^2] ||^2
        where   c_f is the center of the sphere, 
                r_f is the radius of the sphere, 
                V is the set of vertices of the face f, 
                U is the set of vertices of the second envelope face f
        """
        super().__init__()
        self.name = "Sphericty" # Name of the constraint
        self.faces = None # List of faces
        self.v = None # List of vertices

    def initialize_constraint(self, X, var_indices, F, V) -> None:
        # Input
        # X : Variables
        # var_indices: Dictionary with the indices of the variables
        # F: List of faces
        # V: List of vertices

        # Initialize variables and constraints
        self.var_idx = var_indices
        self.var = len(X)

        
        self.const_idx = {"Env1": np.arange(0         , 3*len(F)), # || (c_f - v_i)^2 - r_f ||^2 1st envelope
                          "Env2": np.arange(3*len(F)  , 6*len(F))  # || (c_f - v_i - ei)^2 - r_f ||^2 2st envelope
                          }
        self.const = 6*len(F) 
        
        # Set faces
        self.faces = F
        # Set vertices
        self.v = V



    def compute(self, X) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
        """

        # Get variables
        s_c, s_r, e = self.uncurry_X(X, "sph_c", "sph_r", "e")      

        s_c = s_c.reshape(-1, 3)
        e = e.reshape(-1, 3)

        # Get faces
        f = self.faces

        # Get vertices
        v = self.v

        # get i j k indices per face
        i, j, k = f[:,0], f[:,1], f[:,2]

        # Get second envelope vertices
        vv = v + e 

        # Set e indices
        i_e = self.var_idx["e"][3 * np.repeat(i, 3) + np.tile(range(3), len(i))]
        j_e = self.var_idx["e"][3 * np.repeat(j, 3) + np.tile(range(3), len(j))]
        k_e = self.var_idx["e"][3 * np.repeat(k, 3) + np.tile(range(3), len(k))]

        # Env1 
        cf_vi = np.vstack( (s_c - v[i], s_c - v[j], s_c - v[k] ))
        # Check lenght indices
       
        # d sph_c =>   2*(c_f - v_i)
        self.add_derivatives(self.const_idx["Env1"].repeat(3), np.tile(self.var_idx["sph_c"],3), 2*cf_vi.flatten())
        
        
        # d r =>  -2*r 
        self.add_derivatives(self.const_idx["Env1"], np.tile(self.var_idx["sph_r"],3), -2*np.tile(s_r,3) )

        
        # set r
        self.set_r(self.const_idx["Env1"], np.sum(cf_vi*cf_vi, axis=1) - np.tile(s_r*s_r, 3))

        
        # Env2
        cf_vvi = np.vstack( (s_c - vv[i], s_c - vv[j], s_c - vv[k] ))
        
        # d sph_c =>   2*(c_f - vv_i)
        self.add_derivatives(self.const_idx["Env2"].repeat(3), np.tile(self.var_idx["sph_c"],3), 2*cf_vvi.flatten())

        
        # d e_i,j,k =>  -(2*sph_c-2*v_i-2*e_i)
        self.add_derivatives(self.const_idx["Env2"].repeat(3), np.hstack((i_e, j_e, k_e)), -2*cf_vvi.flatten())

        # d r =>  -2*r
        self.add_derivatives(self.const_idx["Env2"], np.tile(self.var_idx["sph_r"],3), -2*np.tile(s_r,3))

        # set r
        self.set_r(self.const_idx["Env2"], np.sum(cf_vvi*cf_vvi, axis=1) - np.tile(s_r*s_r, 3))

        
                

        


