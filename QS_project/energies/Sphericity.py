# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np


class Sphericity(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{supp} = \sum_{f\in Mesh Dual} \sum_{vi in f} || (vi - cf)^2 - rf^2 ||^2
        """
        super().__init__()
        self.name = "Sphericity" # Name of the constraint
        self.vertex_sph = None # Vertices per face
        self.row_vk = None # Row indices dvk
        self.row_cf = None # Row indices dcf
        self.row_rf = None # Row indices drf
        self.vk_idx = None # Vertex index
        self.cf_idx = None # Center index
        self.rf_idx = None # Radius index

      
    def initialize_constraint(self, X, var_idx, vertex_sph, aux) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            vertex_sph  : list of vertices per face
        """

        c = self.uncurry_X(X, var_idx, "c")

        # Define u_points and v_points
        self.vertex_sph = vertex_sph.flatten()

        # Row indices
        self.row_vk =  self.row_cf = np.repeat( np.arange(len(self.vertex_sph)), 3)
        self.row_rf =  np.arange(len(self.vertex_sph))

        # Col indices
        # v_k 
        fix_idx_vertex_sph = 3* np.repeat(self.vertex_sph, 3) + np.tile(range(3), len(self.vertex_sph))
        self.vk_idx = var_idx["v"][fix_idx_vertex_sph]

        # cf
        self.cf_idx = var_idx["c"].reshape(-1,3).repeat(4, axis=0).flatten()

        # rf
        self.rf_idx = var_idx["r"].repeat(4)

    
        # Number of constraints
        self.add_constraint("sphericity", len(self.vertex_sph))
        
        
    def compute(self, X, var_idx):
        # Unpack and reshape data
        vk = X[self.vk_idx]
        cf = X[self.cf_idx]
        rf = X[self.rf_idx]

        # D_vk E_sph = 2(v_k - c_f)
        d_vk_E_sph =  2*(vk[self.vk_idx] - cf[self.cf_idx])

        # D_cf E_sph = 2(c_f - v_k)
        d_cf_E_sph = -2*(vk[self.vk_idx] - cf[self.cf_idx])

        # D_rf E_sph = 2(r_f)
        d_rf_E_sph = - 2*rf

        # d_vk
        self.add_derivatives(self.row_vk, self.vk_idx, d_vk_E_sph)

        # d_cf
        self.add_derivatives(self.row_cf, self.cf_idx, d_cf_E_sph)

        # d_rf
        self.add_derivatives(self.row_rf, self.rf_idx, d_rf_E_sph)

        # residual
        vk = vk.reshape(-1,3)
        cf = cf.reshape(-1,3)

        self.set_r(self.const_idx["sphericity"], np.linalg.norm(vk - cf, axis=1)**2 - rf**2)

            
            