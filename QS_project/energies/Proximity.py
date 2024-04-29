# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np
import igl


class Proximity(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Proximity to a finer mesh (reference surface).
        E_{proximity} = \sum_{vi in Mesh} || vi - vref ||^2 + \sum_{vi in Mesh} || (vi - vref)nref ||^2
        """
        super().__init__()
        self.name = "Proximity" # Name of the constraint
        self.ref_v = None # Reference vertices
        self.ref_f = None # Reference faces
        self.nf = None # Normal of the reference faces

      
    def initialize_constraint(self, X, var_idx, ref_v, ref_f, epsilon) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            ref_v       : Reference vertices
            ref_f       : Reference faces
            epsilon     : Proximity distance

        """
        self.ref_f = ref_f
        self.ref_v = ref_v
        # # Get the closest points on the remeshed mesh
        # sd, l, cpts = igl.point_mesh_squared_distance(rV, dV, dF)
        

        self.epsilon = epsilon

        # Get vertices
        v = self.uncurry_X(X, var_idx, "v")

        # Distance Energy  E_D = epsilon(v - vf)
        self.add_constraint("E_D", len(v))

        v = v.reshape(-1, 3)

        # Tangential Energy E_T = (v - vf)nf
        self.add_constraint("E_T", len(v))


        # Get the closest points on the remeshed mesh
        _, _, vf = igl.point_mesh_squared_distance(v, self.ref_v, self.ref_f)

        v_vf = v - vf
        self.nf = v_vf/np.linalg.norm(v_vf, axis=1)[:, None]



        
    def compute(self, X, var_idx):
        
        # Get vertices
        v = self.uncurry_X(X, var_idx, var_name)

        v = v.reshape(-1, 3)

        # Get the closest points on the remeshed mesh
        _, _, vf = igl.point_mesh_squared_distance(v, self.ref_v, self.ref_f)

        v_vf = v - vf

        nf = self.nf

        # Distance Energy  E_D = epsilon(v - vf)
        dv_ED = (np.ones_like(v) * self.epsilon).flatten() 
        E_D = self.epsilon*(v_vf).flatten()
        self.add_derivatives(self.const_idx["E_D"], var_idx["v"], dv_ED)
        # res
        self.set_r(self.const_idx["E_D"], E_D)
        
        # Tangential Energy E_T = (v - vf)nf
        dv_ET = nf.flatten() 
        E_T = np.einsum('ij,ij->i', v_vf, nf).flatten()
        self.add_derivatives(self.const_idx["E_T"].repeat(3), var_idx["v"], dv_ET)
        # res
        self.set_r(self.const_idx["E_T"], E_T)

        nf  =  v_vf
        nf /= np.linalg.norm(nf, axis=1)[:, None]

        self.nf = nf
        








            
            