# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np
import igl


class Proximity_C(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Proximity to a finer mesh (reference surface).
        E_{proximity} = \sum_{vi in Mesh} || vi - vref ||^2 + \sum_{vi in Mesh} || (vi - vref)nref ||^2
        """
        super().__init__()
        self.name = "Proximity_C" # Name of the constraint
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
        B, A = self.uncurry_X(X, var_idx, "B", "A")

        self.row_B = np.repeat(np.arange(len(B)), 3)
        self.row_A = np.arange(len(A))

        # Distance Energy  E_D = epsilon(v - vf)
        self.add_constraint("E_D", len(B))

        B = B.reshape(-1, 3)

        # Tangential Energy E_T = (v - vf)nf
        self.add_constraint("E_T", len(B))

        # Compute centers
        c = B/(2*A)[:, None]

        # Get the closest points on the remeshed mesh
        _, _, vf = igl.point_mesh_squared_distance(c, self.ref_v, self.ref_f)

        v_vf = c - vf
        self.nf = v_vf/np.linalg.norm(v_vf, axis=1)[:, None]



        
    def compute(self, X, var_idx):
        
        # Get vertices
        A, B = self.uncurry_X(X, var_idx, "A", "B")
        B = B.reshape(-1, 3)

        v = B/(2*A)[:, None]

        v = v.reshape(-1, 3)

        # Get the closest points on the remeshed mesh
        _, _, vf = igl.point_mesh_squared_distance(v, self.ref_v, self.ref_f)

        v_vf = v - vf

        nf = self.nf

        # Distance Energy  E_D = epsilon(v - vf); v = B/(2A) => B - 2A*vf = 0
        # dB = 1
        dB_ED = np.ones_like(B)*self.epsilon
        self.add_derivatives(self.const_idx["E_D"], var_idx["B"], dB_ED.flatten())
        # dA = - 2 vf
        dA_ED = - 2 * vf * self.epsilon
        self.add_derivatives(self.const_idx["E_D"], var_idx["A"].repeat(3), dA_ED.flatten())
        self.set_r(self.const_idx["E_D"], (v_vf).flatten()*self.epsilon)
    
        
        # Tangential Energy E_T = (v - vf)nf
        # dB = nf 
        dB_ET = (nf).flatten() 
        self.add_derivatives(self.const_idx["E_T"].repeat(3), var_idx["B"], dB_ET)
        # dA = - 2 vf.nf
        dA_ET = - 2*np.einsum('ij,ij->i', vf, nf)
        self.add_derivatives(self.const_idx["E_T"], var_idx["A"], dA_ET)
        
        # res
        res = np.einsum('ij,ij->i', (B - (2*A)[:,None]*vf), nf).flatten()
        self.set_r(self.const_idx["E_T"], res)

        nf  =  v_vf
        nf /= np.linalg.norm(nf, axis=1)[:, None]

        self.nf = nf
        








            
            