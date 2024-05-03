# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np
from geometry.utils import indices_flatten_dim


class Tor_Planarity(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that approximation of the sphere to the vertices of the mesh.
        E_{to_planarity} = \sum_{e_wu \in E} ||  (w - u).n_aux ||^2 + ||  (lw).n_aux ||^2  + ||  (lu).n_aux ||^2
        """
        super().__init__()
        self.name = "Torsal Planarity" # Name of the constraint
        self.edg_l = None # Edges length
        self.e_uw  = None # Edge w-u length
        self.u_idx = None # Indices of vector u
        self.w_idx = None # Indices of vector w
        self.lw_idx = None # Indices of vector lw
        self.lu_idx = None # Indices of vector lu
        


      
    def initialize_constraint(self, X, var_idx, edges_idx, aux) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx     : dictionary of indices of variables
            edges_idx   : list of edges indices
        """

        vertices = X[var_idx["v"]].reshape(-1, 3)

        # Compute the edges w-u
        self.e_uw  = np.linalg.norm(vertices[edges_idx[1]] - vertices[edges_idx[0]], axis=1)
        #self.e_uw /= np.linalg.norm(self.e_uw, axis=1)[:,None]

        self.u_idx = var_idx["v"][indices_flatten_dim(edges_idx[0], n=3)]
        self.w_idx = var_idx["v"][indices_flatten_dim(edges_idx[1], n=3)]


        # Define indices of line congruent to the vertices of the edges
        self.lu_idx = var_idx["nd"][indices_flatten_dim(edges_idx[0], n=3)]
        self.lw_idx = var_idx["nd"][indices_flatten_dim(edges_idx[1], n=3)]

    
        # Add constraints
        # E_e = ||  (w - u).n_aux ||^2 
        self.add_constraint("Ee", len(edges_idx[0]))

        # E_lw = ||  (lw).n_aux ||^2
        self.add_constraint("Ew", len(edges_idx[0]))

        # E_lu = ||  (lu).n_aux ||^2
        self.add_constraint("Eu", len(edges_idx[0]))

        # Define row indices
        self.row_Ee = self.const_idx["Ee"].repeat(3)
        self.row_Ew = self.const_idx["Ew"].repeat(3)
        self.row_Eu = self.const_idx["Eu"].repeat(3)

        lw = X[self.lw_idx].reshape(-1, 3)
        lu = X[self.lu_idx].reshape(-1, 3)
        w_u = vertices[edges_idx[1]] - vertices[edges_idx[0]]
        w_u /= np.linalg.norm(w_u, axis=1)[:,None]

        # Init normals aux
        n_aux = np.cross(w_u, lu)
        n_aux /= np.linalg.norm(n_aux, axis=1)[: , None]

        print( np.linalg.norm(np.einsum('ij,ij->i',n_aux, w_u) ) ) 
        print( np.linalg.norm(np.einsum('ij,ij->i',n_aux, lw) ) )
        print( np.linalg.norm(np.einsum('ij,ij->i',n_aux, lu) ) )

        X[var_idx["n_l"]] = n_aux.flatten()

        
        
        
    def compute(self, X, var_idx):
        """ Compute the constraint
        """
        # Get the variables
        u  = X[self.u_idx]
        w  = X[self.w_idx]
        lw = X[self.lw_idx]
        lu = X[self.lu_idx]
        n  = X[var_idx["n_l"]]

        # Ee = ||  (w - u).n_aux ||^2
        # d_n = (w - u)
        w_u  = w.reshape(-1,3) - u.reshape(-1, 3)
        d_n = (w_u/self.e_uw[:,None]).flatten()
        self.add_derivatives(self.row_Ee, var_idx["n_l"], d_n)
        # d_w = n
        self.add_derivatives(self.row_Ee, self.w_idx, n)
        # d_u = -n
        self.add_derivatives(self.row_Ee, self.u_idx, -n)

        # Ew = ||  (lw).n_aux ||^2
        # d_n = lw
        self.add_derivatives(self.row_Ew, var_idx["n_l"], lw)

        # d_lw = n
        self.add_derivatives(self.row_Ew, self.lw_idx, n)

        # Eu = ||  (lu).n_aux ||^2
        # d_n = lu
        self.add_derivatives(self.row_Eu, var_idx["n_l"], lu)

        # d_lu = n
        self.add_derivatives(self.row_Eu, self.lu_idx, n)

        # Residuals
        n  = n.reshape(-1, 3)
        lw = lw.reshape(-1, 3)
        lu = lu.reshape(-1, 3)
        
        res_Ee = np.einsum('ij,ij->i', w_u, n)/self.e_uw
        res_Ew = np.einsum('ij,ij->i', lw, n)
        res_Eu = np.einsum('ij,ij->i', lu, n)

        self.set_r(self.const_idx["Ee"], res_Ee)
        self.set_r(self.const_idx["Ew"], res_Ew)
        self.set_r(self.const_idx["Eu"], res_Eu)

        self.e_uw = np.linalg.norm( w_u, axis=1)




            
            