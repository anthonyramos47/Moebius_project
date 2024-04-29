# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np


class Supp(Constraint):

    def __init__(self) -> None:
        """ Template constraint
        Energy that focus on the planarity of the faces form by the centers of the sphere congruence.
        E_{supp} = \sum_{f\in Mesh Dual} \sum_{i in f} || (c_{i+1} - c_i)/||norm cii1|| *n_aux||^2
        """
        super().__init__()
        self.name = "Support" # Name of the constraint
        self.sph_sph_adj = None # Faces list
        self.edge_indices = None # Edges indices per face
        self.cij_norm = None # Norm of the vector c_{i+1} - c_i

      
    def initialize_constraint(self, X, var_idx, sph_sph_adj, aux) -> None:
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

        # Get centers of spheres
        c = self.uncurry_X(X, var_idx, "c")
        c = c.reshape(-1, 3)

        # Define u_points and v_points
        self.sph_sph_adj = sph_sph_adj

        self.edge_indices = []
        # Number of constraints
        const = 0
        for i, f_i in enumerate(sph_sph_adj):
            if len(f_i) < 3:
                print(f_i)
            else:
                spheres_number = len(f_i)
                const += spheres_number

            
        # Define length of the edges
        self.cij_norm = []
        for f_i in sph_sph_adj:
            spheres_number = len(f_i)

            if spheres_number < 3:
                print(f_i)
            else:
                edges_face = []

                cij_norm_f = []
                for i in range(spheres_number):
                    cij_norm_f.append(np.linalg.norm(c[f_i[(i+1)%spheres_number]] - c[f_i[i]]))
                    edges_face.append((f_i[i], f_i[(i+1)%spheres_number]))

                self.cij_norm.append(np.array(cij_norm_f))
                self.edge_indices.append(np.array(edges_face))

        self.add_constraint("supp", const)
        
    
    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Inputs:
            <Given by the optimizer>
                X: Variables
                var_idx: dictionary of indices of variables
        """ 

        # Get centers of spheres
        c, nd = self.uncurry_X(X, var_idx, "c", "nd")
        c = c.reshape(-1, 3)
        nd = nd.reshape(-1, 3)

        #rows = [np.hstack((np.arange(self.const).repeat(6), np.arange(self.const).repeat(3) ))]
        cols_ci = []
        cols_cj = []
        cols_nd = []
        values_ci = []
        values_cj = []
        values_nd = []

        res = []
        # Loop over faces
        for i, f_i in enumerate(self.sph_sph_adj):
            edges = self.edge_indices[i]

            #print(np.linalg.norm((c[edges[:, 1]] - c[edges[:, 0]])/self.cij_norm[i][:, None]))

            # Dot porduct of the normal and the edges
            #cij_nd = np.einsum('ij, ij->i', c[edges[:, 1]] - c[edges[:, 0]], nd[None, i])
            cij_nd = np.sum((c[edges[:, 1]] - c[edges[:, 0]])*nd[i], axis=1)
            cij_nd /= self.cij_norm[i]

            # Get edges index
            edges_idx_i = 3 * np.repeat(edges[:,0], 3) + np.tile(range(3), len(edges[:,0]))
            edges_idx_j = 3 * np.repeat(edges[:,1], 3) + np.tile(range(3), len(edges[:,1]))

            # Edges indices
            edg_idx_i = var_idx["c"][edges_idx_i]
            edg_idx_j = var_idx["c"][edges_idx_j]

            # Constraint
            cols_ci.extend(edg_idx_i)
            cols_cj.extend(edg_idx_j)

            nd_cjnorm = np.array([nd[i]]).repeat(len(edges), axis=0)/self.cij_norm[i][:,None]
            # d_c(i+1) E_supp_f = nd
            d_cj_E_f  =   nd_cjnorm 
            # d_c(i) E_supp_f = - nd
            d_ci_E_f  =  -nd_cjnorm

            # Values extend
            values_ci.extend(d_ci_E_f.flatten()) 
            values_cj.extend(d_cj_E_f.flatten())
            
            # DN E_supp_f
            # d_n E_supp_f = (c_{i+1} - c_i)/||c_{i+1} - c_i||
            d_n_E_f = (c[edges[:, 1]] - c[edges[:, 0]])/self.cij_norm[i][:, None]

            values_nd.extend(d_n_E_f.flatten())

            cols_nd.extend( var_idx["nd"][3*i: 3*i+3].repeat(len(edges), axis=0) )
            
            # Residual
            res.extend(cij_nd)
            self.cij_norm[i] = np.linalg.norm(c[edges[:, 1]] - c[edges[:, 0]], axis=1)

        self.add_derivatives(np.arange(self.const).repeat(3), cols_ci, values_ci)
        self.add_derivatives(np.arange(self.const).repeat(3), cols_cj, values_cj)
        self.add_derivatives(np.arange(self.const).repeat(3), cols_nd, values_nd)
        self.set_r(self.const_idx["supp"], res)



        
        

        
                

        


