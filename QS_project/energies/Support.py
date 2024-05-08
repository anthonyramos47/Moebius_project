# Call parent class
from optimization.constraint import Constraint
import splipy as sp
import numpy as np
from geometry.utils import indices_flatten_dim


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


      
    def initialize_constraint(self, X, var_idx, sph_sph_adj, inn_v) -> None:
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

            
        self.nd_idx = var_idx["nd"][indices_flatten_dim(inn_v, n=3)]
        # Define length of the edges
        
        for f_i in sph_sph_adj:
            spheres_number = len(f_i)
            if spheres_number < 3:
                print(f_i)
            else:
                edges_face = []
                for i in range(spheres_number):                    
                    edges_face.append((f_i[i], f_i[(i+1)%spheres_number]))
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
        A, B= self.uncurry_X(X, var_idx, "A", "B")
        nd = X[self.nd_idx]
        B = B.reshape(-1, 3)
        nd = nd.reshape(-1, 3)

        #rows = [np.hstack((np.arange(self.const).repeat(6), np.arange(self.const).repeat(3) ))]
        cols_A1 = []
        cols_A2 = []
        cols_B1 = []
        cols_B2 = []
        cols_nd = []
        values_A1 = []
        values_A2 = []
        values_B1 = []
        values_B2 = []
        values_nd = []

        res = []
        # Loop over faces
        for i, _ in enumerate(self.sph_sph_adj):
            edges = self.edge_indices[i]

            B1 = B[edges[:, 0]]
            B2 = B[edges[:, 1]]

            A1 = A[edges[:, 0]]
            A2 = A[edges[:, 1]]

            # Get edges index for 3 dim case
            edges_idx_i = 3 * np.repeat(edges[:,0], 3) + np.tile(range(3), len(edges[:,0]))
            edges_idx_j = 3 * np.repeat(edges[:,1], 3) + np.tile(range(3), len(edges[:,1]))

            # Edges indices for A and B
            A_idx_1 = var_idx["A"][edges[:,0]]
            A_idx_2 = var_idx["A"][edges[:,1]]

            B_idx_1 = var_idx["B"][edges_idx_i]
            B_idx_2 = var_idx["B"][edges_idx_j]

            
            # E = (A1 B2 - A2 B1) nd
            # dA1 = B2 nd
            cols_A1.extend(A_idx_1)
            values_A1.extend(np.sum(B2*nd[i], axis=1))
            # dA2 = -B1 nd
            cols_A2.extend(A_idx_2)
            values_A2.extend(-np.sum(B1*nd[i], axis=1))
            # dB1 = - A2 nd
            cols_B1.extend(B_idx_1)
            
            ext_nd = np.array([nd[i]]).repeat(len(edges), axis=0)
            values_B1.extend((-A2[:,None]*ext_nd).flatten())
            # dB2 = A1 nd
            cols_B2.extend(B_idx_2)
            values_B2.extend( (A1[:,None]*ext_nd).flatten()) 

            # d_nd = (A1 B2 - A2 B1)
            d_nd = A1[:, None]*B2 - A2[:, None]*B1
            cols_nd.extend(np.tile(var_idx["nd"][3*i:3*(i+1)], len(edges)))
            values_nd.extend(d_nd.flatten())
            
            # Residual
            res.extend( np.sum(d_nd*nd[i], axis=1) )

        self.add_derivatives(np.arange(self.const), cols_A1, values_A1)
        self.add_derivatives(np.arange(self.const), cols_A2, values_A2)
        self.add_derivatives(np.arange(self.const).repeat(3), cols_B1, values_B1)
        self.add_derivatives(np.arange(self.const).repeat(3), cols_B2, values_B2)
        #self.add_derivatives(np.arange(self.const).repeat(3), cols_nd, values_nd)
        self.set_r(self.const_idx["supp"], res)



        
        

        
                

        


