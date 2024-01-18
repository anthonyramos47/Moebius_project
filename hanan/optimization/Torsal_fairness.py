# Planarity constraint implementation

import numpy as np
from hanan.geometry.utils import vec_dot, solve_torsal, unit, barycentric_coordinates
from hanan.optimization.constraint import Constraint


class Torsal_Fair(Constraint):

    def __init__(self) -> None:
        """ 
        Initializes a new instance of the Torsal Fairness class.

        Torsal directions constraint energy, we assume nt are unitary vectors.
        E = \sum{eij \in InnerEdges} 
            || b11 vix  + b12 vjx + b13 vkx - vlx ||^2 + || b11 viy + b12 vjy + b13 vky - vly ||^2 + || b11 viz  + b12 vjz + b13 vkz - vlz ||^2 
            || b21 vvix + b22 vvjx + b23 vvkx - vvlx ||^2 + || b21 vviy + b22 vvjy + b23 vvky - vvly ||^2 + || b21 vviz + b22 vvjz + b23 vvkz - vvlz ||^2
            || b11 - b21 ||^2 + || b12 - b22 ||^2 + || b13 - b23 ||^2;
        
        vvi = vi + 2*(dcv.ei) ei; dcv = cf - vi; cf = bf + df ncf

        where, 
            vi    .- vertices in the triangular mesh
            vvi   .- vertices on the second envelope
            b1i   .- barycentric coordinates vl = b11 vi + b12 vj + b13 vk
            b2i   .- barycentric coordinates vvl = b21 vvi + b22 vvj + b23 vvk
        """
        super().__init__()
        self.inner_edges = None # List of inner edges
        self.inner_edges_opposite_vertices = None # Vertices indices of the opposite vertices of the inner edges 
        self.inner_edges_vertices = None # Vertices indices of the inner edges
        self.inner_edges_faces = None # Faces indices of the inner edges
        
        self.ncf = None # Normals of the circumcenters
        self.bf = None # Circumcenters
        self.v = None # Set of vertices




    
    def initialize_constraint(self, X, var_indices, V, F, inner_edges, ie_f, iv1, iv2, v_if, bf, ncf) -> np.array:
        
        # Set vertices
        self.v = V

        # Set inner edges
        self.inner_edges = inner_edges

        # Set map of edges to vertices
        self.inner_edges_opposite_vertices = v_if  

        # Set vertices of the inner edges
        self.inner_edges_vertices = np.array([iv1, iv2]).T

        # Set faces of the inner edges
        self.inner_edges_faces = ie_f

        # Set circumcenters
        self.bf = bf

        # Set circumcenters axis
        self.ncf = ncf

        # Number of variables
        self.var = len(X)

        # Get df 
        df = X[var_indices["df"]]

        # Get e line congruence
        e = X[var_indices["e"]].reshape(-1, 3)

        # Compute second envelope
        vv1, vv2, vv3, _, _, _ = self.compute_second_env(df, e, F)

        vvf = np.array([vv1, vv2, vv3]).T

        b1 = np.zeros(3*len(self.inner_edges))
        # Compute initial barycentric coordinates
        for i in range(len(self.inner_edges)):
            idx = self.inner_edges[i]

            # Get vertices of the edge
            ivi, ivj = self.inner_edges_vertices[idx]

            # Get vertices of the opposite edge
            ivk, ivl = self.inner_edges_opposite_vertices[idx]

            # Get vertices of the faces
            vi, vj, vk, vl = V[iv1], V[iv2], V[ivk], V[ivl]

            # Get baricentric coordinates
            b1[3*i:3*i+3] = barycentric_coordinates(vi, vj, vk, vl)

            # Second envelope 
             


    def compute(self, X) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                F: Faces
        """
        e, a1, b1, nt1, a2, b2, nt2, df = self.uncurry_X(X, "e", "a1", "b1", "nt1", "a2", "b2", "nt2", "df")

        e   = e.reshape(-1, 3)
        nt1 = nt1.reshape(-1, 3)
        nt2 = nt2.reshape(-1, 3)

        
        # Get vertices of second envelope
        vvi, vvj, vvk, dcvi, dcvj, dcvk = self.compute_second_env(df, e, self.F)

        # Compute ec
        ec = self.compute_ec(df, e, self.F)
              

        # Get vij, vik
        vij = self.fvij[:,0]
        vik = self.fvij[:,1]

        print("Sumary Torsal energies\n")
        # Init indices for sparse J matrix
        self.fill_J_t(e, ec, a1, b1, nt1, vij, vik, vvi, vvj, vvk, dcvi, dcvj, dcvk, self.F, 1)

        self.fill_J_t(e, ec, a2, b2, nt2, vij, vik, vvi, vvj, vvk, dcvi, dcvj, dcvk, self.F, 2)

        # Compute t
        # t1 = self.compute_t(a1, b1)
        # t2 = self.compute_t(a2, b2)

        # # Add energies related to t1,t2 angle
        # # t1.t2g = || t1.t2 - g^2 ||;  t1 = a1 vij + b1 vik, t2 = a2 vij + b2 vik =>  t1, t2 same side
        # # self.add_derivatives(self.const_idx["t1.t2g"], self.var_idx["a1"], w*vec_dot(vij, t2))
        # # self.add_derivatives(self.const_idx["t1.t2g"], self.var_idx["b1"], w*vec_dot(vik, t2))
        # # self.add_derivatives(self.const_idx["t1.t2g"], self.var_idx["a2"], w*vec_dot(vij, t1))
        # # self.add_derivatives(self.const_idx["t1.t2g"], self.var_idx["b2"], w*vec_dot(vik, t1))
        # # self.add_derivatives(self.const_idx["t1.t2g"], self.var_idx["g"] , -w*2*g )     
        # # self.set_r(self.const_idx["t1.t2g"], w*(vec_dot(t1, t2) - g**2))

        # # t1.t2l = || t1.t2^2 - 1 + l^2 ||;  t1 = a1 vij + b1 vik, t2 = a2 vij + b2 vik =>  t1, t2 not parallel
        # self.add_derivatives(self.const_idx["t1.t2l"], self.var_idx["a1"], w*2*vec_dot(t1,t2)*vec_dot(vij, t2))
        # self.add_derivatives(self.const_idx["t1.t2l"], self.var_idx["b1"], w*2*vec_dot(t1,t2)*vec_dot(vik, t2))
        # self.add_derivatives(self.const_idx["t1.t2l"], self.var_idx["a2"], w*2*vec_dot(t1,t2)*vec_dot(vij, t1))
        # self.add_derivatives(self.const_idx["t1.t2l"], self.var_idx["b2"], w*2*vec_dot(t1,t2)*vec_dot(vik, t1))
        # self.add_derivatives(self.const_idx["t1.t2l"], self.var_idx["l"] , w*2*l )
        # self.set_r(self.const_idx["t1.t2l"], w*(vec_dot(t1, t2)**2 -1 + l**2))

        self.ecnorms = np.linalg.norm(ec, axis=1)
        


    def compute_second_env(self, df, ei, F):
        """ Compute the second envelope of the mesh
        Input:
            df .- distance to center np.array
            vi .- Vertices np.array
            ei .- direction vectors np.array
        Output:
            vvi, vvj, vvk, dcvi, dcvj, dcvk
        """

        cf = self.compute_sphere_centers(df)

        # Get vertices of the faces
        vi, vj, vk = self.v 
        
        # Direction from vi to cf
        dcvi = cf - vi
        # Direction from vj to cf
        dcvj = cf - vj
        # Direction from vk to cf
        dcvk = cf - vk

        ue = unit(ei)

        # compute the second envelope as vi + 2*(dcv.ei) uei; uei = ei/||ei||
        vvi = 2*vec_dot(dcvi,ue[F[:,0]])[:, None] * ue[F[:,0]] + vi
        vvj = 2*vec_dot(dcvj,ue[F[:,1]])[:, None] * ue[F[:,1]] + vj
        vvk = 2*vec_dot(dcvk,ue[F[:,2]])[:, None] * ue[F[:,2]] + vk

        return vvi, vvj, vvk, dcvi, dcvj, dcvk


    def compute_sphere_centers(self, df) ->  np.array:
        """ Compute sphere centers
        Input:
            bf .- Circumcenters
            df .- Distance to center
            ncf .- Normal of circumcenters
        """
        return self.bf + df[:, None] * self.ncf
    
    def compute_ec(self, df, e_i, F) -> np.array:
        """ Compute the direction of the line congruence at the barycenters
        Input:
            df .- Distance to center
            e_i .- direction vectors np.array
            F.- Faces
        Output:
            ec .- direction of the line congruence at the barycenters
        """
        # Get first envelope
        vc = self.vc

        # Compute second envelope
        vvi, vvj, vvk, _, _, _ = self.compute_second_env(df, e_i, F)

        # Compute varycenter in second envelope
        vvc = (vvi + vvj + vvk)/3

        # Compute ec as differencen between second and first envelope
        ec = vvc - vc

        if len(ec.shape) == 1:
            ec = np.array([ec])

        return ec


    
    def compute_decnt(self, dcvi, dcvj, dcvk, e, nt, F):
        """ Function to compute the derivative of nt.ec with respect to e_i
            (nt.ec) = (nt.(ei + ej + ek)/3) = (nt).(vvc - vc);
            vc = (vi + vj + vk)/3
            vvi = vi + 2*(dcv.ei) ei
        """
        # Get directions per vertex
        ei = e[F[:,0]]
        ej = e[F[:,1]]
        ek = e[F[:,2]]  

        

        # Compute derivatives of de (nt.ec)
        deix = vec_dot( ( dcvi[:,0][:, None]*ei +  vec_dot(ei, dcvi)[:, None]*np.array([1,0,0]) ), nt)
        deiy = vec_dot( ( dcvi[:,1][:, None]*ei +  vec_dot(ei, dcvi)[:, None]*np.array([0,1,0]) ), nt)
        deiz = vec_dot( ( dcvi[:,2][:, None]*ei +  vec_dot(ei, dcvi)[:, None]*np.array([0,0,1]) ), nt)

        dejx = vec_dot( ( dcvj[:,0][:, None]*ej +  vec_dot(ej, dcvj)[:, None]*np.array([1,0,0]) ), nt)
        dejy = vec_dot( ( dcvj[:,1][:, None]*ej +  vec_dot(ej, dcvj)[:, None]*np.array([0,1,0]) ), nt)
        dejz = vec_dot( ( dcvj[:,2][:, None]*ej +  vec_dot(ej, dcvj)[:, None]*np.array([0,0,1]) ), nt)

        dekx = vec_dot( ( dcvk[:,0][:, None]*ek +  vec_dot(ek, dcvk)[:, None]*np.array([1,0,0]) ), nt)
        deky = vec_dot( ( dcvk[:,1][:, None]*ek +  vec_dot(ek, dcvk)[:, None]*np.array([0,1,0]) ), nt)
        dekz = vec_dot( ( dcvk[:,2][:, None]*ek +  vec_dot(ek, dcvk)[:, None]*np.array([0,0,1]) ), nt)

        return deix, deiy, deiz, dejx, dejy, dejz, dekx, deky, dekz


    def compute_d_df(self, e, nt, F):
        """ Compute the derivative of nt.ec with respect to df
            (nt.ec) = (nt.(ei + ej + ek)/3) = (nt).(vvc - vc);
            vc = (vi + vj + vk)/3
            vvi = vi + 2(dcv.ei)/||enor|| ei/||enor||; dcv = cf - vi 
            cf = bf + df ncf
            => d df (nt.ec) = 2/3 [nt.( (ei.ncf)* ei + (ej.ncf)*ej   + (ek.ncf)*ek )]/||enor||**2
        """
        # Get directions per vertex
        ei = e[F[:,0]]
        ej = e[F[:,1]]
        ek = e[F[:,2]] 

        # Get normals of the circumcenters
        ncf = self.ncf 

        # Compute dot produts
        eicf = vec_dot(ei, ncf)
        ejcf = vec_dot(ej, ncf)
        ekcf = vec_dot(ek, ncf)

        eicf_ei = eicf[:,None]*ei
        ejcf_ej = ejcf[:,None]*ej
        ekcf_ek = ekcf[:,None]*ek

        # Compute the derivative of nt.ec with respect to df
        d_df = 2/3*vec_dot( (eicf_ei + ejcf_ej + ekcf_ek ), nt)

        return d_df, eicf_ei, ejcf_ej, ekcf_ek   


    
    def compute_t(self, a, b):
        """ Compute t = a vij + b vik
        """
        return a[:, None]*self.fvij[:,0] + b[:, None]*self.fvij[:,1]

    def compute_tt(self, a, b, vvi, vvj, vvk):
        """ Compute tt = a vvij + b vvik
        """
        vvij = vvj - vvi
        vvik = vvk - vvi
        return a[:, None]*vvij + b[:, None]*vvik, vvij, vvik
    
       
 
    






        
       