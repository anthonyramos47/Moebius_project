# Planarity constraint implementation

import numpy as np
from hanan.geometry.utils import vec_dot, solve_torsal, unit, barycentric_coordinates_app, orth_proj
from hanan.optimization.constraint import Constraint


class Torsal_Fair(Constraint):

    def __init__(self) -> None:
        """ 
        Initializes a new instance of the Torsal Fairness class.

        Torsal directions constraint energy, we assume nt are unitary vectors.
        E = \sum{eij \in InnerEdges} 
            || vl - sum(ui vi) - ( vl.l/l^2 - sum(ui vi.l/l^2 ))l ||^2 + 
            || el - sum(ui ei) - ( el.l/l^2 - sum(ui ei.l/l^2 ))l ||^2 + 
    
        where, 
            InnerEdges are the edges that are not on the boundary of the mesh
            vl is the opposite vertex of the edge eij
            el is the middle edge of the edge eij
            ui are the barycentric coordinates of the vertices of the edge eij
            vi are the vertices of the edge eij
            ei are the edges of the edge eij
            l is the middle edge of the edge eij
        """


        super().__init__()
        self.name = "Torsal_Fair" 
        self.inner_edges = None # List of inner edges
        self.inner_edges_opposite_vertices = None # Vertices indices of the opposite vertices of the inner edges 
        self.inner_edges_vertices = None # Vertices indices of the inner edges
        self.L_norm = None # Norm of middle edge ei+ej
        
        self.v = None # Set of vertices

    
    def initialize_constraint(self, X, var_indices, v, inner_edges, ie_i, ie_j, ie_k, ie_l) -> np.array:
        # Set vertices
        self.v = v

        e = X[var_indices["e"]].reshape(-1, 3)

        # Set inner edges
        self.inner_edges = inner_edges

        # Set map of edges to vertices
        self.inner_edges_opposite_vertices = np.array([ie_k, ie_l]).T  

        # Set vertices of the inner edges
        self.inner_edges_vertices = np.array([ie_i, ie_j]).T 

        # Number of variables
        self.var = len(X)

        # Set indices of the variables
        var_idx = var_indices

        # Vertices
        vi, vj, vk, vl = v[ie_i], v[ie_j], v[ie_k], v[ie_l]

        # Line congruence middle edge
        L = e[ie_i] + e[ie_j] 

        # Norm of the middle edge
        self.L_norm = np.linalg.norm(L, axis=1)

        vbi = orth_proj(vi, L)
        vbj = orth_proj(vj, L)
        vbk = orth_proj(vk, L)
        vbl = orth_proj(vl, L)

        # Compute the barycentric coordinates of the vertices 
        for i in range(len(inner_edges)):
            
            u1, u2, u3 = barycentric_coordinates_app(vbi[i], vbj[i], vbk[i], vbl[i])
            
            X[var_idx["u"][3*i]   ] = u1 
            X[var_idx["u"][3*i+1] ] = u2 
            X[var_idx["u"][3*i+2] ] = u3 
            # X[var_idx["u"][3*i]   ] = 1/3
            # X[var_idx["u"][3*i+1] ] = 1/3
            # X[var_idx["u"][3*i+2] ] = 1/3
        
        self.add_constraint("T1", 3*len(inner_edges))
        self.add_constraint("T2", 3*len(inner_edges))
        
            
    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                F: Faces
        """

        # Get vertices
        v = self.v

        # Get barycentric coordinates
        u = X[var_idx["u"]].reshape(-1, 3)

        # Get Lines
        e = X[var_idx["e"]].reshape(-1, 3)

        # Get inner edges
        inner_edges = self.inner_edges
        # Number of inner edges
        n_ie= len(inner_edges)

        # Get map of vertices
        i, j = self.inner_edges_vertices.T

        # Get map of oposite vertices
        k, l = self.inner_edges_opposite_vertices.T

        # v indices 
        vi, vj, vk, vl = v[i], v[j], v[k], v[l]

        # Get norm of the middle edge
        L_2 = self.L_norm**2

        # Get middle edge
        L = e[i] + e[j]

        # Get e indices
        # e_i
        e_idx_i_x = var_idx["e"][3*i]
        e_idx_i_y = var_idx["e"][3*i + 1] 
        e_idx_i_z = var_idx["e"][3*i + 2] 
        # e_j
        e_idx_j_x = var_idx["e"][3*j]
        e_idx_j_y = var_idx["e"][3*j + 1] 
        e_idx_j_z = var_idx["e"][3*j + 2] 
        # e_k
        e_idx_k_x = var_idx["e"][3*k]
        e_idx_k_y = var_idx["e"][3*k + 1] 
        e_idx_k_z = var_idx["e"][3*k + 2] 
        # e_l
        e_idx_l_x = var_idx["e"][3*l]
        e_idx_l_y = var_idx["e"][3*l + 1]
        e_idx_l_z = var_idx["e"][3*l + 2]

        # Dot product vi L
        viL = vec_dot(vi, L)
        vjL = vec_dot(vj, L)
        vkL = vec_dot(vk, L)
        vlL = vec_dot(vl, L)

        # Get barycentric coordinates
        ui = u[:,0]
        uj = u[:,1]
        uk = u[:,2]
        
        # L/L^2
        Ln = L/L_2[:,None]

        # ux indices
        ux = var_idx["u"][3*np.arange(0, n_ie)].repeat(3)
        uy = var_idx["u"][3*np.arange(0, n_ie) + 1].repeat(3)
        uz = var_idx["u"][3*np.arange(0, n_ie) + 2].repeat(3)
        
        # Dui => -vi + vi.l/l^2 l
        self.add_derivatives(self.const_idx["T1"], ux, (-v[i] + (viL/L_2)[:,None]*L).flatten())
        # Duj => -vj + vj.l/l^2 l
        self.add_derivatives(self.const_idx["T1"], uy, (-v[j] + (vjL/L_2)[:,None]*L).flatten())
        # Duk => -vk - vk.l/l^2 l
        self.add_derivatives(self.const_idx["T1"], uz, (-v[k] + (vkL/L_2)[:,None]*L).flatten())

        # # Projection
        proj = (vec_dot(vl, L) - ui*viL - uj*vjL - uk*vkL)/L_2
        
     

        # Proj dx 
        proj_dx = (-(vl[:,0]  - ui*vi[:,0] - uj*vj[:,0] - uk*vk[:,0]))[:,None]*Ln - proj[:,None]*np.array([1,0,0])
        # Proj dy
        proj_dy = (-(vl[:,1]  - ui*vi[:,1] - uj*vj[:,1] - uk*vk[:,1]))[:,None]*Ln - proj[:,None]*np.array([0,1,0])
        # Proj dz
        proj_dz = (-(vl[:,2]  - ui*vi[:,2] - uj*vj[:,2] - uk*vk[:,2]))[:,None]*Ln - proj[:,None]*np.array([0,0,1])

        # Deix => - (vl[0]/l^2  - sum(ui vi[0]/l^2))l - (vl.l/l^2 - sum(ui vi.l/l^2 ))[1,0,0]
        self.add_derivatives(self.const_idx["T1"], e_idx_i_x.repeat(3), proj_dx.flatten())
        # Deiy => - (vl[1]/l^2  - sum(ui vi[1]/l^2))l - (vl.l/l^2 - sum(ui vi.l/l^2 ))[0,1,0]
        self.add_derivatives(self.const_idx["T1"], e_idx_i_y.repeat(3), proj_dy.flatten())
        # Deiz => - (vl[2]/l^2  - sum(ui vi[2]/l^2))l - (vl.l/l^2 - sum(ui vi.l/l^2 ))[0,0,1]
        self.add_derivatives(self.const_idx["T1"], e_idx_i_z.repeat(3), proj_dz.flatten())

        # Dejx => - (vl[0]/l^2  - sum(ui vj[0]/l^2))l - (vl.l/l^2 - sum(ui vj.l/l^2 ))[1,0,0]
        self.add_derivatives(self.const_idx["T1"], e_idx_j_x.repeat(3), proj_dx.flatten())
        # Dejy => - (vl[1]/l^2  - sum(ui vj[1]/l^2))l - (vl.l/l^2 - sum(ui vj.l/l^2 ))[0,1,0]
        self.add_derivatives(self.const_idx["T1"], e_idx_j_y.repeat(3), proj_dy.flatten())
        # Dejz => - (vl[2]/l^2  - sum(ui vj[2]/l^2))l - (vl.l/l^2 - sum(ui vj.l/l^2 ))[0,0,1]
        self.add_derivatives(self.const_idx["T1"], e_idx_j_z.repeat(3), proj_dz.flatten())
        
        self.set_r(self.const_idx["T1"], (vl - ui[:,None]*vi - uj[:,None]*vj - uk[:,None]*vk - proj[:,None]*L).flatten())

        #print("E T1:", self.r[self.const_idx["T1"]]@self.r[self.const_idx["T1"]])

        # Energy T2  = || vl - sum(ui vi) - ( vl.l/l^2 - sum(ui vi.l/l^2 ))l + el - sum(ui ei) - ( el.l/l^2 - sum(ui ei.l/l^2 ))l ||^2
        #            = || vvl - sum(ui vvi) - ( vvl.l/l^2 - sum(ui vvi.l/l^2 ))l ||^2                      

        # Get ei,j,k,l
        ei = e[i]
        ej = e[j]
        ek = e[k]
        el = e[l]

        vvi = vi + ei 
        vvj = vj + ej
        vvk = vk + ek
        vvl = vl + el

        vviL = vec_dot(vvi, L)
        vvjL = vec_dot(vvj, L)
        vvkL = vec_dot(vvk, L)
        vvlL = vec_dot(vvl, L)

        # ei.l 
        # eiL = vec_dot(ei, L)
        # ejL = vec_dot(ej, L)
        # ekL = vec_dot(ek, L)
        # elL = vec_dot(el, L)

        # Dui => - vvi + vvi.l/l^2 l
        self.add_derivatives(self.const_idx["T2"], 
                             ux,
                             (-vvi + (vviL/L_2)[:,None]*L).flatten())
        # Duj => - ej + ej.l/l^2 l
        self.add_derivatives(self.const_idx["T2"], 
                             uy, 
                             (-vvj + (vvjL/L_2)[:,None]*L).flatten())
        # Duk => - ek - ek.l/l^2 l
        self.add_derivatives(self.const_idx["T2"], 
                             uz, 
                             (-vvk + (vvkL/L_2)[:,None]*L).flatten())

        # # # Projection ( vl.l/l^2 - sum(ui vi.l/l^2 ))l
        proj = (vvlL - ui*vviL - uj*vvjL - uk*vvkL)/L_2


        # proj = (  vvl.l/l^2 - sum(ui vvi.l/l^2 ))l ; l = ei + ej => dl_ei = 1, dl_ej = 1 
        # dei_x vvi = vi + ei => 
        # Proj dx
        proj_ei_dx = ((vvl[:,0] - ui*vvi[:,0] - uj*vvj[:,0] - uk*vvk[:,0] - ui*L[:,0]))[:,None]*Ln + proj[:,None]*np.array([1,0,0])

        # Proj dy
        proj_ei_dy = ((vvl[:,1] - ui*vvi[:,1] - uj*vvj[:,1] - uk*vvk[:,1] - ui*L[:,1]))[:,None]*Ln + proj[:,None]*np.array([0,1,0])

        # Proj dz
        proj_ei_dz = ((vvl[:,2] - ui*vvi[:,2] - uj*vvj[:,2] - uk*vvk[:,2] - ui*L[:,2]))[:,None]*Ln + proj[:,None]*np.array([0,0,1])

        # Proj dx
        proj_ej_dx = ((vvl[:,0] - ui*vvi[:,0] - uj*vvj[:,0] - uk*vvk[:,0] - uj*L[:,0]))[:,None]*Ln + proj[:,None]*np.array([1,0,0])

        # Proj dy
        proj_ej_dy = ((vvl[:,1] - ui*vvi[:,1] - uj*vvj[:,1] - uk*vvk[:,1] - uj*L[:,1]))[:,None]*Ln + proj[:,None]*np.array([0,1,0])

        # Proj dz
        proj_ej_dz = ((vvl[:,2] - ui*vvi[:,2] - uj*vvj[:,2] - uk*vvk[:,2] - uj*L[:,2]))[:,None]*Ln + proj[:,None]*np.array([0,0,1])

        # Deix => - ui - 1/l^2(el[0] - sum(ui ei[0]) - ui [1,0,0] )*l - proj[:,None]*[1,0,0]
        self.add_derivatives(self.const_idx["T2"], e_idx_i_x.repeat(3),( - ui[:,None]*np.array([1,0,0]) - proj_ei_dx).flatten())
        # Deiy => - ui - 1/l^2(el[1] - sum(ui ei[1]) - ui [0,1,0] )*l - proj[:,None]*[0,1,0]
        self.add_derivatives(self.const_idx["T2"], e_idx_i_y.repeat(3),( - ui[:,None]*np.array([0,1,0]) - proj_ei_dy).flatten())
        # Deiz => - ui - 1/l^2(el[2] - sum(ui ei[2]) - ui [0,0,1] )*l - proj[:,None]*[0,0,1]
        self.add_derivatives(self.const_idx["T2"], e_idx_i_z.repeat(3),( - ui[:,None]*np.array([0,0,1]) - proj_ei_dz).flatten())

        # Dejx => - uj - 1/l^2(el[0] - sum(ui ej[0]) - uj [1,0,0] )*l - proj[:,None]*[1,0,0]
        self.add_derivatives(self.const_idx["T2"], e_idx_j_x.repeat(3),( - uj[:,None]*np.array([1,0,0]) - proj_ej_dx).flatten())
        # Dejy => - uj - 1/l^2(el[1] - sum(ui ej[1]) - uj [0,1,0] )*l - proj[:,None]*[0,1,0]
        self.add_derivatives(self.const_idx["T2"], e_idx_j_y.repeat(3),( - uj[:,None]*np.array([0,1,0]) - proj_ej_dy).flatten())
        # Dejz => - uj - 1/l^2(el[2] - sum(ui ej[2]) - uj [0,0,1] )*l - proj[:,None]*[0,0,1]
        self.add_derivatives(self.const_idx["T2"], e_idx_j_z.repeat(3),( - uj[:,None]*np.array([0,0,1]) - proj_ej_dz).flatten())

        # Dekx => - uk[1,0,0] + uk l[0]/l^2 l
        self.add_derivatives(self.const_idx["T2"], e_idx_k_x.repeat(3), (-uk[:,None]*np.array([1,0,0]) + uk[:,None]*(L[:, 0]/L_2)[:,None]*L).flatten())
        # Deky => - uk[0,1,0] + uk l[1]/l^2 l
        self.add_derivatives(self.const_idx["T2"], e_idx_k_y.repeat(3), (-uk[:,None]*np.array([0,1,0]) + uk[:,None]*(L[:, 1]/L_2)[:,None]*L).flatten())
        # Dekz => - uk[0,0,1] + uk l[2]/l^2 l
        self.add_derivatives(self.const_idx["T2"], e_idx_k_z.repeat(3), (-uk[:,None]*np.array([0,0,1]) + uk[:,None]*(L[:, 2]/L_2)[:,None]*L).flatten())

        # Del => [1,0,0] - l[0]/l^2 l
        self.add_derivatives(self.const_idx["T2"], e_idx_l_x.repeat(3), (np.array([1,0,0]) - (L[:, 0]/L_2)[:,None]*L).flatten())
        # Del => [0,1,0] - l[1]/l^2 l
        self.add_derivatives(self.const_idx["T2"], e_idx_l_y.repeat(3), (np.array([0,1,0]) - (L[:, 1]/L_2)[:,None]*L).flatten())
        # Del => [0,0,1] - l[2]/l^2 l
        self.add_derivatives(self.const_idx["T2"], e_idx_l_z.repeat(3), (np.array([0,0,1]) - (L[:, 2]/L_2)[:,None]*L).flatten())


        self.set_r(self.const_idx["T2"], (vvl - ui[:,None]*vvi - uj[:,None]*vvj - uk[:,None]*vvk - proj[:,None]*L).flatten())
        
        #print("E T2:", self.r[self.const_idx["T2"]]@self.r[self.const_idx["T2"]])
        #print("Torsal Fair Energy: ", self.r@self.r)
        self.L_norm = np.linalg.norm(L, axis=1)


        #print("\nT1 E:", self.r[self.const_idx["T1"]]@self.r[self.const_idx["T1"]])
        #print("T2 E:", self.r[self.const_idx["T2"]]@self.r[self.const_idx["T2"]])
        #print("Energy T2", self.r[self.const_idx["T2"]]@self.r[self.const_idx["T2"]])








        



 
    






        
       