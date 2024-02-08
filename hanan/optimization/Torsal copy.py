# Planarity constraint implementation

import numpy as np
from hanan.geometry.utils import vec_dot, solve_torsal, unit, find_initial_torsal_th_phi
from hanan.optimization.constraint import Constraint



class Torsal(Constraint):

    def __init__(self) -> None:
        """ 
        Initializes a new instance of the Torsal class.

        Torsal directions constraint energy, we assume nt are unitary vectors.
        E = \sum{f \in F} || n_t1 . e_c ||^2 + || n_t1 . t1 ||^2 + || n_f1 . tt1 ||^2 
            +             || n_t2 . e_c ||^2 + || n_t2 . t2 ||^2 + || n_f2 . tt2 ||^2 
        where, 
            n_t .- normal of torsal plane.
            t   .- torsal direction in the face of triangular mesh T.
            tt  .- torsal direction in the face of triangular mesh \tilde{T} [second envelope]
            e_c .- direction of the line congruence at the barycenter of the face.
        """
        super().__init__()
        self.name = "Torsal_Dir"
        self.nV = None # Number of vertices
        self.F = None # Faces
        self.nF = None # Number of faces
        self.v = None # List of the vertices per each face
        self.vc= None # List og the barycenters of the faces
        self.fvij = None # List of the edge vectors per each face
        self.ecnorms = None # List of the norms of the line congruence directions
        self.tnorms1 = None # List of the norms of the torsal directions
        self.tnorms2 = None # List of the norms of the torsal directions
        self.ttnorms1 = None # List of the norms of the torsal directions
        self.ttnorms2 = None # List of the norms of the torsal directions


    
    def initialize_constraint(self, X, var_indices, V, F) -> np.array:
        # Input
        # X: variables [ e   | a | b | n_t  | d_i ] 
        # X  size      [ 3*V | F | F | 3*F  | F   ]
        # V: Vertices
        # F: Faces
        # bf: circumcenters of the faces
        # ncf: normals of the circumcenters

        self.F = F

        # Number of vertices
        self.nV = len(V)
        
        # Number of faces
        self.nF = len(F)

        # Number of constraints 4*|F|
        self.const = 8*self.nF

        # Define indices indices
        self.var_idx = var_indices
    
        self.const_idx = {  "nt1.ec"  : np.arange( 0                  , self.nF),
                            "nt1.t1"  : np.arange( self.nF            , 2*self.nF),
                            "nt1.tt1" : np.arange( 2*self.nF          , 3*self.nF),
                            "nt2.ec"  : np.arange( 3*self.nF          , 4*self.nF),
                            "nt2.t2"  : np.arange( 4*self.nF          , 5*self.nF),
                            "nt2.tt2" : np.arange( 5*self.nF          , 6*self.nF)                 
                        }
        
        # Number of variables
        self.var = len(X)
        
        # Get ei
        e  = X[var_indices["e"]].reshape(-1, 3)

        # Get barycentric ec 
        ec = np.sum(e[F], axis=1)/3

        i, j, k = F[:,0], F[:,1], F[:,2]
        
        # Get vertices of the faces
        vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]

        # Set vertices per face
        self.v = vi, vj, vk
        # Set barycenters per face
        self.vc = (vi + vj + vk)/3

        # Compute the edge vectors per each face
        self.fvij = np.empty((self.nF, 2, 3), dtype=float)

        # Compute the edge vectors per each face
        self.fvij[:,0] = vj - vi
        self.fvij[:,1] = vk - vi

        # Set ec norms
        self.ecnorms = np.linalg.norm(ec, axis=1)

        # Compute initial torsal directions
        t1, t2, a1, a2, b, validity = solve_torsal(vi, vj, vk, e[i], e[j], e[k])

        vij = self.fvij[:,0]
        vik = self.fvij[:,1]
        # Compute angles between t1 and vij 
        th, phi, _ = find_initial_torsal_th_phi(t1, t2, vij, vik)

        idx_f = np.where(validity == False)[0]
        phi[idx_f] = th[idx_f] + np.pi/2

        X[var_indices["th"]] = th
        X[var_indices["phi"]] = phi

        # Compute torsal directions with respec to angle
        at1 = np.cos(th)[:,None]*vij + np.sin(th)[:,None]*vik
        at2 = np.cos(th + phi)[:,None]*vij + np.sin(th + phi)[:,None]*vik
 
        # Compute torsal norms
        self.tnorms1 = np.linalg.norm(at1, axis=1)
        self.tnorms2 = np.linalg.norm(at2, axis=1)

        # Compute torsal in second envelope
        vvi = vi + e[i]
        vvj = vj + e[j]
        vvk = vk + e[k]

        tt1 = unit(a1[:,None]* (vvj - vvi) + b[:,None] * (vvk - vvi))
        tt2 = unit(a2[:,None] * (vvj - vvi) + b[:,None] * (vvk - vvi))

        att1 =       np.cos(th)[:,None]*(vvj - vvi) + np.sin(th)[:,None]*(vvk - vvi)
        att2 = np.cos(th + phi)[:,None]*(vvj - vvi) + np.sin(th + phi)[:,None]*(vvk - vvi)

        # Compute torsal norms
        self.ttnorms1 = np.linalg.norm(att1, axis=1)
        self.ttnorms2 = np.linalg.norm(att2, axis=1)

        # Set initial normals to torsal planes

        nt1 = unit(np.cross(unit(at1), unit(att1)))
        nt2 = unit(np.cross(unit(at2), unit(att2)))

        # t1_tt1 = np.cross(unit(at1), unit(att1))
        # t2_tt2 = np.cross(unit(at2), unit(att2))

        # print("planar t1 :", np.sum(abs(vec_dot(t1_tt1, unit(ec))))/self.nF)
        # print("planar t2 :", np.sum(abs(vec_dot(t2_tt2, unit(ec))))/self.nF)

        # print("nt1ec: ", np.sum(vec_dot(nt1, ec)/self.ecnorms)/self.nF)
        # print("nt2ec: ", np.sum(vec_dot(nt2, ec)/self.ecnorms)/self.nF)
        # print("nt1t1: ", np.sum(vec_dot(nt1, at1)/self.tnorms1)/self.nF)
        # print("nt2t2: ", np.sum(vec_dot(nt2, at2)/self.tnorms2)/self.nF)
        # print("nt1tt1: ", np.sum(vec_dot(nt1, att1)/self.ttnorms1)/self.nF)
        # print("nt2tt2: ", np.sum(vec_dot(nt2, att2)/self.ttnorms2)/self.nF)

        X[self.var_idx["nt1"]] = nt1.flatten()
        X[self.var_idx["nt2"]] = nt2.flatten()

        



    def compute(self, X) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
        """
        e, nt1, nt2, th, phi = self.uncurry_X(X, "e", "nt1", "nt2", "th", "phi")

        e   = e.reshape(-1, 3)
        nt1 = nt1.reshape(-1, 3)
        nt2 = nt2.reshape(-1, 3)

        ec = np.sum(e[self.F], axis=1)/3

        # Set derivatives || nt1.ec/||ec|| ||^2  + || nt2.ec/||ec|| ||^2
        self.energy_nt_ec(e, ec, nt1, nt2)

        # Set derivatives || nt1.t1/||t1|| ||^2  + || nt2.t2\||t2|| ||^2
        self.energy_nt_t(nt1, nt2, th, phi)

        # Set derivatives || nt1.tt1/||tt1|| ||^2  + || nt2.tt2/||tt2|| ||^2
        self.energy_nt_tt(e, nt1, nt2, th, phi)
        
        self.ecnorms = np.linalg.norm(ec, axis=1)
    

    def energy_nt_ec(self, e, ec, nt1, nt2):
        """ Function to set up the derivatives and residual for
        the energy || nt1.ec/||ec|| ||^2  + || nt2.ec/||ec|| ||^2
        """

        i, j, k = self.F[:,0], self.F[:,1], self.F[:,2]

        e_idx = self.var_idx["e"]

        indices_i = e_idx[3 * np.repeat(i, 3) + np.tile(range(3), len(i))]
        indices_j = e_idx[3 * np.repeat(j, 3) + np.tile(range(3), len(j))]
        indices_k = e_idx[3 * np.repeat(k, 3) + np.tile(range(3), len(k))]

        # d nt1 => ec/||ec||
        self.add_derivatives(self.const_idx["nt1.ec"].repeat(3), self.var_idx["nt1"], (ec/self.ecnorms[:, None]).flatten())
        
        deijk = (nt1/self.ecnorms[:, None]).flatten()
        # d ei,j,k => 1/3 nt1.ec/||ec||
        self.add_derivatives(self.const_idx["nt1.ec"].repeat(3), indices_i, deijk/3)
        self.add_derivatives(self.const_idx["nt1.ec"].repeat(3), indices_j, deijk/3)
        self.add_derivatives(self.const_idx["nt1.ec"].repeat(3), indices_k, deijk/3)

        # Set r
        self.set_r(self.const_idx["nt1.ec"], vec_dot(nt1, ec)/self.ecnorms)

        # d nt2 => ec/||ec||
        self.add_derivatives(self.const_idx["nt2.ec"].repeat(3), self.var_idx["nt2"], (ec/self.ecnorms[:, None]).flatten())

        deijk2 = (nt2/self.ecnorms[:, None]).flatten()
        # d ei,j,k => 1/3 nt2.ec/||ec||
        self.add_derivatives(self.const_idx["nt2.ec"].repeat(3), indices_i, deijk2/3)
        self.add_derivatives(self.const_idx["nt2.ec"].repeat(3), indices_j, deijk2/3)
        self.add_derivatives(self.const_idx["nt2.ec"].repeat(3), indices_k, deijk2/3)

        # Set r
        self.set_r(self.const_idx["nt2.ec"], vec_dot(nt2, ec)/self.ecnorms)

    def energy_nt_t(self, nt1, nt2, th, phi):
        """ Function to set up the derivatives and residual for
        the energy || nt1.t1/||t1|| ||^2  + || nt2.t2\||t2|| ||^2
        """

        # Get directions of the faces
        vij = self.fvij[:,0]
        vik = self.fvij[:,1]

        alpha = th + phi

        t1 =    np.cos(th)[:,None]*vij + np.sin(th)[:,None]*vik
        t2 = np.cos(alpha)[:,None]*vij + np.sin(alpha)[:,None]*vik

        dt1 =    -np.sin(th)[:,None]*vij + np.cos(th)[:,None]*vik
        dt2 = -np.sin(alpha)[:,None]*vij + np.cos(alpha)[:,None]*vik

        # E: ||nt1.t1||; t1 = cos(th) vij + sin(th) vik
        # d nt1 => t1\||t1||
        self.add_derivatives(self.const_idx["nt1.t1"].repeat(3), self.var_idx["nt1"], (t1/self.tnorms1[:, None]).flatten())

        # d th => nt1.(-sin(th) vij + cos(th) vik)\||t1||
        self.add_derivatives(self.const_idx["nt1.t1"], self.var_idx["th"], vec_dot(nt1, dt1)/self.tnorms1)

        # r 
        self.set_r(self.const_idx["nt1.t1"], vec_dot(nt1, t1)/self.tnorms1)

        # E: ||nt2.t2||; t2 = cos(alpha) vij + sin(alpha) vik
        self.add_derivatives(self.const_idx["nt2.t2"].repeat(3), self.var_idx["nt2"], (t2/self.tnorms2[:, None]).flatten())

        # d th => nt2.(-sin(alpha) vij + cos(alpha) vik)\||t2||
        self.add_derivatives(self.const_idx["nt2.t2"], self.var_idx["th"], vec_dot(nt2, dt2)/self.tnorms2)

        # d phi => nt2.(-sin(alpha) vij + cos(alpha) vik)\||t2||
        self.add_derivatives(self.const_idx["nt2.t2"], self.var_idx["phi"], vec_dot(nt2, dt2)/self.tnorms2)

        # r
        self.set_r(self.const_idx["nt2.t2"], vec_dot(nt2, t2)/self.tnorms2)

        # update torsal norms
        self.tnorms1 = np.linalg.norm(t1, axis=1)
        self.tnorms2 = np.linalg.norm(t2, axis=1)

    def energy_nt_tt(self, e, nt1, nt2, th, phi):
        """ Function to set up the derivatives and residual for
        the energy || nt1.tt1/||tt1|| ||^2  + || nt2.tt2/||tt2|| ||^2
        """

        # Get direction on the faces
        vij = self.fvij[:,0]
        vik = self.fvij[:,1]

        # Get vertices of the faces
        vi, vj, vk = self.v
        
        i, j, k = self.F[:,0], self.F[:,1], self.F[:,2]

        indices_i = self.var_idx["e"][3 * np.repeat(i, 3) + np.tile(range(3), len(i))]
        indices_j = self.var_idx["e"][3 * np.repeat(j, 3) + np.tile(range(3), len(j))]
        indices_k = self.var_idx["e"][3 * np.repeat(k, 3) + np.tile(range(3), len(k))]

        # compute the second envelope as vvi = vi + e[i]
        vvi = vi + e[i]
        vvj = vj + e[j]
        vvk = vk + e[k]

        vvij = vvj - vvi
        vvik = vvk - vvi

        alpha = th + phi

        # Compute torsal directions in the second envelope
        tt1 =    np.cos(th)[:,None]*vvij +    np.sin(th)[:,None]*vik
        tt2 = np.cos(alpha)[:,None]*vvij + np.sin(alpha)[:,None]*vvik

        dtt1_th =    -np.sin(th)[:,None]*vij + np.cos(th)[:,None]*vik
        dtt2_th = -np.sin(alpha)[:,None]*vij + np.cos(alpha)[:,None]*vik

        nt1nor = nt1/self.tnorms1[:, None]

        # E : ||nt1.tt1||; tt1 = cos(th) (vvj - vvi) + sin(th) (vvk - vvi)
        # d nt1 => tt1/||tt1||
        self.add_derivatives(self.const_idx["nt1.tt1"].repeat(3), self.var_idx["nt1"], (tt1/self.ttnorms1[:, None]).flatten())

        # d th => nt1.(-sin(th) (vvj - vvi) + cos(th) (vvk - vvi))/||tt1||
        self.add_derivatives(self.const_idx["nt1.tt1"], self.var_idx["th"], vec_dot(nt1, dtt1_th)/self.ttnorms1)

        # d ei => -nt1*(cos(th) + sin[th])/||tt1||
        self.add_derivatives(self.const_idx["nt1.tt1"].repeat(3), indices_i, (-(np.cos(th) + np.sin(th))[:,None]*nt1nor).flatten())

        # d ej => nt1*cos(th)/||tt1||
        self.add_derivatives(self.const_idx["nt1.tt1"].repeat(3), indices_j, (np.cos(th)[:,None]*nt1nor).flatten())

        # d ek => nt1*sin(th)/||tt1||
        self.add_derivatives(self.const_idx["nt1.tt1"].repeat(3), indices_k, (np.sin(th)[:,None]*nt1nor).flatten())

        # r
        self.set_r(self.const_idx["nt1.tt1"], vec_dot(nt1, tt1)/self.ttnorms1)

        nt2nor = nt2/self.tnorms2[:, None]

        # E : ||nt2.tt2||; tt2 = cos(alpha) (vvj - vvi) + sin(alpha) (vvk - vvi)
        # d nt2 => tt2/||tt2||
        self.add_derivatives(self.const_idx["nt2.tt2"].repeat(3), self.var_idx["nt2"], (tt2/self.ttnorms2[:, None]).flatten())

        # d th => nt2.(-sin(alpha) (vvj - vvi) + cos(alpha) (vvk - vvi))/||tt2||
        self.add_derivatives(self.const_idx["nt2.tt2"], self.var_idx["th"], vec_dot(nt2, dtt2_th)/self.ttnorms2)

        # d phi => nt2.(-sin(alpha) (vvj - vvi) + cos(alpha) (vvk - vvi))/||tt2||
        self.add_derivatives(self.const_idx["nt2.tt2"], self.var_idx["phi"], vec_dot(nt2, dtt2_th)/self.ttnorms2)

        # d ei => -nt2*(cos(alpha) + sin[alpha])/||tt2||
        self.add_derivatives(self.const_idx["nt2.tt2"].repeat(3), indices_i, (-(np.cos(alpha) + np.sin(alpha))[:,None]*nt2nor).flatten())

        # d ej => nt2*cos(alpha)/||tt2||
        self.add_derivatives(self.const_idx["nt2.tt2"].repeat(3), indices_j, (np.cos(alpha)[:,None]*nt2nor).flatten())

        # d ek => nt2*sin(alpha)/||tt2||
        self.add_derivatives(self.const_idx["nt2.tt2"].repeat(3), indices_k, (np.sin(alpha)[:,None]*nt2nor).flatten())

        # r
        self.set_r(self.const_idx["nt2.tt2"], vec_dot(nt2, tt2)/self.ttnorms2)


        # update torsal norms
        self.ttnorms1 = np.linalg.norm(tt1, axis=1)
        self.ttnorms2 = np.linalg.norm(tt2, axis=1)

       
 







        
       