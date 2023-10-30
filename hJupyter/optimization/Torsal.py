# Planarity constraint implementation

import numpy as np
import geometry as geo
from optimization.constraint import Constraint

class Torsal(Constraint):

    def __init__(self) -> None:
        """ Torsal directions constraint energy
        E = \sum{f \in F} || n_t1 . e_c ||^2 + || n_t1 . t1 ||^2 + || n_f1 . tt1 ||^2 + || n_t1 . n_t1 - 1 ||^2
            +             || n_t2 . e_c ||^2 + || n_t2 . t2 ||^2 + || n_f2 . tt2 ||^2 + || n_t2 . n_t2 - 1 ||^2
        where, 
            n_t .- normal of torsal plane.
            t   .- torsal direction in the face of triangular mesh T.
            tt  .- torsal direction in the face of triangular mesh \tilde{T} [second envelope]
            e_c .- direction of the line congruence at the barycenter of the face.
        """
        super().__init__()
        self.nV = None # Number of vertices
        self.nF = None # Number of faces
        self.ncf = None # List of circumcenter axis direction
        self.bf = None # List of circumcenters
        self.fvij = None # List of the edge vectors per each face
        self.v = None # List of the vertices per each face
        self.vc= None # List og the barycenters of the faces
        self.var_idx = None # Dic to store the index of the variables
        self.const_idx = None # Dic to store the index of the constraints

    
    def initialize_constraint(self, X, V, F, bf, ncf, w=1 ) -> np.array:
        # Input
        # X: variables [ e   | a | b | n_t  | d_i ] 
        # X  size      [ 3*V | F | F | 3*F  | F   ]
        # V: Vertices
        # F: Faces
        # bf: circumcenters of the faces
        # ncf: normals of the circumcenters
        # w: weight

        # Set weight
        self.w = w

        # Set circumcenters
        self.bf = bf

        # Set circumcenters axis
        self.ncf = ncf

        # Number of vertices
        self.nV = len(V)
        
        # Number of faces
        self.nF = len(F)

        # Number of constraints 4*|F|
        self.const = 8*self.nF

        # Define indices indices
        self.var_idx = {    "e"  : np.arange( 0                     , 3*self.nV), 
                            "a1" : np.arange( 3*self.nV             , 3*self.nV + self.nF),
                            "b1" : np.arange( 3*self.nV + self.nF   , 3*self.nV + 2*self.nF),
                            "nt1": np.arange( 3*self.nV + 2*self.nF , 3*self.nV + 5*self.nF),
                            "a2" : np.arange( 3*self.nV + 5*self.nF , 3*self.nV + 6*self.nF),
                            "b2" : np.arange( 3*self.nV + 6*self.nF , 3*self.nV + 7*self.nF),
                            "nt2": np.arange( 3*self.nV + 7*self.nF , 3*self.nV + 10*self.nF),
                            "df" : np.arange( 3*self.nV + 10*self.nF, 3*self.nV + 11*self.nF),
                    }
        
        self.const_idx = {  "nt1.ec"  : np.arange( 0                  , self.nF),
                            "nt1.t1"  : np.arange( self.nF            , 2*self.nF),
                            "nt1.tt1" : np.arange( 2*self.nF          , 3*self.nF),
                            "nt1.nt1" : np.arange( 3*self.nF          , 4*self.nF),
                            "nt2.ec"  : np.arange( 4*self.nF          , 5*self.nF),
                            "nt2.t2"  : np.arange( 5*self.nF          , 6*self.nF),
                            "nt2.tt2" : np.arange( 6*self.nF          , 7*self.nF),
                            "nt2.nt2" : np.arange( 7*self.nF          , 8*self.nF),

                    }
        
        # Number of variables
        self.var = len(X)

        # Get df 
        df = X[-self.nF:]

        # Get ei
        e_i = X[:3*self.nV].reshape(self.nV, 3)
        
        # Get vertices of the faces
        vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]
        self.v = np.vstack((vi, vj, vk))
        self.vc = np.sum(vi + vj + vk, axis=0)/3

        # Compute the edge vectors per each face
        self.fvij = np.empty((self.nF, 2, 3), dtype=float)

        # Compute the edge vectors per each face
        self.fvij[:,0] = vj - vi
        self.fvij[:,1] = vk - vi

        # Compute the direction of the line congruence at the barycenters
        ec = self.compute_ec(df, e_i, F)

        # Compute initial directions of normals of torsal plane
        nt1 = np.cross(ec, self.fvij[:,0])
        nt1 /= np.linalg.norm(nt1, axis=1)[:, None]

        nt2 = np.cross(ec, self.fvij[:,0])
        nt2 /= np.linalg.norm(nt2, axis=1)[:, None]
        

        # Set initial directions of normals of torsal plane
        X[self.var_idx["nt1"]] = nt1.flatten()
        X[self.var_idx["nt2"]] = nt2.flatten()

        # Set initial a1 
        X[self.var_idx["a1"]] = np.zeros(self.nF)
        X[self.var_idx["b1"]] = np.ones(self.nF)

        # Set initial b
        X[self.var_idx["a2"]] = -np.ones(self.nF)
        X[self.var_idx["b2"]] = -np.zeros(self.nF)

        return X

    def compute(self, X, F) -> None:

        # Init J
        J = np.zeros((self.const, self.var), dtype=np.float64)

        # Init r
        r = np.zeros(self.const, dtype=np.float64)

        J, r = self.fill_J(X, J, r, F)

        # Update J
        self.J =  J
        # Update r
        self.r =  r

    def fill_J_t(self, J, r, e, ec, a1, b1, nt1, vij, vik, vvi, vvj, vvk, dcvi, dcvj, dcvk, F, torsal=1):

        if torsal == 1:
            nt_ec = "nt1.ec"
            nt_t = "nt1.t1"
            nt_tt = "nt1.tt1"
            nt_nt = "nt1.nt1"
            nt = "nt1"
            va = "a1"
            vb = "b1"
        else:
            nt_ec = "nt2.ec"
            nt_t = "nt2.t2"
            nt_tt = "nt2.tt2"
            nt_nt = "nt2.nt2"
            nt = "nt2"
            va = "a2"
            vb = "b2"

        # Compute derivatives of de (nt.ec)
        deix, deiy, deiz, dejx, dejy, dejz, dekx, deky, dekz = self.compute_decnt(dcvi, dcvj, dcvk, e, nt1, F)
        # Compute derivatives of d df (nt.ec)
        d_df, eicf_ei, ejcf_ej, ekcf_ek = self.compute_d_df(e, nt1, F)

        # Compute t
        t = self.compute_t(a1, b1)

        # Compute tt
        tt, vvij, vvik = self.compute_tt(a1, b1, vvi, vvj, vvk)

        # constrain indices
        c_idx = self.const_idx
        v_idx = self.var_idx

        # Fill J for || nt.ec ||^2
        # Set derivatives de (nt.ec)
        J[c_idx[nt_ec], 3*F[:,0] ]   = 2/3*deix
        J[c_idx[nt_ec], 3*F[:,0]+1 ] = 2/3*deiy
        J[c_idx[nt_ec], 3*F[:,0]+2 ] = 2/3*deiz
        J[c_idx[nt_ec], 3*F[:,1] ]   = 2/3*dejx
        J[c_idx[nt_ec], 3*F[:,1]+1 ] = 2/3*dejy
        J[c_idx[nt_ec], 3*F[:,1]+2 ] = 2/3*dejz
        J[c_idx[nt_ec], 3*F[:,2] ]   = 2/3*dekx
        J[c_idx[nt_ec], 3*F[:,2]+1 ] = 2/3*deky
        J[c_idx[nt_ec], 3*F[:,2]+2 ] = 2/3*dekz


        # Set derivateives dnt (nt.ec)
        J[c_idx[nt_ec], v_idx[nt]     ] = ec.flatten()
        
        # Set derivatives d(df) (nt.ec)
        J[c_idx[nt_ec], v_idx["df"]] = d_df

        # Set r
        r[c_idx[nt_ec]] = np.sum(nt1*ec, axis=1)
        
        # Fill J for || nt.t ||^2; t = a vij + b vik

        # Set derivatives dnt (nt.t)
        J[c_idx[nt_t], v_idx[nt]    ] = t.flatten()
        

        # Set derivatives da (nt.t)
        J[c_idx[nt_t], v_idx[va]] = self.vec_dot(vij, nt1) 
        # Set derivatives db (nt.t)
        J[c_idx[nt_t], v_idx[vb]] = self.vec_dot(vik, nt1)

        # Set r 
        r[c_idx[nt_t]] = np.sum(t*nt1, axis=1)

        # Fill J for || nt.tt ||^2; tt = a vvij + b vvik
        # Set derivatives dnt (nt.tt)
        J[c_idx[nt_tt], v_idx[nt]   ] = tt.flatten()

        # Set derivatives da (nt1.tt1)
        J[c_idx[nt_tt], v_idx[va]] = self.vec_dot(vvij, nt1)
        # Set derivatives db (nt1.tt1)
        J[c_idx[nt_tt], v_idx[vb]] = self.vec_dot(vvik, nt1)

        # Set derivatives d e (nt.tt)
        J[c_idx[nt_tt], 3*F[:,0] ]   = -2*(a1+b1)*deix
        J[c_idx[nt_tt], 3*F[:,0]+1 ] = -2*(a1+b1)*deiy
        J[c_idx[nt_tt], 3*F[:,0]+2 ] = -2*(a1+b1)*deiz

        J[c_idx[nt_tt], 3*F[:,1] ]   = 2*a1*dejx
        J[c_idx[nt_tt], 3*F[:,1]+1 ] = 2*a1*dejy
        J[c_idx[nt_tt], 3*F[:,1]+2 ] = 2*a1*dejz

        J[c_idx[nt_tt], 3*F[:,2] ]   = 2*b1*dekx
        J[c_idx[nt_tt], 3*F[:,2]+1 ] = 2*b1*deky
        J[c_idx[nt_tt], 3*F[:,2]+2 ] = 2*b1*dekz

        # Set derivatives d df (nt.tt)
        J[c_idx[nt_tt], v_idx["df"]] = self.vec_dot((2*a1*(ejcf_ej - eicf_ei) + 2*b1*(ekcf_ek - eicf_ei)), nt1)
        
        # Set r
        r[c_idx[nt_tt]] = np.sum(tt*nt1, axis=1)

        # Fill J for || nt.nt - 1 ||^2
        J[c_idx[nt_nt], v_idx[nt]] = nt1.flatten()


        # Set r
        r[c_idx[nt_nt]] = np.sum(nt1*nt1, axis=1) - 1

        return J, r 


    def fill_J(self, X, J, r, F):

        
        e, a1, b1, nt1, a2, b2, nt2, df = self.uncurry_variables(X)

        # Get vertices of second envelope
        vvi, vvj, vvk, dcvi, dcvj, dcvk = self.compute_second_env(df, e, F)

        # Compute ec
        ec = self.compute_ec(df, e, F)

        # Get vij, vik
        vij = self.fvij[:,0]
        vik = self.fvij[:,1]

        J, r = self.fill_J_t(J, r, e, ec, a1, b1, nt1, vij, vik, vvi, vvj, vvk, dcvi, dcvj, dcvk, F, 1)
        J, r = self.fill_J_t(J, r, e, ec, a2, b2, nt2, vij, vik, vvi, vvj, vvk, dcvi, dcvj, dcvk, F, 2)

        return J, r




            


    def compute_second_env(self, df, ei, F):
        """ Compute the second envelope of the mesh
        Input:
            df .- distance to center np.array
            vi .- Vertices np.array
            ei .- direction vectors np.array
        Output:
            vvi .- Vertices of the second envelope
            dcv .- Direction from vi to cf
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

        ei /= np.linalg.norm(ei, axis=1)[:, None]


        # compute the second envelope as vi + (dcv.ei) ei 
        vvi = 2*np.sum(dcvi*ei[F[:,0]], axis=1)[:, None] * ei[F[:,0]] + vi
        vvj = 2*np.sum(dcvj*ei[F[:,1]], axis=1)[:, None] * ei[F[:,1]] + vj
        vvk = 2*np.sum(dcvk*ei[F[:,2]], axis=1)[:, None] * ei[F[:,2]] + vk

        return vvi, vvj, vvk, dcvi, dcvj, dcvk


    def compute_sphere_centers(self, df):
        """ Compute sphere centers
        Input:
            bf .- Circumcenters
            df .- Distance to center
            ncf .- Normal of circumcenters
        """
        return self.bf + df[:, None] * self.ncf
    
    def compute_ec(self, df, e_i, F):

        # Get first envelope
        vc = self.vc

        # Compute second envelope
        vvi, vvj, vvk, _, _, _ = self.compute_second_env(df, e_i, F)

        # Compute varycenter in second envelope
        vvc = np.sum(vvi + vvj + vvk, axis=0)/3

        ec = vvc - vc

        if len(ec.shape) == 1:
            ec = np.array([ec])

        return ec

    def uncurry_variables(self, X):

        v_idx = self.var_idx


        # Get e_i directions
        e = X[v_idx["e"]].reshape(-1, 3)

        # Get a 
        a1 = X[v_idx["a1"]]

        # Get b
        b1 = X[v_idx["b1"]]

        # Get n_t
        nt1 = X[v_idx["nt1"]].reshape(-1, 3)

        # Get a2
        a2 = X[v_idx["a2"]]

        # Get b2
        b2 = X[v_idx["b2"]]

        # Get n_t2
        nt2 = X[v_idx["nt2"]].reshape(-1, 3)

        # Get d_i
        di = X[v_idx["df"]]

        return e, a1, b1, nt1, a2, b2, nt2, di
    
    def vec_dot(self, a, b):
        return np.sum(a*b, axis=1)
    
    def compute_decnt(self, dcvi, dcvj, dcvk, e, nt, F):
        """ Function to compute the derivative of nt.ec with respect to e_i
            (nt.ec) = (nt.(ei + ej + ek)/3) = (nt).(vvc - vc);
            vc = (vi + vj + vk)/3
            vvi = vi + (dcv.ei) ei
        """
        # Get directions per vertex
        ei = e[F[:,0]]
        ej = e[F[:,1]]
        ek = e[F[:,2]]  

        deix = self.vec_dot( ( dcvi[:,0]*ei +  self.vec_dot(ei, dcvi)*np.array([1,0,0]) ), nt)
        deiy = self.vec_dot( ( dcvi[:,1]*ei +  self.vec_dot(ei, dcvi)*np.array([0,1,0]) ), nt)
        deiz = self.vec_dot( ( dcvi[:,2]*ei +  self.vec_dot(ei, dcvi)*np.array([0,0,1]) ), nt)

        dejx = self.vec_dot( ( dcvj[:,0]*ej +  self.vec_dot(ej, dcvj)*np.array([1,0,0]) ), nt)
        dejy = self.vec_dot( ( dcvj[:,1]*ej +  self.vec_dot(ej, dcvj)*np.array([0,1,0]) ), nt)
        dejz = self.vec_dot( ( dcvj[:,2]*ej +  self.vec_dot(ej, dcvj)*np.array([0,0,1]) ), nt)

        dekx = self.vec_dot( ( dcvk[:,0]*ek +  self.vec_dot(ek, dcvk)*np.array([1,0,0]) ), nt)
        deky = self.vec_dot( ( dcvk[:,1]*ek +  self.vec_dot(ek, dcvk)*np.array([0,1,0]) ), nt)
        dekz = self.vec_dot( ( dcvk[:,2]*ek +  self.vec_dot(ek, dcvk)*np.array([0,0,1]) ), nt)

        return deix, deiy, deiz, dejx, dejy, dejz, dekx, deky, dekz


    def compute_d_df(self, e, nt, F):
        """ Compute the derivative of nt.ec with respect to df
            (nt.ec) = (nt.(ei + ej + ek)/3) = (nt).(vvc - vc);
            vc = (vi + vj + vk)/3
            vvi = vi + (dcv.ei) ei; dcv = cf - vi 
            cf = bf + df ncf
            => d df (nt.ec) = 2/3 [nt.( (ei.ncf)* ei + (ej.ncf)*ej   + (ek.ncf)*ek )]
        """
        
        # Get directions per vertex
        ei = e[F[:,0]]
        ej = e[F[:,1]]
        ek = e[F[:,2]] 

        # Get normals of the circumcenters
        ncf = self.ncf 

        # Compute dot produts
        eicf = self.vec_dot(ei, ncf)
        ejcf = self.vec_dot(ej, ncf)
        ekcf = self.vec_dot(ek, ncf)

        eicf_ei = eicf[:,None]*ei
        ejcf_ej = ejcf[:,None]*ej
        ekcf_ek = ekcf[:,None]*ek

        # Compute the derivative of nt.ec with respect to df
        d_df = 2/3*self.vec_dot( (eicf_ei + ejcf_ej + ekcf_ek ), nt)

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
    
       
 
    






        
       