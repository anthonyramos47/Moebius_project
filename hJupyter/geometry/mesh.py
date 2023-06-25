"""
    Mesh Data Structure
    ===================
    This module contains the mesh data structure and its associated methods.
    The mesh data structure is a collection of vertices, edges, and faces.
    The mesh data structure is used to represent the geometry of a 3D object.

    The mesh data is stored in half-edge data structure in a matrix form.
    Half-Edge r-th row : | Origin | Twin | Face | Next | Previous | Edge |
                             0        1     2      3       4          5

    The code is based on the code implemented by Davide Pellis. 

"""
__author__ = "Anthony Ramos"

import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix

def Mesh(object):

    def __init__(self):

        # Number of Vertices
        self._V = 0

        # Number of Faces
        self._F = 0

        # Number of Edges
        self._E = 0

        # Half-Edge Data Structure
        self._halfedges = None

        # Vertices
        self._vertices = None

        # Boundary Half-Edges
        self._boundary_halfedges = None

    # Properties: ------------------------------------------------------------------

    @property
    def V(self):
        return self._V
    
    @property
    def F(self):
        return self._F
    
    @property
    def E(self):
        return self._E
    
    @property
    def BH(self):
        return self._boundary_halfedges
    
    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices):
        vertices = np.asarray(vertices, dtype=np.float64)
        # Assert that vertices are 3D 
        assert vertices.ndim == 2, "vertices must be an array of shape (n,3)"
        assert vertices.shape[1] == 3, "vertices must be an array of shape (n,3)"
        
        # Set the vertices
        self._vertices = vertices
        self.update_dimensions()        

    @property
    def H(self):
        return self._halfedges
    
    @H.setter
    def H(self, halfedges):
        halfedges = np.asarray(halfedges, dtype=np.int64)
        # Assert that halfedges are 6D 
        assert halfedges.ndim == 2, "halfedges must be an array of shape (n,6)"
        assert halfedges.shape[1] == 6, "halfedges must be an array of shape (n,6)"
        
        # Set the halfedges
        self._halfedges = halfedges
        self.update_dimensions()


    # Updates Functions: ------------------------------------------------------------------ 

    def update_dimensions(self):
        self._V =  np.amax(self.halfedges[:,0]) + 1
        self._F =  np.amax(self.halfedges[:,2]) + 1
        self._E =  np.amax(self.halfedges[:,5]) + 1

    # Methods: ------------------------------------------------------------------

    def __str__(self):
        string = "Mesh Data Structure: |V| = {}, |F| = {}, |E| = {}".format(self.V, self.F, self.E)
        return string
    
    def copy_mesh(self):
        copyMesh = Mesh()
        copyMesh._vertices = self.vertices.copy()
        copyMesh.H = self.H.copy()
        copyMesh._V = self.V
        copyMesh._F = self.F
        copyMesh._E = self.E
        return copyMesh

    # Mesh Functions: ------------------------------------------------------------------

    def make_mesh(self, vertices, faces):
        
        self._vertices = np.array(vertices, dtype=np.float64)
        self._V = vertices.shape[0]
        self._F = len(faces)
        
        # Init Columns of Half-Edge Data Structure
        origin = np.zeros((self.V, 1), dtype=np.int16)
        twin_i = np.empty((1), dtype=np.int16)
        twin_j = np.empty((1), dtype=np.int16)
        face = np.zeros((self.F, 1), dtype=np.int16)
        next = np.empty((1), dtype=np.int16)
        previous = np.empty((1), dtype=np.int16)
        edge = np.empty((1), dtype=np.int16)

        # halfedge counter
        h_count = 0
        for f in range(self.F):
            # First half-edge of the face
            # ---------------------------
            n = len(faces[f]) # Number of vertices in face f
            origin.append(faces[f][0]) # First vertex in the face
            face[h_count] = f # Face of the corresponding half-edge
            next = np.hstack(next, (h_count + 1)) # Next vertex of the face
            previous = np.hstack( previous, (h_count + n - 1)) # Last vertex of the face
            
            # Record half edge direction of twin ij -> ji
            twin_i = np.hstack(twin_i, (face[f][1]))
            twin_j = np.hstack(twin_j, (face[f][0]))
            
            h_count += 1
            # ---------------------------

            
            # Loop over the remaining vertices in the face
            for i in range(1, n - 1):
                # Add next origin
                origin = np.hstack(origin, (faces[f][i]))
                # Face to next half-edge
                face[h_count] = f

                next = np.hstack(next, (h_count + i + 1))
                previous = np.hstack(previous, (h_count + i- 1))
                
                # Record half edge direction of twin ij -> ji
                twin_i = np.hstack(twin_i, (face[f][i + 1]))
                twin_j = np.hstack(twin_j, (face[f][i]))

                h_count += 1

            # Last half-edge of the face
            # ---------------------------
            origin = np.hstack(origin, (faces[f][n - 1]))
            face[h_count] = f
            next = np.hstack(next, (h_count - n + 1))
            previous = np.hstack(previous, (h_count - 1))
            twin_i = np.hstack(twin_i, (face[f][0]))
            twin_j = np.hstack(twin_j, (face[f][n - 1]))

            h_count += 1

        # Set twin half-edges
        # Create a sparse matrix with the twin half-edges (hafl-edge index) = (i,j)
        twin = coo_matrix((np.arange(h_count) + 1, (twin_i, twin_j)), shape=(self.h_count, self.h_count)).toarray()
        twin = twin.tocsc()
        
        # Set half-edges data structure
        H = np.zeros((h_count, 6), dtype=np.int16)
        H[:, 0] = origin
        H[:, 2] = face
        H[:, 3] = next
        H[:, 4] = previous

        H[:, 1] = twin[origin, H[H[:,2], 0] ] - 1

        # Boundary half-edges
        # Indices of half-edges with no twin
        b = np.where(H[:, 1] == -1)[0]
        # Boundary half-edges
        boundary = H[b, :]
        boundary[:, 0] = H[H[b,2],0]
        boundary[:, 2] = -1
        boundary[:, 1] = b

        B = len(boundary)

        

        

            




    
    
    
