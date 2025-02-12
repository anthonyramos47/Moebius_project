# Import the necessary libraries
import os
import sys
from pathlib import Path
import argparse

# Obtain the path HananLab; this is the parent directory of the hananLab/hanan directory
# <Here you can manually add the path direction to the hananLab/hanan directory>
# Linux 
path = os.getenv('HANANLAB_PATH')
if not path:
    raise EnvironmentError("HANANLAB_PATH environment variable not set")
sys.path.append(path)

# Verification of the path
print(path)

# Import the necessary libraries for visualization and computation
import igl
import polyscope as ps
import polyscope.imgui as psim
import numpy as np


# Import Mesh class
from geometry.mesh import Mesh
from geometry.utils import *


class Lmesh:
    """
    Class L-mesh
    """

    def __init__(self):
        self.topology = Mesh()
        self.PV = None # Vertex planes
        self.CE = None # Conical Edges is matrix (n, n, 6) non-symmetric
        self.SF = None # Spherical faces
        self.Spheres = None # Spheres

    def init_L_mesh(self):
        """
        Function to initialize the L-mesh 
        """
        self.PV = np.zeros((self.topology.V, 6))
        # Vertex planes are define by normal, point
        
        self.SF = []
        # Spherical face are define by sphere[center, radius] and points of spherical face [3, 4, 6] or n if irre
        # gular mesh

        self.Spheres = np.zeros((self.topology.F, 4))

        self.CE = np.zeros((self.topology.V, self.topology.V, 7))
        # Conical edge is a matrix of n x n x 7 non-symmetric, where we have center, normal, radius


    def get_tangent_circles(self, sph0_id, sph1_id):
        """
        Function that compute the contact circles between two spheres and its enveloping cone
        """
        sph0, sph1 = self.Spheres[sph0_id], self.Spheres[sph1_id]

        # Get Spheres 
        c0, r0 = sph0
        c1, r1 = sph1 
        id0 = sph0_id
        id1 = sph1_id

        # If r1 is greater than r0, swap the spheres
        if r1 > r0:
            c0, r0, c1, r1 = c1, r1, c0, r0
            id0 = sph1_id
            id1 = sph0_id


        # Compute the distance between the spheres
        dc = np.linalg.norm(c1 - c0)

        # unit vector from c0 to c1
        d01 = unit(c1 - c0)

        # Scaling factor 
        fac = (r0 - r1)/dc 

        # Compute radius of circles
        circ0_r = np.sqrt( r0**2 - (fac*r0)**2)
        circ1_r = np.sqrt( r1**2 + (fac*r1)**2)

        # Compute the centers of the circles
        circ0_c = c0 + fac * d01 * r0
        circ1_c = c1 + fac * d01 * r1

        self.CE[id0, id1] = np.join(circ0_c, d01, circ0_r)
        self.CE[id1, id0] = np.join(circ1_c, d01, circ1_r)

    
    def get_int_points(self, sph, circ0, circ1):
        # sphere, circle1, circle2

        c0, n0, r0 = circ0  # 3D circle (center, normal, radius)
        c1, n1, r1 = circ1 
        sc, sr = sph
        # Get line of intersection 
        l = unit(np.cross(n0, n1)) # Line direction

        # Compute point in the line
        dc01 = c1 - c0 # Vector from c0 to c1
        p = (dc01) -  proj(dc01 , n0) # Projection of dc01 on n0
        p = (p-c1) -  proj(p -c1, n1) # Projection of p-c1 on n1

        # Project center of the sphere on the line
        p_sc = proj(sc - p, l)

        # Compute the distance between the center of the sphere and the line
        d = np.linalg.norm(p_sc - sc)

        # Compute the intersection points
        if d > sr:
            return None
        else:
            h = np.sqrt(sr**2 - d**2)
            p0 = p_sc + h * unit(sc - p)
            p1 = p_sc - h * unit(sc - p)
            return p0, p1
        
    
    def draw_sphere(self, sph_id):
        """
        Function to draw a sphere
        """
        center, radius = self.Spheres[sph_id]
        print(center, radius)
        sphere = ps.register_point_cloud("Sphere_"+str(sph_id), np.array(center).reshape(1, 3))
        sphere.set_radius(radius, relative=False)

        

def main():

    # Create an instance of the L-mesh
    lmesh = Lmesh()

    # 
    vertices = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ]

    f = [
        [0, 1, 2],
        [0, 2, 5],
        [0, 4, 1],
        [1, 3, 2]
    ]

    lmesh.Spheres = [
        [np.array([0, -1, 0.21]), 1],
        [np.array([0, 3, -0.1]), 0.5]
    ]

    # Load the mesh
    lmesh.topology.make_mesh(vertices, f)
    
    # Visualize the mesh
    ps.init()

    # Create a mesh object
    #mesh = ps.register_surface_mesh("Mesh", np.array(lmesh.topology.vertices[:,:3]), np.array(lmesh.topology.faces))
    lmesh.draw_sphere(0)
    lmesh.draw_sphere(1)

    ps.show()

main()


        