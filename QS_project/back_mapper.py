import numpy as np
import igl
import polyscope as ps
from scipy.spatial import KDTree

# Import the necessary libraries
import os
import sys
from pathlib import Path
import argparse

hanan_path = os.getenv('HANANLAB_PATH')
if not hanan_path:
    raise EnvironmentError("HANANLAB_PATH environment variable not set")
sys.path.append(hanan_path)

from geometry.utils import write_obj


# -----------------------------------------------------------------------------

def distance_point_to_triangle(p, v0, v1, v2):
    """
    Compute the minimum distance between points p and a triangle defined by vertices v0, v1, and v2.
    """

    print(p.shape, v0.shape, v1.shape, v2.shape)

    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    p_v0 = p - v0

    # Compute dot products
    # Change the np.sums for np.einsum

    dot00 = np.einsum('ij,ij -> i', v0v1, v0v1)
    dot01 = np.einsum('ij,ij -> i', v0v1, v0v2)
    dot02 = np.einsum('ij,ij -> i', v0v1, p_v0) 
    dot11 = np.einsum('ij,ij -> i', v0v2, v0v2)
    dot12 = np.einsum('ij,ij -> i', v0v2, p_v0)

    

    # Compute barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Clamp barycentric coordinates to avoid points outside the triangle
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    # Compute closest points on the triangles
    closest_points = v0 + u[:, None] * v0v1 + v[:, None] * v0v2

    return closest_points, np.hstack((1-u-v, u, v)).reshape(-1, 3)

def closest_point_on_mesh(mesh_vertices, mesh_triangles, query_points):
    """
    Compute the closest points on a triangular mesh to multiple query points using KDTree for efficiency.
    """

    vc = np.sum(mesh_vertices[mesh_triangles], axis=1) / 3

    tree = KDTree(vc)

    # Find nearest triangles
    dists, nearest_triangle_idxs = tree.query(query_points)

    # Get the faces that contain the nearest vertex idx
    # Search in wich face is contained that index

    # Get vertices of the nearest triangles
    nearest_triangles = mesh_triangles[nearest_triangle_idxs]

    print(nearest_triangles)

    # Get vertices of the nearest triangles
    v0 = mesh_vertices[nearest_triangles[:, 0]]
    v1 = mesh_vertices[nearest_triangles[:, 1]]
    v2 = mesh_vertices[nearest_triangles[:, 2]]

    # Compute closest points on the nearest triangles
    closest_points_on_triangles, bar_coord  = distance_point_to_triangle(query_points, v0, v1, v2)

    return closest_points_on_triangles, bar_coord


# Read obj file
def read_obj(file):
    """
    Read an obj file and return the vertices
    """
    vertices = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line[2:].split()])
    return np.array(vertices)

# Read obj with faces
def read_obj_faces(file):
    """
    Read an obj file and return the vertices and faces
    """
    vertices = []
    faces = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line[2:].split()])
            if line.startswith('f '):
                faces.append([int(x) - 1 for x in line[2:].split()])
                if len(faces[-1]) != 4:
                    print('Warning: face with more than 4 vertices', len(faces[-1]))

    return np.array(vertices), faces          


# -----------------------------------------------------------------------------

# Create the parser
parser = argparse.ArgumentParser(description="Backmapper name")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
file_name = parser.parse_args().file_name            

path = os.getcwd()
path = os.path.join(path, 'data', 'Remeshing', file_name)


or_mesh = file_name + '_start'
de_mesh = file_name + '_deformed'
remesh  = file_name + '_remeshed'


or_mesh_path = os.path.join(path, or_mesh + '.obj')
de_mesh_path = os.path.join(path, de_mesh + '.obj')
remesh_path  = os.path.join(path, remesh + '.obj')

back_mesh_path = os.path.join(path, file_name + '_backmapped.obj')

# Load meshes
orV, orF = igl.read_triangle_mesh(or_mesh_path)
rV , rF = read_obj_faces(remesh_path)
dV , dF = igl.read_triangle_mesh(de_mesh_path)

# Get the closest points on the remeshed mesh
sd, l, cpts = igl.point_mesh_squared_distance(rV, dV, dF)

# Get vertices of the nearest triangles
v0, v1, v2 = dV[dF[l, 0]], dV[dF[l, 1]], dV[dF[l, 2]]

# Compute the barycentric coordinates of each point projected on the mesh
iglbar = igl.barycentric_coordinates_tri(cpts, v0, v1, v2)

# Get vertices of the original mesh triangles
ov0, ov1, ov2 = orV[dF[l, 0]], orV[dF[l, 1]], orV[dF[l, 2]]

# Back map the points with the barycentric coordinates
Backmapped = iglbar[:,0][:,None]*ov0 + iglbar[:,1][:,None]*ov1 + iglbar[:,2][:,None]*ov2


write_obj(back_mesh_path, Backmapped, rF)


# Visualize the results
ps.init()

ps.register_surface_mesh("deformed", dV, dF, transparency=0.5, enabled=False)
ps.register_surface_mesh("original", orV, orF, transparency=0.5)
ps.register_surface_mesh("remeshed" ,      rV, rF, enabled=False)
ps.register_surface_mesh("Backmapped", Backmapped, rF)

ps.show()


