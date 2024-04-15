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


# -----------------------------------------------------------------------------

def distance_point_to_triangle(p, v0, v1, v2):
    """
    Compute the minimum distance between points p and a triangle defined by vertices v0, v1, and v2.
    """

    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    p_v0 = p - v0

    # Compute dot products
    dot00 = np.sum(v0v1*v0v1, axis=1)
    dot01 = np.sum(v0v1*v0v2, axis=1)
    dot02 = np.sum(v0v1*p_v0, axis=1)
    dot11 = np.sum(v0v2*v0v2, axis=1)
    dot12 = np.sum(v0v2*p_v0, axis=1)

    

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

# Separate cross field
def sep_cross_field(v):
    """
    Separate the cross field into two vector fields
    """

    # Number of vertices
    n_v = v.shape[0]//4

    # First vector field
    v1i, v1j = v[:n_v, :], v[n_v:2*n_v, :]

    # Second vector field
    v2i, v2j = v[2*n_v:3*n_v, :], v[3*n_v:, :]

    # Create the two vector fields
    v1_i = 0.5*(v1i + v1j)
    v1_f = v1_i + 0.1*(v1j - v1i)/np.linalg.norm(v1j - v1i, axis=1)[:, None]
    
    v2_i = 0.5*(v2i + v2j)
    v2_f = v2_i + 0.1*(v2j - v2i)/np.linalg.norm(v2j - v2i, axis=1)[:, None]

    v1 = np.concatenate([v1_i, v1_f], axis=0)
    v2 = np.concatenate([v2_i, v2_f], axis=0)




    # Lines joining
    l1 = [[i, i+n_v] for i in range(n_v)]
    l2 = [[i, i+n_v] for i in range(n_v)]

    return v1, v2, l1, l2


def save_fields(file_name, v1, l1, v2, l2):
    """
    Save the two vector fields into an obj file
    """

    # Create two files for each vector field
    file1 = file_name + '_D1.obj'
    file2 = file_name + '_D2.obj'


    # Write the first vector field
    with open(file1, 'w') as f:
        for v in v2:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for l in l1:
            f.write('l {} {}\n'.format(l[0]+1, l[1]+1))

    # Write the second vector field
    with open(file2, 'w') as f:
        for v in v1:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for l in l2:
            f.write('l {} {}\n'.format(l[0]+1, l[1]+1))

def get_dir_diag(v,f):
    """
    Get directions of the diagonals of the faces for reorientation
    """
    # v0, v1, v2, v3 = v[f[:,0]], v[f[:,1]], v[f[:,2]], v[f[:,3]]

    # d1 = v2 - v0
    # d1 /= np.linalg.norm(d1, axis=1)[:, None]
    # d2 = v1 - v3
    # d2 /= np.linalg.norm(d2, axis=1)[:, None]

    # n = np.cross(d1, d2)

    # d2 = np.cross(n, d1)

    # d2 /= np.linalg.norm(d2, axis=1)[:, None]

    d1 = np.array([0, 0, 1])
    d2 = np.array([1, 0, 0])

    return d1, d2

def reorient(d1, d2, v1, v2, l1, l2):
    """
    Reorient the lines of the cross field
    """

    l1 = np.array(l1)
    l2 = np.array(l2)

    # Get initial point and direction
    v1_p, v1_d = v1[l1[:,0]], v1[l1[:,1]]

    # Get initial point and direction
    v2_p, v2_d = v2[l2[:,0]], v2[l2[:,1]]

    # Check min angle between the diagonals
    angle_v1_d1 = np.arccos(np.abs(np.sum(v1_d*d1, axis=1)))
    angle_v2_d1 = np.arccos(np.abs(np.sum(v2_d*d1, axis=1)))
    

    idx_flip = np.where(angle_v1_d1 > angle_v2_d1)[0]
    idx_nor = np.where(angle_v1_d1 <= angle_v2_d1)[0]

    # Get signs 
    s11 = np.sign(np.sum(v1_d*d1, axis=1))
    s22 = np.sign(np.sum(v2_d*d2, axis=1))

    s12 = np.sign(np.sum(v1_d*d2, axis=1))
    s21 = np.sign(np.sum(v2_d*d1, axis=1))

    

    # Reorient the lines
    #v1[l1[:,1]][idx_flip] = s12[idx_flip, None] * v1_d[idx_flip]
    #v1[l1[:,1]][idx_nor]  = s11[idx_nor, None]  * v1_d[idx_nor]

    v1[l1[:,1]] = v1_p + s11[:,None]*v1[l1[:,1]]

    #v2[l2[:,1]][idx_flip] = s21[idx_flip, None] * v2_d[idx_flip]
    #v2[l2[:,1]][idx_nor]  = s22[idx_nor, None]  * v2_d[idx_nor]

    v2[l2[:,1]] = v2_p + s22[:,None]*v2[l2[:,1]]

    # v1[l1[:,1]] = v1_p + d1
    # v2[l2[:,1]] = v2_p + d2 


# -----------------------------------------------------------------------------

# Create the parser
parser = argparse.ArgumentParser(description="Backmapper name")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
file_name = parser.parse_args().file_name            

path =  hanan_path


or_mesh = file_name + '_start'
de_mesh = file_name + '_deformed'
remesh  = file_name + '_remesh'

or_mesh_path = path + or_mesh + '.obj'
de_mesh_path = path + de_mesh + '.obj'
remesh_path  = path + remesh + '.obj'

orV, orF = igl.read_triangle_mesh(or_mesh_path)
rV , rF = read_obj_faces(remesh_path)
dV , dF = igl.read_triangle_mesh(de_mesh_path)

sd, l, cpts = igl.point_mesh_squared_distance(rV, dV, dF)

# Get vertices of the nearest triangles
v0, v1, v2 = dV[dF[l, 0]], dV[dF[l, 1]], dV[dF[l, 2]]

cpt1, bar = distance_point_to_triangle(cpts, v0, v1, v2)
iglbar = igl.barycentric_coordinates_tri(cpts, v0, v1, v2)

orV += [0,0,2]

#newpts = iglbar[:, 0][:,None]*v0 + iglbar[:, 1][:,None]*v1 + iglbar[:, 2][:,None]*v2
newpts = iglbar[:,0][:,None]*orV[orF[l,0]] + iglbar[:,1][:,None]*orV[orF[l,1]] + iglbar[:,2][:,None]*orV[orF[l,2]]

valid = np.zeros(len(orF))
valid[0] = 1
valid[4] = 1

ps.init()
dmesh = ps.register_surface_mesh("deformed", dV, dF)
dmesh.add_scalar_quantity("valid", valid, defined_on='faces',enabled = False)

omesh = ps.register_surface_mesh("original", orV, orF)
omesh.add_scalar_quantity("valid", valid, defined_on='faces', enabled = False)

ps.register_surface_mesh("Remap", newpts, rF)
ps.register_surface_mesh("remeshed", rV, rF)

# ps.register_point_cloud("closest", cpt1, radius = 0.002)

# ps.register_surface_mesh("Closest", cpts, rF)
#ps.register_surface_mesh("original", orV, orF)
ps.show()


