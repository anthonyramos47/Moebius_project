import numpy as np
import os
import argparse


# Create the parser
parser = argparse.ArgumentParser(description="Name of experiment")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
file_name = parser.parse_args().file_name

# Get the current working directory
cwd = os.getcwd()

# save_folder
save_folder = os.path.join(cwd, 'QS_project', 'data', 'Remeshing', file_name)



# -----------------------------------------------------------------------------
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

    return np.array(vertices), np.array(faces)          

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

    print(v1.shape, v2.shape)

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
    

    for i in range(len(v1_d)):
        if angle_v1_d1[i] > angle_v2_d1[i]:
            v1[l1[i,1]] = np.sign( d2 @ v1_d[i]) * v1_d[i]
            v2[l2[i,1]] = np.sign( d1 @ v1_d[i]) * v2_d[i]
        else:
            v1[l1[i,1]] = np.sign( d1 @ v2_d[i]) * v1_d[i]
            v2[l2[i,1]] = np.sign( d2 @ v2_d[i]) * v2_d[i]
   
   


# -----------------------------------------------------------------------------
cf   =  file_name +'_deformed_crossfield'
mesh =  file_name +'_deformed'


file_path = os.path.join(save_folder , cf + '.obj')
mesh_path = os.path.join(save_folder , mesh + '.obj')
save_path = os.path.join(save_folder, file_name)

# Read the cross field
v = read_obj(file_path)

vd, fd = read_obj_faces(mesh_path)

# Separate the cross field
v1, v2, l1, l2 = sep_cross_field(v)

# Get the directions of the diagonals
d1, d2 = get_dir_diag(vd, fd)

l1 = np.array(l1)
l2 = np.array(l2)

# print(v1[l1[0,0]], v1[l1[0,1]] )
# Reorient the lines
reorient(d1, d2, v1, v2, l1, l2)
# print(v1[l1[0,0]], v1[l1[0,1]] )
# Save the two vector fields
save_fields(save_path, v1, l1, v2, l2)
