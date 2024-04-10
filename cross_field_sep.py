import numpy as np

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
    v1_f = (v1j - v1i)/np.linalg.norm(v1j - v1i, axis=1)[:, None]
    
    v2_i = 0.5*(v2i + v2j)
    v2_f = (v2j - v2i)/np.linalg.norm(v2j - v2i, axis=1)[:, None]

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
            
path = '/Users/cisneras/hanan/hananLab/'
name = 'Nice_result_deformed_crossfield'
mesh = 'Nice_result_deformed'


file_path = path + name + '.obj'
mesh_path = path + mesh + '.obj'

# Read the cross field
v = read_obj(file_path)

vd, fd = read_obj_faces(mesh_path)

# Separate the cross field
v1, v2, l1, l2 = sep_cross_field(v)

# Get the directions of the diagonals
d1, d2 = get_dir_diag(vd, fd)

# Reorient the lines
reorient(d1, d2, v1, v2, l1, l2)

# Save the two vector fields
save_fields(name, v1, l1, v2, l2)
