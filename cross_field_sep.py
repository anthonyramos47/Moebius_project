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

# Separate cross field
def sep_cross_field(v):
    """
    Separate the cross field into two vector fields
    """

    # Number of vertices
    n_v = v.shape[0]//4

    # First vector field
    v1 = v[:2*n_v, :]
    v2 = v[2*n_v:, :]

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
        for v in v1:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for l in l1:
            f.write('l {} {}\n'.format(l[0]+1, l[1]+1))

    # Write the second vector field
    with open(file2, 'w') as f:
        for v in v2:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for l in l2:
            f.write('l {} {}\n'.format(l[0]+1, l[1]+1))
# -----------------------------------------------------------------------------
            
path = '/Users/cisneras/hanan/hananLab/'
name = 'Florian_A_25_TA_45_deformed_crossfield'

file_path = path + name + '.obj'

# Read the cross field
v = read_obj(file_path)

# Separate the cross field
v1, v2, l1, l2 = sep_cross_field(v)

# Save the two vector fields
save_fields(name, v1, l1, v2, l2)
