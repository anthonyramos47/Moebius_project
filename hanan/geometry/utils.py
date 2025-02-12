import numpy as np
import os
import polyscope as ps
import igl
from scipy.optimize import minimize
from scipy.spatial import KDTree

def unit(v):
    """normalize a list of vectors v
    """
    if len(v.shape) == 1:
        unit_v = v/np.linalg.norm(v)
    else:
        unit_v = v/np.linalg.norm(v, axis=1)[:, None]
    return unit_v

def proj(v, u):
    """
        Project v on u
    """
    
    v = np.array(v)
    u = np.array(u)

    vu = vec_dot(v, u)
    uu = vec_dot(u, u)

    if len(v.shape) == 1 and len(u.shape) == 1:
        proj = vu/uu*u
    else:
        proj = (vu/uu)[:,None]*u
    return proj


def barycenters(v, f):
    """ Function to compute the barycenters of the faces
    Input:
        v: vertices
        f: faces
    """
     
    bary = np.zeros((len(f), 3))
    for i, face in enumerate(f):
        b = v[face]
        b = np.sum(b, axis=0)/len(face)
        bary[i] = b

    return bary

def barycentric_coordinates_app(vi, vj, vk, vl):
    """ Function to find the barycentric coordinates of a point vl 
        in the triangle defined by vi, vj, vk
    """
    def bar_coord(x, vi, vj, vk, vl):
        b1, b2, b3 = x
        return np.linalg.norm(b1*vi + b2*vj + b3*vk - vl)
    
    init = np.array([1/3, 1/3, 1/3])

    # Perform the optimization
    result = minimize(bar_coord, init, args=(vi, vj, vk, vl), tol=1e-6)
    
    # Sol
    b1, b2, b3 = result.x

    return b1, b2, b3 


def barycentric_coordinates(vi, vj, vk, vl):
    """ Function to find the barycentric coordinates of a point vl 
        in the triangle defined by vi, vj, vk
    """

    # Define the linear system matrix
    A = np.vstack([vi, vj, vk])
    
    # Sol
    b1, b2, b3 = np.linalg.solve(A, vl)

    return b1, b2, b3    

def orth_proj(v, u):
    """ Orthogonal projection of v on u
    """
    return v - proj(v, u)

def vec_dot(v1, v2, ax=1):
    """ Dot product between two lists of vectors v1, v2
    """
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        dot =  v1@v2
    elif ax == 1:
        dot = np.einsum('ij,ij->i', v1, v2)
    elif ax == 0:
        dot = np.einsum('ij,ij->j', v1, v2)

    return dot

def circle_3pts(p1, p2, p3):
    """
    Function that take three list of points
    return the center and radius of the circle that pass through them.
    """ 

    # Check if p1, p2, p3 are list of points or single points
    if len(p1.shape) == 1:
        p1 = p1[None, :]
    if len(p2.shape) == 1:
        p2 = p2[None, :]
    if len(p3.shape) == 1:
        p3 = p3[None, :]
    

    # Create local coordinate system
    
    u1 = unit(p2-p1) 
    u2 = unit( np.cross(p3-p1, u1) ) # axis direction
    u3 = np.cross(u2, u1)
    
    # Find the center and radius in the new system

    # bx = p2-p1 . u1
    bx = np.sum((p2 - p1)*u1, axis=1)
    cx = np.sum((p3 - p1)*u1, axis=1)
    cy = np.sum((p3 - p1)*u3, axis=1)

    # h = ( (cx - bx/2)**2 + cy**2 - (bx/2)**2 )/(2*cy)
    h = ((cx-bx/2)**2 + cy**2 - (bx/2)**2 )/(2*cy)
    
    bx /= 2

    radius = np.linalg.norm(np.vstack((bx,h)),axis=0)


    center = p1 + (bx)[:,None]*u1 + h[:,None]*u3 
    
    return center, radius, u2


# ====================== Polyscope Functions =================

def draw_polygon(vertices, color, name="_"):
    """
        Register a polygon as a surface
    """
    ps.register_surface_mesh(name, vertices, [np.arange(len(vertices))[:, None]], color=color, transparency=0.8)
    
def draw_plane(p0, n, size=(1,1), name="_"):
    """
        Register a plane as a surface
    """
    aux = n + np.array([1,0,0])

    v1 = unit(orth_proj(aux, n))

    v2 = unit(np.cross(n, v1))

    v1 *= size[0]
    v2 *= size[1]

    vertices = np.array([p0 + v1 + v2, p0 + v1 - v2, p0 - v1 - v2, p0 - v1 + v2])

    ps.register_surface_mesh(name, vertices, [np.arange(len(vertices))[:, None]], color=(0.1, 0.1, 0.1), transparency=0.6)


def write_obj(filename, vertices, faces):
    """
        Write obj file
    """
    file_name = str(filename)
    obj_file = open(file_name, 'w')
    for v in vertices:
        obj_file.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    for f in faces:
        obj_file.write('f ')
        for idx in f:
            obj_file.write('{} '.format(idx+1))
        obj_file.write('\n')
        
    obj_file.close()

def read_obj(filename):
    """
        Read obj file and return vertices and faces
    """
    file_name = str(filename)
    obj_file = open(file_name, encoding='utf-8')
    vertices_list = []
    faces_list = []
    for l in obj_file:
        splited_line = l.split(' ')
        if splited_line[0] == 'v':
            split_x = splited_line[1].split('\n')
            x = float(split_x[0])
            split_y = splited_line[2].split('\n')
            y = float(split_y[0])
            split_z = splited_line[3].split('\n')
            try:
                z = float(split_z[0])
            except ValueError:
                print('WARNING: disable line wrap when saving .obj')
            vertices_list.append([x, y ,z])
        elif splited_line[0] == 'f':
            v_list = []
            L = len(splited_line)
            try:
                for i in range(1, L):
                    splited_face_data = splited_line[i].split('/')
                    v_list.append(int(splited_face_data[0]) - 1 )
                faces_list.append(v_list)
            except ValueError:
                v_list = []
                for i in range(1, L-1):
                    v_list.append(int(splited_line[i]) - 1 )
                faces_list.append(v_list)
    try:
        faces_list = np.array(faces_list)
    except:
        pass 
    
    return np.array(vertices_list), faces_list


def add_cross_field(mesh, name, vec1, vec2, rad, size, col):
    """ Function to add cross field to polyscope
    """
    mesh.add_vector_quantity(name+"_vec1" ,    vec1, defined_on ='faces', enabled=True, radius=rad, length=size, color=col)
    mesh.add_vector_quantity(name+"_-vec1",   -vec1, defined_on ='faces', enabled=True, radius=rad, length=size, color=col)
    mesh.add_vector_quantity(name+"_vec2" ,    vec2, defined_on ='faces', enabled=True, radius=rad, length=size, color=col)
    mesh.add_vector_quantity(name+"_-vec2",   -vec2, defined_on ='faces', enabled=True, radius=rad, length=size, color=col)



def normalize_vertices(v, factor=1):
    """ Function set the mesh into the a bounding box.
    """

    # Get the bounding box
    min_v = np.min(v, axis=0)
    max_v = np.max(v, axis=0)
    
    # Compute the center
    size = max_v - min_v

    max_dimension = max(size)

    # Compute scale factors for each dimension
    scale_factors = factor / max_dimension

    # Translate by the negative of the min_coords
    translated_vertices = v - min_v

    # Scale vertices to fit into the unit bounding box
    normalized_vertices = translated_vertices * scale_factors

    return normalized_vertices


def create_hex_face(radius, offset, n=6):
    """
        Function to create a hexagon face
    """

    # Define the center of the hexagon
    center = np.array([0, 0, 0])

    # Calculate the coordinates of the hexagon vertices
    h_v = np.array([center + radius * np.array([np.cos(2 * np.pi * k / 6), np.sin(2 * np.pi * k / 6), offset + np.random.random()]) for k in range(6)])

    # Move the center to the average of the vertices
    center = center +  np.array([0, 0, np.mean(h_v[:,2])])

    # Include the center as a vertex
    h_v = np.vstack((center, h_v)) 

    # Define the face list with triangle indices
    h_f = np.vstack((np.array([[i, (i + 1)%7, 0] for i in range(1, 6)]), np.array([6,1,0])))

    return h_v, h_f


def flat_array_variables(arr, n=3):
    # Multiply elements by 3
    multiplied = arr * n
    # Create a sequence for each element and reshape for concatenation
    for i in range(1, n):
        multiplied = np.vstack([multiplied, arr * n + i])
    
    # Flatten the array to match the desired output format
    transformed = multiplied.flatten('F')
    return transformed



def compute_volume_of_tetrahedron(p1, p2, p3, p4):
    # Each p should be an array of coordinates [x, y, z]
    mat = np.ones((4, 4))
    mat[0, 1:] = p1
    mat[1, 1:] = p2
    mat[2, 1:] = p3
    mat[3, 1:] = p4
    volume = np.abs(np.linalg.det(mat)) / 6
    return volume


def compute_planarity(p1,p2,p3,p4):
    """ Function to compute planarity of 4 points
    """

    # p1 --v0--> p2
    #  |        |
    # v1        |
    #  |        |
    # p4 --v2--> p3

    # Define v0 vector
    v0 = unit(p2 - p1)

    # Definve v1 vector
    v1 = unit(p4 - p1)

    # Define v2 vector
    v2 = unit(p3 - p4)

    # Compute the normal
    n = np.cross(v0, v1)

    # Compute the planarity
    planarity = np.abs(vec_dot(n, v2))

    return planarity
    

def extract_edges(faces):
    edges = set()
    for face in faces:
        num_vertices = len(face)
        if num_vertices < 2:
            continue  # Skip if the face has less than two vertices (not a valid face)
        # Loop through each vertex in the face
        for i in range(num_vertices):
            # Create an edge from the current vertex to the next, wrapping around to the first
            v1 = face[i]
            v2 = face[(i + 1) % num_vertices]  # Wrap around using modulo
            edge = tuple(sorted([v1, v2]))  # Sort the tuple to avoid duplicates like (v2, v1)
            edges.add(edge)
    return edges

def indices_flatten_dim(arr, n=3):
    
    return 3 * np.repeat(arr, n) + np.tile(range(n), len(arr))



def find_sphere(p1, p2, p3, p4):
    # Create matrix A from coordinates subtraction
    A = np.array([
        [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]],
        [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]],
        [p4[0] - p1[0], p4[1] - p1[1], p4[2] - p1[2]]
    ])
    
    # Create vector d from squared distances difference
    d = 0.5 * np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1),
        np.dot(p4, p4) - np.dot(p1, p1)
    ])
    
    # Solve the linear system A * center = d
    center = np.linalg.solve(A, d)
    # Calculate the radius
    radius = np.sqrt(np.sum((p1 - center)**2))
    
    return center, radius


def get_Implicit_sphere(c, r):

    A = 1/(2*r)

    B = (2*A)[:,None]*c

    C = (1 - np.linalg.norm(B, axis=1)**2 )/(-4*A)

    # Check if value is 1 or close to 1
    for i in range(len(c)):
        assert np.isclose(B[i]@B[i] - 4*A[i]*C[i], 1), "Error in the implicit sphere" 

    return A, B, C

def Implicit_to_CR(A,B,C):

    c = B/(2*A)[:,None]


    r = np.sqrt(np.einsum('ij,ij->i',B,B) - 4*A*C)/(2*A)

    return c, r

def remove_edges_by_ls(edges, ls):

    new_ls = []
    for i in range(len(edges)):
        
        if edges[i][0] in ls and edges[i][1] in ls:
            new_ls.append(edges[i])

    return np.array(new_ls)

def search_edge(edges, e):
    # Find indices where both elements match in any order
    matching_indices = np.where((edges[:, 0] == e[0]) & (edges[:, 1] == e[1]) |
                            (edges[:, 0] == e[1]) & (edges[:, 1] == e[0]))
    
    # Return the first matching index
    return matching_indices[0][0] if len(matching_indices[0]) > 0 else None


def torsal_recompute(opt, du, dv):
    """ 
    Function to initialize the torsal directions optimization
    Input:
        n: Normals (u,v,3)
        l: Line congruence (u, v, 3)
        du: Surface derivative u (u,v,3)
        dv: Surface derivative v (u,v,3)
    """

    # Reorient l 
    sign = np.sign(np.einsum('ijk,ijk->ij', l, n))
    l = l*sign[:,:,None]
    
    # Compute the line congruence at the baricenter and the line congruence directions
    lc, lu, lv = lc_info_at_grid_points(l)

    # Reshape line congruence and normals
    lc = lc.reshape(-1, 3)
    lu = lu.reshape(-1, 3)
    lv = lv.reshape(-1, 3)

    # Normalize the line congruence
    lc /= np.linalg.norm(lc, axis=1)[:, None]

    # Compute the torsal directions 
    t1, t2, ut1, vt1, ut2, vt2, _ = torsal_directions(lc, lu, lv, du, dv)

    # Copute lines in torsal directions
    lt1 = ut1[:, None]*lu + vt1[:, None]*lv
    lt2 = ut2[:, None]*lu + vt2[:, None]*lv

    nt1 = np.cross(lc, unit(t1))
    nt2 = np.cross(lc, unit(t2))

    return ut1, vt1, ut2, vt2, lc, lt1, lt2, nt1, nt2

    # lt1 = unit(ut1[:, None]*lu + vt1[:, None]*lv)
    # lt2 = unit(ut2[:, None]*lu + vt2[:, None]*lv)

    # # Compute the torsal plane normal
    # nt1 = unit(np.cross(lc, lt1))
    # nt2 = unit(np.cross(lc, lt2))

    # # Init the torsal directions
    # X[var_idx["u1"]] = ut1 
    # X[var_idx["v1"]] = vt1 
    # X[var_idx["u2"]] = ut2 
    # X[var_idx["v2"]] = vt2 

    # # Init the torsal plane normals
    # X[var_idx["nt1"]] = nt1.flatten()
    # X[var_idx["nt2"]] = nt2.flatten()

def orient_crossfield(M1, M2, M3):
    """
    Function to orient the cross fields, M1, M2, M3 represent the cross field directions vectors in matrix form.
    """

    # Compute angles between cross fields M1 M2 
    angles_matrix = M1.T@M2

    # Check which cross field is the closest to the other by checking tuples at each row min
    idx = np.argmin(angles_matrix, axis=1)

    # Transform idx into a matrix if [1, 0] then P = I if [0, 1] then P = [0 ,1 ; 1, 0]
    if idx[0] == 0:
        P = np.array([[1, 0], [0, 1]])
    else:
        P = np.array([[0, 1], [1, 0]])

    M2 = M2@P


    # Compute angles between cross fields M1 M3
    angles_matrix = M1.T@M3

    # Check which cross field is the closest to the other by checking tuples at each row min
    idx = np.argmin(angles_matrix, axis=1)

    # Transform idx into a matrix if [1, 0] then P = I if [0, 1] then P = [0 ,1 ; 1, 0]
    if idx[0] == 0:
        P = np.array([[1, 0], [0, 1]])
    else:
        P = np.array([[0, 1], [1, 0]])
    

    M3 = M3@P

    return M1, M2, M3


def interpolate_torsal_Q_tri(t1, t2, V, F):
    from geometry.mesh import Mesh


    # Get the vertices
    v0, v1, v2, v3 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]], V[F[:, 3]]

    # Compute the barycenter
    vc = (v0 + v1 + v2 + v3)/4

    # Get topology of V F 
    m = Mesh()
    m.make_mesh(V, F)

    # Get adjacent faces
    f_f_adj = m.face_face_adjacency_list()

    # Get number of vertices
    #new_idx = len(V)-1

    # New torsal directions
    t1_new = []
    t2_new = []

    # New vc
    vc_new = []

    # Loop 
    for i in range(len(F)):
        print(f"Face {i}")

        f = F[i]

        # Subdivide the quad into four triangles
        local_F = np.array([
            [0, 1, 2],
            [0, 2, 3]
            ])

        # Local vertices
        local_V = np.vstack((v0[i], v1[i], v2[i], v3[i], vc[i]))

        lv0, lv1, lv2 = local_V[local_F[:, 0]], local_V[local_F[:, 1]], local_V[local_F[:, 2]]

        n_f = unit(np.cross(lv1 - lv0, lv2 - lv0))
        

        # Compute barycenters of new triangles
        local_vc = np.sum(local_V[local_F], axis=1)/3
        

        for bary in local_vc:
            vc_new.append(bary)

        # Interpolation
        if len(f_f_adj[i]) == 4:
            print("Im in 4")
            adj_f = f_f_adj[i]
            # create local triangle mesh for interpolation
            B = np.vstack((vc[i], vc[adj_f]))

            # Triangles 
            T = np.array(
                [[0, 1, 2], 
                 [0, 2, 3], 
                 [0, 3, 4], 
                 [0, 4, 1]])
                        
            # Get the closest points on the remeshed mesh
            _, id_T, cpts = igl.point_mesh_squared_distance(local_vc, B, T)

            # Get vertices of the nearest triangles
            tv0, tv1, tv2 = B[T[id_T, 0]], B[T[id_T, 1]], B[T[id_T, 2]]

            # Compute the barycentric coordinates of each point projected on the mesh
            iglbar = igl.barycentric_coordinates_tri(cpts, tv0, tv1, tv2)

            # Interpolate per point at corresponding triangle
            for j, idx_t in enumerate(id_T):
                

                M1 = np.array([t1[i], t2[i]]).T
                M2 = np.array([t1[adj_f[T[idx_t, 1] - 1 ]], t2[adj_f[T[idx_t, 1] - 1 ]]]).T
                M3 = np.array([t1[adj_f[T[idx_t, 2] - 1 ]], t2[adj_f[T[idx_t, 2] - 1 ]]]).T

                # Orient cross fields
                #M1, M2, M3 = orient_crossfield(M1, M2, M3)

                # Get corresponding barycentric coordinates
                w1, w2, w3 = iglbar[j]

                # Interpoalte
                M_int = w1*M1 + w2*M2 + w3*M3
                M_int /= np.linalg.norm(M_int, axis=1)[:, None]

                nt1 = M_int[:, 0]
                nt2 = M_int[:, 1]

                # Project onto corresponding triangle
                nt1 = nt1 - (nt1@n_f[j])*n_f[j]
                nt2 = nt2 - (nt2@n_f[j])*n_f[j]



                t1_new.append(nt1)
                t2_new.append(nt2)
        else: 
            adj_f = f_f_adj[i]
            
            # create local triangle mesh for interpolation
            B = np.vstack((vc[i], vc[adj_f]))

            T = np.zeros((len(adj_f)-1, 3), dtype=int)

            for j in range(len(adj_f)-1):
                T[j] = [0, j+1, j+2]

                        
            # Get the closest points on the remeshed mesh
            _, id_T, cpts = igl.point_mesh_squared_distance(local_vc, B, T)

            # Get vertices of the nearest triangles
            tv0, tv1, tv2 = B[T[id_T, 0]], B[T[id_T, 1]], B[T[id_T, 2]]

            # Compute the barycentric coordinates of each point projected on the mesh
            iglbar = igl.barycentric_coordinates_tri(local_vc, tv0, tv1, tv2)

            # Interpolate per point at corresponding triangle
            for j, idx_t in enumerate(id_T):
                M1 = np.array([t1[i], t2[i]]).T
                M2 = np.array([t1[adj_f[T[idx_t, 1] - 1 ]], t2[adj_f[T[idx_t, 1] -1 ]]]).T
                M3 = np.array([t1[adj_f[T[idx_t, 2] - 1  ]], t2[adj_f[T[idx_t, 2] - 1]]]).T

                # Orient cross fields
                #M1, M2, M3 = orient_crossfield(M1, M2, M3)

                # Get corresponding barycentric coordinates
                w1, w2, w3 = iglbar[j]

                # Interpoalte
                M_int = w1*M1 + w2*M2 + w3*M3

                M_int /= np.linalg.norm(M_int, axis=1)[:, None]

                
                nt1 = M_int[:, 0]
                nt2 = M_int[:, 1]

                # Project onto corresponding triangle
                nt1 = nt1 - (nt1@n_f[j])*n_f[j]
                nt2 = nt2 - (nt2@n_f[j])*n_f[j]

                
                t1_new.append(nt1)
                t2_new.append(nt2)

    return np.array(t1_new), np.array(t2_new), np.array(vc_new)
            

                


