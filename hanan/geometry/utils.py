import numpy as np
import polyscope as ps

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
    vv = vec_dot(u, u)

    if len(v.shape) == 1 and len(u.shape) == 1:
        proj = vu/vv*u
    else:
        proj = vu/vv[:, None]*u
    return proj

def barycenters(v, f):
     
    bary = np.zeros((len(f), 3))
    for i, face in enumerate(f):
        b = v[face]
        b = np.sum(b, axis=0)/len(face)
        bary[i] = b

    return bary

def barycentric_coordinates(vi, vj, vk, vl):
    """ Function to find the barycentric coordinates of a point vl 
        in the triangle defined by vi, vj, vk
    """

    # Define the linear system matrix
    A = np.vstack([vi, vj, vk])
    
    # Sol
    b1, b2, b3 = np.linalg.solve(A, vl)

    return b1,b2,b3


    



def orth_proj(v, u):
    return v - proj(v, u)

def vec_dot(v1, v2):
    """dot product between two lists of vectors v1, v2
    """
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        dot =  v1@v2
    else: 
        dot = np.sum(v1*v2, axis=1) 

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


    # h 
    # h = ( (cx - bx/2)**2 + cy**2 - (bx/2)**2 )/(2*cy)
    h = ((cx-bx/2)**2 + cy**2 - (bx/2)**2 )/(2*cy)
    
    bx /= 2

    radius = np.linalg.norm(np.vstack((bx,h)),axis=0)


    center = p1 + (bx)[:,None]*u1 + h[:,None]*u3 
    
    return center, radius, u2

def torsal_directions(v, f, e_i):
    num_faces = len(f)
    tors = np.zeros((2 * num_faces, 3))
    cos_tors = np.zeros(num_faces)
    for i in range(num_faces):
        # Extract vertex indices for the current face
        vertex_indices = f[i]

        # Calculate vectors e(u, v), e_u, e_v for the current face
        e_u, e_v = e_i[vertex_indices[1]] - e_i[vertex_indices[0]], \
                    e_i[vertex_indices[2]] - e_i[vertex_indices[0]]
        e = (e_i[vertex_indices[0]] + e_i[vertex_indices[1]] + e_i[vertex_indices[2]])/3

        # Calculate vectors a(u, v), a_u, a_v for the current face
        v_u, v_v = v[vertex_indices[1]] - v[vertex_indices[0]], \
                    v[vertex_indices[2]] - v[vertex_indices[0]]
        vv = (v[vertex_indices[0]] + v[vertex_indices[1]]+ v[vertex_indices[2]])/3

        # Calculate coefficients of the equation and solve it
        gamma_0 = np.linalg.det(np.array([e_u, v_u, e]))
        gamma_1 = np.linalg.det(np.array([e_u, v_v, e])) + np.linalg.det(np.array([e_v, v_u, e]))
        gamma_2 = np.linalg.det(np.array([e_v, v_v, e]))

        discr = gamma_1 * gamma_1 - 4.0 * gamma_0 * gamma_2
        # here should be exception about the negative discriminant

        # Negative discriminant
        if discr < 0 and abs(discr) > 1e-7:
            print("Discriminant is negative")
            tors[2*i] = np.array([0,0,0])
            tors[2 * i + 1] = np.array([0,0,0])
            cos_tors[i] = -1

        # Zero discriminant
        elif abs(discr) < 1e-7 :
            if discr < 0 : 
                soln = [(-gamma_1 ), (-gamma_1 )]
                sold = (2.0 * gamma_0)
            else:
                soln = [(-gamma_1 - np.sqrt(discr)), (-gamma_1 + np.sqrt(discr))]
                sold = (2.0 * gamma_0)
                     
            # Calculate torsal directions for the current face
            tors1 = soln[0] * v_u + sold * v_v
            tors1 = tors1 / np.linalg.norm(tors1)
            tors[2*i] = tors1
            tors2 = soln[1] * v_u + sold * v_v
            tors2 = tors2 / np.linalg.norm(tors2)
            tors[2 * i + 1] = tors2

            normal_plane_1 = np.cross(tors1, e)
            normal_plane_2 = np.cross(tors2, e)
            
            cos_tors[i] = abs(np.dot(normal_plane_1, normal_plane_2)) / (np.linalg.norm(normal_plane_1) * np.linalg.norm(normal_plane_2))

        # Positive discriminant
        else:            
            soln = [(-gamma_1 - np.sqrt(discr)), (-gamma_1 + np.sqrt(discr))]
            sold = (2.0 * gamma_0)
            
            # Calculate torsal directions for the current face
            tors1 = soln[0] * v_u + sold * v_v
            tors1 = tors1 / np.linalg.norm(tors1)
            tors[2*i] = tors1
            tors2 = soln[1] * v_u + sold * v_v
            tors2 = tors2 / np.linalg.norm(tors2)
            tors[2 * i + 1] = tors2

            normal_plane_1 = np.cross(tors1, e)
            normal_plane_2 = np.cross(tors2, e)
            
            cos_tors[i] = abs(np.dot(normal_plane_1, normal_plane_2)) / (np.linalg.norm(normal_plane_1) * np.linalg.norm(normal_plane_2))

    return tors, cos_tors
def torsal_dir_vec(tv, tf, e_i):
    
    # Get vertices
    vi, vj, vk = tv[tf[:,0]], tv[tf[:,1]], tv[tf[:,2]]  

    # Get directions
    ei, ej, ek = e_i[tf[:,0]], e_i[tf[:,1]], e_i[tf[:,2]]

    # Compute edges
    eij = ej - ei
    eik = ek - ei

    # Compute vectors
    vij = vj - vi
    vik = vk - vi

    # Compute barycenter of directions
    ec = np.sum( e_i[tf], axis = 1)/3

    # Compute barycenter of vertices
    barycenters = (vi + vj+ vk)/3

    # Cross products
    vijXec = np.cross(vij, ec)
    vikXec = np.cross(vik, ec)

    # Compute coefficients
    g0 = np.sum(eij*vijXec, axis=1)
    g1 = np.sum(eij*vikXec, axis=1) + np.sum(eik*vijXec, axis=1)
    g2 = np.sum(eik*vikXec, axis=1)

    # Compute discriminant
    disc = g1**2 - 4*g0*g2
    

    # Get indices with negative discriminant, positive discriminant and small discriminant (i.e almost zero)
    negative_disc = np.where((disc < 0) & (np.abs(disc) > 1e-7))
    zero_disc = np.where((np.abs(disc) <= 1e-7))
    pos_disc = np.where((disc > 0) & (np.abs(disc) > 1e-7))

    # check disjoint sets
    assert len(np.intersect1d(negative_disc, zero_disc)) == 0
    assert len(np.intersect1d(negative_disc, pos_disc)) == 0
    assert len(np.intersect1d(zero_disc, pos_disc)) == 0

    # Init torsal directions
    t1 = np.zeros((len(disc), 3))
    t2 = np.zeros((len(disc), 3))

    # For discriminant near zero < 1e-7
    t1[zero_disc] = (-g1[zero_disc] + np.sqrt(abs(disc[zero_disc])))[:, None] * vij[zero_disc] + (2 * g0[zero_disc])[:, None] * vik[zero_disc]
    t2[zero_disc] = (-g1[zero_disc] - np.sqrt(abs(disc[zero_disc])))[:, None] * vij[zero_disc] + (2 * g0[zero_disc])[:, None] * vik[zero_disc]

    # For negative discriminant with absolute value > 1e-5
    t1[negative_disc] = np.zeros((len(negative_disc), 3))
    t2[negative_disc] = np.zeros((len(negative_disc), 3))

    # For positive discriminant > 1e-5
    t1[pos_disc] = (-g1[pos_disc] + np.sqrt(disc[pos_disc]))[:, None] * vij[pos_disc] + (2 * g0[pos_disc])[:, None] * vik[pos_disc]
    t2[pos_disc] = (-g1[pos_disc] - np.sqrt(disc[pos_disc]))[:, None] * vij[pos_disc] + (2 * g0[pos_disc])[:, None] * vik[pos_disc]

    # Normalize
    t1[zero_disc] /= np.linalg.norm(t1[zero_disc], axis=1)[:, None]
    t2[zero_disc] /= np.linalg.norm(t2[zero_disc], axis=1)[:, None]

    t1[pos_disc] /= np.linalg.norm(t1[pos_disc], axis=1)[:, None]
    t2[pos_disc] /= np.linalg.norm(t2[pos_disc], axis=1)[:, None]

    # Init cosines vectors
    cos_tors = np.zeros(len(disc))

    # Compute cross products
    t1Xec = np.cross(t1, ec)
    t2Xec = np.cross(t2, ec)

    # Get indices of nonzero vectors
    nonzeroInd = np.where(np.linalg.norm(t1Xec, axis=1) * np.linalg.norm(t2Xec, axis=1) >= 1e-7)

    # Compute cosines for nonzero vectors
    cos_tors[nonzeroInd] = np.sum( abs(t1Xec[nonzeroInd] * t2Xec[nonzeroInd]), axis=1) / (np.linalg.norm(t1Xec[nonzeroInd], axis=1) * np.linalg.norm(t2Xec[nonzeroInd], axis=1))

    # Compute cosines for zero vectors
    cos_tors[np.where(np.linalg.norm(t1Xec, axis=1) * np.linalg.norm(t2Xec, axis=1) < 1e-7)] = -1

    # print(f"f : 214 \n  disc : {disc[214]} \n ei : {ei[214]} \n ej : {ej[214]} \n ek : {ek[214]} \n eij : {eij[214]} \n eik : {eik[214]} \n vij : {vij[214]} \n vik : {vik[214]} \n g0 : {g0[214]} \n g1 : {g1[214]} \n g2 : {g2[214]} \n t1 : {t1[214]} \n t2 : {t2[214]} \n vijXec : {vijXec[214]} \n vikXec : {vikXec[214]} \n ec : {ec[214]} \n barycenters : {barycenters[214]} \n cos_tors : {cos_tors[214]}")

    return  barycenters, t1, t2, cos_tors


# ====================== Polyscope Functions =================

def draw_polygon(vertices, name="_"):
    """
        Register a polygon as a surface
    """
    ps.register_surface_mesh(name, vertices, [np.arange(len(vertices))[:, None]])
    
def draw_plane(p0, n, size=(1,1), name="_"):
    """
        Register a plane as a surface
    """
    aux = n + np.array([1,0,0])

    v1 = unit(orth_proj(aux, n))

    print(v1@aux)
    v2 = unit(np.cross(n, v1))

    v1 *= size[0]
    v2 *= size[1]

    vertices = np.array([p0 + v1 + v2, p0 + v1 - v2, p0 - v1 - v2, p0 - v1 + v2])

    ps.register_surface_mesh(name, vertices, [np.arange(len(vertices))[:, None]], color=(0.1, 0.1, 0.1), transparency=0.2)


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
    mesh.add_vector_quantity(name+"_vec1" ,    vec1, defined_on ='faces', enabled=True, radius=rad, length=size, color=col)
    mesh.add_vector_quantity(name+"_-vec1",   -vec1, defined_on ='faces', enabled=True, radius=rad, length=size, color=col)
    mesh.add_vector_quantity(name+"_vec2" ,    vec2, defined_on ='faces', enabled=True, radius=rad, length=size, color=col)
    mesh.add_vector_quantity(name+"_-vec2",   -vec2, defined_on ='faces', enabled=True, radius=rad, length=size, color=col)

def solve_torsal(vi, vj, vk, vvi, vvj, vvk) :

    # Get edges
    vij = vj - vi 
    vik = vk - vi

    ei = vvi - vi 
    ej = vvj - vj
    ek = vvk - vk

    eij = ej - ei 
    eik = ek - ei
    

    ec = (vvi + vvj + vvk)/3 - (vi + vj + vk)/3

    vijxec = np.cross(vij, ec)
    vikxec = np.cross(vik, ec)

    # g0 
    g0 = np.sum(eij*vijxec, axis=1)

    # g1
    g1 = np.sum(eij*vikxec, axis=1) + np.sum(eik*vijxec, axis=1)

    # g2
    g2 = np.sum(eik*vikxec, axis=1)


    disc = g1**2 - 4*g0*g2 

    t1 = np.zeros_like(vij)
    t2 = np.zeros_like(vij)

    a1 = np.zeros(len(vij))
    a2 = np.zeros(len(vij))
    b1 = np.zeros(len(vij))

    # indices disc >0 
    idx = np.where(disc >= 0)[0]

    a1[idx] = (-g1[idx] + np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))
    a2[idx] = (-g1[idx] - np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))
    b1[idx] = 2*g0[idx]

    # sol
    t1[idx] = (-g1[idx] + np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))[:, None]*vij[idx] + 2*g0[idx,None]*vik[idx]
    t2[idx] = (-g1[idx] - np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))[:, None]*vij[idx] + 2*g0[idx,None]*vik[idx]

    # Normalize
    t1[idx] /= np.linalg.norm(t1[idx], axis=1)[:, None]
    t2[idx] /= np.linalg.norm(t2[idx], axis=1)[:, None]

    # # indices disc < 0
    # idx = np.where(disc < 0)[0]
    # a1[idx] = 1
    # a2[idx] = ( - vec_dot(vij[idx],vik[idx]) -  vec_dot(vik[idx],vik[idx]) )/( vec_dot(vij[idx],vij[idx]) + vec_dot(vij[idx], vik[idx]))
    # b1[idx] = 1

    # # sol
    # t1[idx] = a1[idx,None]*vij[idx] + vik[idx]
    # t2[idx] = a2[idx,None]*vij[idx] + vik[idx]

    # # Normalize
    # t1[idx] /= np.linalg.norm(t1[idx], axis=1)[:, None]
    # t2[idx] /= np.linalg.norm(t2[idx], axis=1)[:, None]

    return t1, t2, a1, a2, b1 


def vv_second(vvi, vvj, vvk, f, numV):

    vv = np.zeros((numV, 3))

    for i in range(len(f)):      
        vv[f[i,0]] = vvi[i]
        vv[f[i,1]] = vvj[i]
        vv[f[i,2]] = vvk[i]

    return vv


def init_test_data(data):
    # # Define paths
    dir_path = os.getcwd()
    data_path = dir_path+"/approximation/data/" # data path

    # Data of interest
    k = data

    # Load M mesh (centers of sphere mesh)
    mv, mf = igl.read_triangle_mesh( os.path.join(data_path ,"centers.obj") ) 

    # Load test mesh
    tv, tf = igl.read_triangle_mesh(os.path.join(data_path,  "test_remeshed_"+str(k)+".obj"))

    # Create dual mesh
    tmesh = Mesh()
    tmesh.make_mesh(tv,tf)

    # Get inner vertices
    inner_vertices = tmesh.inner_vertices()

    # Get vertex normals for test mesh
    e_i = igl.per_vertex_normals(tv, tf)

    # Fix normal directions
    signs = np.sign(np.sum(e_i * ([0,0,1]), axis=1))
    e_i = e_i * signs[:, None]

    # Compute circumcenters and axis vectors for each triangle
    p1, p2, p3 = tv[tf[:, 0]], tv[tf[:, 1]], tv[tf[:, 2]]

    ct, _, nt = circle_3pts(p1, p2, p3)

    # Dual topology 
    dual_tf = tmesh.vertex_ring_faces_list()

    dual_top = tmesh.dual_top()

    # Create hexagonal mesh                            
    h_pts = np.empty((len(tf), 3), dtype=np.float64)
    
    li = np.zeros(len(tf), dtype=np.float64)
    
    center = vd.Mesh((mv, mf), alpha = 0.9, c=[0.4, 0.4, 0.81])

    # Intersect circumcircle axis with center mesh
    for i in range(len(tf)):
        # Get points on circumcircle axis
        p0  = ct[i] - 10*nt[i]
        p1  = ct[i] + 10*nt[i]

        # Get intersection points
        h_pts[i,:] = np.array(center.intersect_with_line(p0, p1)[0])

        # Set li 
        li[i] = np.linalg.norm(h_pts[i] - ct[i])

    # Get radius of spheres
    r = np.linalg.norm(h_pts - tv[tf[:,0]], axis=1)

    return tv, tf, ct, nt, li, inner_vertices, e_i, dual_tf, dual_top, r 


def visualize_data(data):
        # # Define paths
    dir_path = os.getcwd()
    data_path = dir_path+"/approximation/data/" # data path

    # Data of interest
    k = data

    # Load M mesh (centers of sphere mesh)
    mv, mf = igl.read_triangle_mesh( os.path.join(data_path ,"centers.obj") ) 

    # Load test mesh
    tv, tf = igl.read_triangle_mesh(os.path.join(data_path,  "test_remeshed_"+str(k)+".obj"))

    # Create dual mesh
    tmesh = Mesh()
    tmesh.make_mesh(tv,tf)

    # Get inner vertices
    inner_vertices = tmesh.inner_vertices()

    # Get vertex normals for test mesh
    e_i = igl.per_vertex_normals(tv, tf)

    # Fix normal directions
    signs = np.sign(np.sum(e_i * ([0,0,1]), axis=1))
    e_i = e_i * signs[:, None]

    # Compute circumcenters and axis vectors for each triangle
    p1, p2, p3 = tv[tf[:, 0]], tv[tf[:, 1]], tv[tf[:, 2]]

    ct, _, nt = circle_3pts(p1, p2, p3)

    # Dual topology 
    dual_tf = tmesh.vertex_ring_faces_list()

    # Create hexagonal mesh                            
    h_pts = np.empty((len(tf), 3), dtype=np.float64)
    center = vd.Mesh((mv, mf), alpha = 0.9, c=[0.4, 0.4, 0.81])

    ps.init()

    ps.remove_all_structures()

    v_mesh = ps.register_surface_mesh("test", tv, tf)

    c_mesh = ps.register_surface_mesh("centers", mv, mf)

    v_mesh.add_vector_quantity("n_t", nt, defined_on = "faces", enabled=True, length=1.5, color=(0.1, 0.1, 0.1))
    v_mesh.add_vector_quantity("e", e_i, defined_on = "vertices", enabled=True, length=1.5, color=(0.1, 0.1, 0.1))

    ps.show()


def compute_disc(tv, tf, e_i):

    # # Compute the edge vectors per each face
    vi, vj, vk = tv[tf[:,0]], tv[tf[:,1]], tv[tf[:,2]]

    # # Compute the edge vectors per each face
    vij = vj - vi
    vik = vk - vi

    # Set up X 
    eij = e_i[tf[:,1]] - e_i[tf[:,0]]
    eik = e_i[tf[:,2]] - e_i[tf[:,0]]

    ec = np.sum( e_i[tf], axis = 1) / 3

    # A = [vij, eik, ec] + [eij, vik, ec], where [ , , ] denotes determinant
    # A = gamma11 +  gamma12
    eikXec = np.cross(eik, ec)
    vikXec = np.cross(vik, ec)

    det1 = np.sum(vij*eikXec, axis=1)
    det2 = np.sum(eij*vikXec, axis=1)

    # b = [eij, eik, ec]  c = [vij, vik, ec]

    gamma0 = np.sum(eij*eikXec, axis=1)
    gamma2 = np.sum(vij*vikXec, axis=1)

    A = det1 + det2 

    return A, A**2 - 4*gamma0*gamma2

def unormalize_dir(h_pts, dual, inner_vertices, tv, e_i, rad):
    """Input 
        h_pts: sphere centers
        e_i: edge directions normalized
        rad: sphere radii
        Ouput:
        le: unormalized edge directions

    """
    le = np.ones_like(e_i)
    for i in range(len(inner_vertices)):
        # Get dual faces index
        idx = inner_vertices[i]

        # Get dual face
        f = dual[idx]

        # Get sphere centers
        p = h_pts[f]

        # Get edge direction/'
        e = e_i[idx]

        # Get radius
        r = rad[idx]

        # angle e_i with the direction to the center
        theta = np.arccos(np.sum(e*(p - tv[idx]), axis=1))

        print(theta)

        # Get the lambda
        le[idx] = 2*r*np.cos(theta)

    return le

def planarity_check(nt1, t1, tt1, ec):

    t1 = unit(t1)
    tt1 = unit(tt1)
    ec = unit(ec)
    t1_tt1 = np.cross(t1, tt1)
    # Check planarity
    planar = abs(vec_dot(t1_tt1, ec))

    #planar = abs(vec_dot(nt1, t1)) + abs(vec_dot(nt1, tt1)) + abs(vec_dot(nt1, ec))

    # # Replace nan with 0
    planar[np.where(np.isnan(planar))] = 1

    return planar

def compute_torsal_angles(t1, t2, ec):

    # Compute nt1 
    nt1 = np.cross(t1, ec)
    nt1 /= np.linalg.norm(nt1, axis=1)[:, None]

    # Compute nt2
    nt2 = np.cross(t2, ec)
    nt2 /= np.linalg.norm(nt2, axis=1)[:, None]

    # Compute torsal angles
    torsal_angles = np.arccos(np.sum(nt1*nt2, axis=1))

    return torsal_angles, nt1, nt2