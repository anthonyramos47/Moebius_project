import numpy as np
import os
import polyscope as ps
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


def fit_sphere(pts, sph_c, sph_r):

    # Initial guess
    init = np.zeros(4)
    init[:3] = np.mean(pts, axis=0)
    init[3] = np.mean(np.linalg.norm(pts - init[:3], axis=1))

    # Perform the optimization
    result = minimize(fit_sphere_energy, init, args=(pts))

    return result.x[:3], result.x[3]


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

# def find_initial_torsal_th_phi(t1, t2, vij, vik):
#     """ Function to find the initial torsal directions parameters
#     Input:
#         t: Torsal direction
#         vij: edge vector
#         vik: edge vector
#     """
    
#     theta = np.zeros(len(t1))
#     phi   = np.zeros(len(t1))

#     alpha = np.zeros(len(t1)) # alpha = theta + phi

#     for i in range(len(t1)):
#         # Compute theta
#         theta[i]   = find_angles(0, t1[i], vij[i], vik[i])

#         if t1[i]@unit(np.cos(theta[i])*vij[i] + np.sin(theta[i])*vik[i]) < 0.8:
#             print( t1[i]@unit(np.cos(theta[i])*vij[i] + np.sin(theta[i])*vik[i]))

#         alpha[i]   = find_angles(0, t2[i], vij[i], vik[i])

#         if t2[i]@unit(np.cos(alpha[i])*vij[i] + np.sin(alpha[i])*vik[i]) < 0.8:
#             print( t2[i]@unit(np.cos(alpha[i])*vij[i] + np.sin(alpha[i])*vik[i]))

#         # Compute phi
#         phi[i] = alpha[i] - theta[i]
#         if t2[i]@unit(np.cos(theta[i] + phi[i])*vij[i] + np.sin(theta[i] + phi[i])*vik[i]) < 0.8:
#             print( t2[i]@unit(np.cos(theta[i] + phi[i])*vij[i] + np.sin(theta[i] + phi[i])*vik[i]))
    

#     return theta, phi, alpha    

def solve_torsal(vi, vj, vk, ei, ej, ek) :
    """ Function to solve the torsal directions analytically
    Input:
        vi, vj, vk: vertices
        vvi, vvj, vvk: second envelope vertices
    """

    # Get edges
    vij = vj - vi 
    vik = vk - vi

    eij = ej - ei 
    eik = ek - ei
    

    ec = (ei + ej + ek)/3

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

    # Valid
    valid = np.zeros(len(vij))

    # indices disc >0 
    idx = np.where(disc >= 0)[0]

    a1[idx] = (-g1[idx] + np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))
    a2[idx] = (-g1[idx] - np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))
    b1[idx] = 2*g0[idx]

    # sol
    t1[idx] = (-g1[idx] + np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))[:, None]*vij[idx] + 2*g0[idx,None]*vik[idx]
    t2[idx] = (-g1[idx] - np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))[:, None]*vij[idx] + 2*g0[idx,None]*vik[idx]

    # Normalize
    t1[idx] = unit(t1[idx])
    t2[idx] = unit(t2[idx])

    # Put 1 on valid disc
    valid[idx] = 1

    # For disc < 0 we approximate the solution
    app_idx = np.where(disc < 0)[0]
    for i in app_idx:
        a1[i] = approximate_torsal(100, g0[i], g1[i], g2[i])
        a2[i] = approximate_torsal(-100, g0[i], g1[i], g2[i])
        b1[i] = 1

        t1[i] = unit(a1[i]*vij[i] + b1[i]*vik[i])

        t2[i] = unit(a2[i]*vij[i] + b1[i]*vik[i])

    return t1, t2, a1, a2, b1, valid


def vv_second(vvi, vvj, vvk, f, numV):
    """ Compute second envelope 
    """

    vv = np.zeros((numV, 3))
    nv = np.zeros(numV)

    for i in range(len(f)):      
        vv[f[i,0]] += vvi[i]
        vv[f[i,1]] += vvj[i]
        vv[f[i,2]] += vvk[i]

        nv[f[i,0]] += 1
        nv[f[i,1]] += 1
        nv[f[i,2]] += 1
    
    vv /= nv[:, None]

    return vv



def compute_disc(tv, tf, e_i):
    """ Function to compute the discriminant of the torsal directions
    Input:
        tv: vertices
        tf: faces
        e_i: edge directions normalized
    """


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

def planarity_check(t1, tt1, ec):
    """ Function to check the planarity of the torsal directions
    Input:
        t1: Torsal direction
        tt1: Second envelope torsal direction
        ec: Lince congruence joining barycenters of the faces
    """

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
    """ Function to compute the torsal angles between two cross fields
    Input:
        t1: Torsal direction
        t2: Second torsal direction
        ec: Lince congruence joining barycenters of the faces
    """

    # Compute nt1 
    nt1 = np.cross(t1, ec)
    nt1 /= np.linalg.norm(nt1, axis=1)[:, None]

    # Compute nt2
    nt2 = np.cross(t2, ec)
    nt2 /= np.linalg.norm(nt2, axis=1)[:, None]

    # Compute torsal angles
    torsal_angles = np.arccos(np.sum(nt1*nt2, axis=1))

    return torsal_angles, nt1, nt2


# ====================== Torsal Approximation Functions =================

def find_angles(init, t, vij, vik):
    # Perform the optimization
    result = minimize(quadratic_equation_angle, init, args=(t, vij, vik))


    return result.x

# Define the quadratic equation to be minimized
def quadratic_equation_angle(x, t, vij, vik):
    l = x
    t_th = np.cos(l)*vij + np.sin(l)*vik
    return (unit(t)@unit(t_th) - 1)**2


def approximate_torsal(lc, lu, lv, du, dv):
    """
    Function to approximate the torsal direction.
    Input:
        lc: Lince congruence joining barycenters of the faces
        lu: Line congruence u direction
        lv: Line congruence v direction
        du: surface u direction
        dv: surface v direction
    """


    # Define a minimization of the function
    # ut^2 [du, lu, lc] + (ut vt) ([du, lv, lc] + [dv, lu, lc]) + vt^2 [dv, lv, lc] = 0
    # variables ut and vt
    def torsal_energy(X, lc, lu, lv, du, dv):
        
        ut, vt = X[0], X[1]
        # Energy
        E = ut**2 *du@np.cross(lu, lc) + (ut * vt) * ( du@np.cross(lv, lc)  + dv@np.cross(lu, lc)) +  vt**2 * dv@np.cross(lv, lc)
        return E**2

    # Define the constraints: ut du + vt dv = 1
    cons = {'type': 'eq', 'fun': lambda X: np.sum((np.linalg.norm((X[0]*du + X[1]*dv),axis=1)**2 - 1))}

    # T1 initial guess
    X0 = np.zeros((len(lc)*2)) 
    X0[::2] = 1
    X0 = X0.reshape(-1, 2)

    # T2 initial guess
    X1 = np.zeros((len(lc)*2)) 
    X1[1::2] = -1
    X1 = X1.reshape(-1, 2)
    
    for i in range(len(X0)):
        res1 = minimize(torsal_energy, X0[i], args=(lc[i], lu[i], lv[i], du[i], dv[i]), constraints=cons)
        res2 = minimize(torsal_energy, X1[i], args=(lc[i], lu[i], lv[i], du[i], dv[i]), constraints=cons)

        X0[i] = res1.x
        X1[i] = res2.x

    return X0, X1

def fit_sphere_energy(init, pts):
    c = init[:3]
    r = init[3]
    return np.sum((np.linalg.norm(pts - c, axis=1)**2 - r**2)**2)

# Define the quadratic equation to be minimized
def quadratic_equation(x, g0, g1, g2):
    l = x
    return l**2 * g0 + l * g1 + g2

# Function to be minimized
def objective_function(x, g0, g1, g2):
    return (quadratic_equation(x, g0, g1, g2))**2


def compute_barycenter_coord_proj(v, vv, inner_edges, l, ie_i, ie_j, ie_k, ie_l):
    # Get vertices
    vi = v[ie_i]
    vj = v[ie_j]
    vk = v[ie_k]
    vl = v[ie_l]

    # Get envelope 2 vertices
    vvi = vv[ie_i]
    vvj = vv[ie_j]
    vvk = vv[ie_k]
    vvl = vv[ie_l]


    vbi = orth_proj(vi, l)
    vbj = orth_proj(vj, l)
    vbk = orth_proj(vk, l)
    vbl = orth_proj(vl, l)

    vvb_i = orth_proj(vvi, l)
    vvb_j = orth_proj(vvj, l)
    vvb_k = orth_proj(vvk, l)
    vvb_l = orth_proj(vvl, l)

    Diff = 0

    # Compute the barycentric coordinates of the vertices 
    for i in range(len(inner_edges)):
        
        u1, u2, u3 = barycentric_coordinates_app(vbi[i], vbj[i], vbk[i], vbl[i])
        
        uu1, uu2, uu3 = barycentric_coordinates_app(vvb_i[i], vvb_j[i], vvb_k[i], vvb_l[i])

        #print(f"u1 : {u1} u2 : {u2} u3 : {u3}\n uu1 : {uu1} uu2 : {uu2} uu3 : {uu3} \n")
        Diff += np.linalg.norm(np.array([u1, u2, u3]) - np.array([uu1, uu2, uu3]))**2

    print(f"Diff : {Diff}")

def residuals(params, x):
    # params contains the center coordinates (cx, cy, cz) and the radius (r)
    cx, cy, cz, r = params
    # Compute distances from each point to the center of the sphere
    distances = np.sqrt((x[:, 0] - cx)**2 + (x[:, 1] - cy)**2 + (x[:, 2] - cz)**2)
    # Compute residuals: differences between distances and the radius
    return distances - r

  
def sphere_initialization(v, f, e):
    """
    Function to find the best fitting spheres to two envelopes.

    Args:
        v (numpy.ndarray): Vertices
        f (numpy.ndarray): Faces
        e (numpy.ndarray): Line congruence
    
    Returns:
        numpy.ndarray: The midpoint of the closest points between the two envelopes.
    """
    
    # Points in second envelope
    vv = v + e 

    # Get vi, vj, vk 
    vi  ,  vj, vk  = v[f[:,0]], v[f[:, 1]], v[f[:, 2]]
    # Get vvi, vvj, vvk 
    vvi , vvj, vvk = vv[f[:,0]], vv[f[:, 1]], vv[f[:, 2]]

    # Get circum circles for both envelopes
    p1,  cr,  v1 = circle_3pts(vi, vj, vk)
    p2, cr2,  v2 = circle_3pts(vvi, vvj, vvk)

    # Compute intersection of circumcenter axis
    l = unit(np.cross(v1,v2))  # Normal direction between lines

    # Compute rejection vector bettwen p2-p1
    rej = (p2-p1) - proj(p2-p1, v1) - proj(p2-p1, l)
    
    # Closest point l2 
    cls2 = p2 - (np.linalg.norm(rej, axis= 1)/vec_dot(p2-p1,v2))[:,None]*v2

    # Closest point to l2
    cls = cls2 - proj(p2-p1, l)

    mid = 0.5*(cls + cls2)

    return mid

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


def torsal_directions(lc, lu, lv, du, dv):
    """
    Function to compute the torsal directions.
    This is going to compute the torsal directions at a point vc on the surface with 
    line lc. We use the determinant of 
    [vt, lt, lc] = 0; vt = ut du + vt dv; lt = ut lu + vt lv
    We will find ut:vt
    Input:
        lc : Line at computation place (np.array)
        lu : Line derivative u (np.array)
        lv : Line derivative v (np.array)
        du : Surface derivative u (np.array)
        dv : Surface derivative v (np.array)
    """
   
    # t = ut/vt
    # We solve the quadratic equation
    # ut^2[vu, lu, lc] + ut vt ([vu, lv, lc] + [dv, lu, lc]) + vt^2 [lv, dv, lc] = 0
    # <=> ut^2 g0 + ut vt g1 + vt^2 g2 = 0
    lc = unit(lc)
    
    g0 = vec_dot(du, np.cross(lu, lc))
    g1 = vec_dot(du, np.cross(lv, lc)) + vec_dot(dv, np.cross(lu, lc))
    g2 = vec_dot(dv, np.cross(lv, lc))

    # Discriminant
    disc = g1**2 - 4*g0*g2

    # Init torsal directions
    ut1 = np.zeros(len(lc))
    vt1  = np.zeros(len(lc))
    vt2 = np.zeros(len(lc))
    ut2 = np.zeros(len(lc))

    # ut/vt =(-g1 +/- sqrt(g1^2 - 4*g0*g2))/2*g0
    # ut = (-g1 +/- sqrt(g1^2 - 4*g0*g2))
    # vt = 2*g0

    # For disc > 0
    idx = np.where(disc > 1e-8)[0]
    ut1[idx] = (-g1[idx] + np.sqrt(disc[idx]))
    ut2[idx] = (-g1[idx] - np.sqrt(disc[idx]))
    vt1[idx]  = vt2[idx] =  2*g0[idx]
    
    # For disc < 0
    idx = np.where(disc <= 1e-8)[0]
    print(f"Disc < 0 : {len(idx)}")
    if idx.size > 0:
        opt_t1, opt_t2 = approximate_torsal(lc[idx], lu[idx], lv[idx], du[idx], dv[idx])

        ut1[idx]  = opt_t1[:,0]
        vt1[idx]  = opt_t1[:,1]

        ut2[idx]  = opt_t2[:,0]
        vt2[idx]  = opt_t2[:,1]


    t1 = ut1[:,None]*du + vt1[:,None]*dv
    t2 = ut2[:,None]*du + vt2[:,None]*dv

    

    ut1 /= np.linalg.norm(t1, axis=1)
    vt1 /= np.linalg.norm(t1, axis=1)

    ut2 /= np.linalg.norm(t2, axis=1)
    vt2 /= np.linalg.norm(t2, axis=1)

    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 /= np.linalg.norm(t2, axis=1)[:, None]
   
    return t1, t2, ut1, vt1, ut2, vt2, idx
    
    
def lc_info_at_grid_points(l):
    """ Function to compute the line congruence information for torsal computations.
    Input:
        l: Line congruence (u,v,3)
        n: Normals direction (u,v,3)
        r_uv: radius of the spheres (u,v,3)
    Return:
        lc: Line congruence at the baricenter of the faces
        lu: Line congruence u direction
        lv: Line congruence v direction
    """
    # Get line congruence at grid points
    l0 = l[:-1,:-1]
    l1 = l[:-1,1:]
    l2 = l[1:,1:]
    l3 = l[1:,:-1]

    lc = (l0 + l1 + l2 + l3)/4

    # lu 
    lu = l2 - l0 
    lv = l1 - l3

    return lc, lu, lv


def flat_array_variables(arr, n=3):
    # Multiply elements by 3
    multiplied = arr * n
    # Create a sequence for each element and reshape for concatenation
    for i in range(1, n):
        multiplied = np.vstack([multiplied, arr * n + i])
    
    # Flatten the array to match the desired output format
    transformed = multiplied.flatten('F')
    return transformed

def torsal_dir_show(baricenter, t1, t2, size=0.005, rad=0.0005,  color=(1,1,1), name=""):

    # Torsal directions t1
    t1_dir_i = baricenter.reshape(-1,3) + size*t1
    t1_dir_f = baricenter.reshape(-1,3) - size*t1

    # Torsal directions t2 
    t2_dir_i = baricenter.reshape(-1,3) + size*t2
    t2_dir_f = baricenter.reshape(-1,3) - size*t2
    
    t2_nodes = np.concatenate((t2_dir_i, t2_dir_f), axis=0)
    t1_nodes = np.concatenate((t1_dir_i, t1_dir_f), axis=0)

    t1_edges = np.array([[i, i + len(t1_dir_i)] for i in range(len(t1_dir_i))])

    t1_net = ps.register_curve_network(name+"t1", 
                                       t1_nodes, 
                                       t1_edges, 
                                       color=color)
    t2_net = ps.register_curve_network(name+"t2", t2_nodes, t1_edges, 
                                        color=color)
    t1_net.set_radius(rad, relative=False) 
    t2_net.set_radius(rad, relative=False)

def save_torsal(baricenter, t1, t2, size=0.005, rad=0.0005, path=""):

    # Get the last name of name
    name = path.split('/')[-1]

    # Torsal directions t1
    t1_dir_i = baricenter.reshape(-1,3) 
    t1_dir_f = baricenter.reshape(-1,3) + size*t1


    # Torsal directions t2 
    t2_dir_i = baricenter.reshape(-1,3) 
    t2_dir_f = baricenter.reshape(-1,3) + size*t2

    
    
    t2_nodes = np.concatenate((t2_dir_i, t2_dir_f), axis=0)
    t1_nodes = np.concatenate((t1_dir_i, t1_dir_f), axis=0)

    t1_edges = np.array([[i, i + len(t1_dir_i)] for i in range(len(t1_dir_i))])

    # Check if path exist
    if not os.path.exists(path):
        os.makedirs(path)


    # Create two files for each vector field
    file1 = os.path.join(path, name+'_TD1.obj')
    file2 = os.path.join(path, name+'_TD2.obj')


    # Write the first vector field
    with open(file1, 'w') as f:
        for v in t1_nodes:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for l in t1_edges:
            f.write('l {} {}\n'.format(l[0]+1, l[1]+1))

    # Write the second vector field
    with open(file2, 'w') as f:
        for v in t2_nodes:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for l in t1_edges:
            f.write('l {} {}\n'.format(l[0]+1, l[1]+1))
        

def get_torsal_Mesh(V, F, L):
    """ Function that compute the torsal directions given a polyhedral surface
    with a line congruence per vertex
    Input:
        V: Vertices
        F: Faces
        L: Line congruence
    Output:
        t1: Torsal direction 1
        t2: Torsal direction 2
    """

    l0, l1, l2 = L[F[:, 0]], L[F[:, 1]], L[F[:, 2]]

    lu = l1 - l0
    lv = l2 - l0

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    du = v1 - v0
    dv = v2 - v0

    lc = (l0 + l1 + l2)/3
    vc = (v0 + v1 + v2)/3

    t1, t2, _, _, _, _, _  = torsal_directions(lc, lu, lv, du, dv)

    return t1, t2, vc

def triangulate_quads(quads):
    """
    Triangulate a list of quads into triangles.

    Args:
    - quads (list of list of int): List of quads, where each quad is a list of four vertex indices.

    Returns:
    - list of list of int: List of triangles, where each triangle is a list of three vertex indices.
    """
    triangles = []
    for quad in quads:
        # Ensure the quad has exactly 4 vertices
        if len(quad) == 4:
            # First triangle from first, second, and third vertices
            triangles.append([quad[0], quad[1], quad[2]])
            # Second triangle from first, third, and fourth vertices
            triangles.append([quad[0], quad[2], quad[3]])
        else:
            print("Error: Quad does not have exactly 4 vertices.", quad)
    return triangles



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

    return closest_points, np.hstack((1-u-v, u, v))

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

def np_pop(arr, idx):
    # Step 1: Retrieve the element at the given index
    element = arr[idx]

    # Step 2: Create a new array without the element at 'idx'
    new_arr = np.delete(arr, idx, axis=0)

    # Return the popped element and the new array
    return element, new_arr

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