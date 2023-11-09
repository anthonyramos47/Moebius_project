import numpy as np

def unit(v):
    """normalize a list of vectors v
    """

    return v/np.linalg.norm(v, axis=1)[:, None]


def vec_dot(v1, v2):
    """dot product between two lists of vectors v1, v2
    """
    return np.sum(v1*v2, axis=1)

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