import os 
import sys

# Add hananLab to path
#path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
path = os.path.dirname(os.getcwd())
sys.path.append(path)

import igl
#import meshplot as mp
import polyscope as ps
import vedo as vd
import numpy as np
import os
import matplotlib.pyplot as plt
from geometry.mesh import Mesh
from geometry.utils import *
from optimization.Torsal import Torsal
from optimization.Torsal_angle import Torsal_angle
from optimization.Optimizer import Optimizer
from optimization.LineCong import LineCong
from optimization.Sphere_angle import Sphere_angle
from optimization.LineCong_Fairness import LineCong_Fair

# Define paths
dir_path = os.getcwd()
data_path = dir_path+"/approximation/data/" # data path
out_path = dir_path+"/approximation/outputs/" # output path
math_path = dir_path+"/approximation/mathematica/" # mathematica path


# Iterations
It = 100
Data = 1

Weights = {"linecong": 1, 
           "torsal": 1, 
           "torsal_angle": 1, 
           "sphere_angle": 0,
            "lc_fairness": 1
           }

# Global Torsal variables
T1 = None 
T2 = None 

def init_test_data(data):
    # Define paths
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

    # Get inner edges
    inner_edges = tmesh.inner_edges()

    # Get vertices of inner edges
    ev1, ev2 = tmesh.edge_vertices()
    iv1 = ev1[inner_edges]
    iv2 = ev2[inner_edges]

    # Get faces of inner edges
    f1, f2 = tmesh.edge_faces()

    # Get vertices indices of inner edges
    vf1 = tf[f1[inner_edges]]
    vf2 = tf[f2[inner_edges]]

    vf = []
    # Delete vertices of inner edges in each vf1 and vf2
    for i in range(len(inner_edges)):
        valid_vertices = [] 
        for j in range(len(vf1[i])):
            if vf1[i][j] not in np.array([iv1[i], iv2[i]]):
                valid_vertices.append(vf1[i][j])
        for j in range(len(vf2[i])):
            if vf2[i][j] not in np.array([iv1[i], iv2[i]]):
                valid_vertices.append(vf2[i][j])

        vf.append(valid_vertices)

    ie_f = np.array([f1, f2]).T

    vertex_adj = tmesh.vertex_adjacency_list()
    
    return tv, tf, ct, nt, li, inner_vertices, e_i, dual_tf, inner_edges, vertex_adj



def run_optimization(it, data):
    
    # Init data 
    tv, tf, _, nt, df, inner_vertices, e_i, dual_tf, inner_edges, vertex_adj = init_test_data(data)

    # Correct normals
    nt = - nt

    # Get number of vertices, faces and inner edges
    nV = len(tv)
    nF = len(tf)
    nIE = len(inner_edges)

    

    # Define variable indices
    var_idx = {     "e"  : np.arange( 0            , 3*nV), 
                    "a1" : np.arange( 3*nV        , 3*nV +    nF),
                    "b1" : np.arange( 3*nV +    nF, 3*nV +  2*nF),
                    "nt1": np.arange( 3*nV +  2*nF, 3*nV +  5*nF),
                    "a2" : np.arange( 3*nV +  5*nF, 3*nV +  6*nF),
                    "b2" : np.arange( 3*nV +  6*nF, 3*nV +  7*nF),
                    "nt2": np.arange( 3*nV +  7*nF, 3*nV + 10*nF),
                    "df" : np.arange( 3*nV + 10*nF, 3*nV + 11*nF),
                    "u"  : np.arange( 3*nV + 11*nF, 3*nV + 12*nF)
#                    "bb1" : np.arange( 3*nV + 12*nF, 3*nV + 12*nF + 3*nIE),
#                    "bb2" : np.arange( 3*nV + 12*nF + 3*nIE, 3*nV + 12*nF + 6*nIE)
            }


    # Compute the circumcircle
    bf, _, _ = circle_3pts(tv[tf[:,0]], tv[tf[:,1]], tv[tf[:,2]])

    # Init X 
    X = np.zeros(sum(len(arr) for arr in var_idx.values()))

    X[var_idx["e"]]  = e_i.flatten() 
    X[var_idx["df"]] = df
    X[var_idx["u"]]  = 0.1
    


    # Init LineCong
    linecong = LineCong()
    linecong.initialize_constraint(X, var_idx, len(tv), bf, nt, len(tf), dual_tf, inner_vertices)
    linecong.set_weigth(Weights["linecong"])

    # Init Torsal 
    torsal = Torsal()
    torsal.initialize_constraint(X, var_idx, tv, tf, bf, nt)
    torsal.set_weigth(Weights["torsal"])

    # Init Torsal angle
    tang = Torsal_angle()
    tang.initialize_constraint(X, var_idx, tv, tf)
    tang.set_weigth(Weights["torsal_angle"])

    # Init LineCong Fairness
    lc_fair = LineCong_Fair()
    lc_fair.initialize_constraint(X, var_idx, vertex_adj, inner_vertices)
    lc_fair.set_weigth(Weights["lc_fairness"])

    # Sphere angle
    sph_ang = Sphere_angle()
    sph_ang.initialize_constraint(X, var_idx, tv, tf, bf, nt) 
    sph_ang.set_weigth(Weights["sphere_angle"])



    # Init optimizer
    optimizer = Optimizer()
    optimizer.initialize_optimizer(X, var_idx, "LM", 0.5)

    a1, b1 = torsal.uncurry_X(X, "a1", "b1")

    T1 = unit(torsal.compute_t(a1, b1))

    a2, b2 = torsal.uncurry_X(X, "a2", "b2")

    T2 = unit(torsal.compute_t(a2, b2))


    for _ in range(it):
        
        optimizer.unitize_variable("nt1", 3)
        optimizer.unitize_variable("nt2", 3)
        optimizer.unitize_variable("e", 3)

        optimizer.get_gradients(linecong)
        optimizer.get_gradients(lc_fair)
        optimizer.get_gradients(tang)
        optimizer.get_gradients(torsal)
        

        optimizer.optimize()
   
    visualization(torsal, optimizer, tv, tf, T1, T2)


def cross_field_error(t1, t2, t1a, t2a):
    """ Function to measure the cross field error between two cross fields
    """

    t1 = unit(t1)
    t2 = unit(t2)
    t1a = unit(t1a)
    t2a = unit(t2a)

    error = np.zeros(len(t1))
    for i in range(len(t1)):
        # check nan values 
     
        if abs(t1[i]@t1a[i]) > abs(t1[i]@t2a[i]):
            error[i] = 1 - (abs(t1[i]@t1a[i]) + abs(t2[i]@t2a[i]))/2 
        else:
            error[i] = 1 - (abs(t1[i]@t2a[i]) + abs(t2[i]@t1a[i]))/2
    
    return error


   
def visualization(constraint, optimizer, tv, tf, T1, T2):

    # Get variables
    e, a1, b1, nt1, a2, b2, nt2, di = constraint.uncurry_X(optimizer.X, "e", "a1", "b1", "nt1", "a2", "b2", "nt2", "df")

    # Reshape variables
    e   = unit(e.reshape(-1,3))
    nt1 = unit(nt1.reshape(-1,3))
    nt2 = unit(nt2.reshape(-1,3))

    # Get vertices on faces
    vi, vj, vk = tv[tf[:,0]], tv[tf[:,1]], tv[tf[:,2]]

    # Line congruence per face
    ei, ej, ek = e[tf[:,0]], e[tf[:,1]], e[tf[:,2]]

    # Second envelope
    vvi, vvj, vvk, _, _, _ = constraint.compute_second_env(di, e, tf)

    # Second envelope edges
    vvij = vvj - vvi
    vvik = vvk - vvi

    # Second envelope vertices
    vv = vv_second(vvi, vvj, vvk, tf, len(tv))

    ptvv = np.vstack((vvi, vvj, vvk))
    
    # Compute torsal directions 
    at1, at2, aa1, aa2, bb = solve_torsal(tv[tf[:,0]], tv[tf[:,1]], tv[tf[:,2]], vvi, vvj, vvk)

    # Compute torsal directions
    t1 = unit(constraint.compute_t(a1, b1))
    t2 = unit(constraint.compute_t(a2, b2))

    # print("nt1 norm", np.sum(np.linalg.norm(nt1, axis=1))/len(nt1))
    # print("nt2 norm", np.sum(np.linalg.norm(nt2, axis=1))/len(nt2))


    # print("t1 norms:", np.sum(np.linalg.norm(t1, axis=1))/len(t1))
    # print("t2 norms:", np.sum(np.linalg.norm(t2, axis=1))/len(t2))

    # Compute torsal directions on second envelope
    tt1, _, _ = constraint.compute_tt(a1, b1, vvi, vvj, vvk)
    tt2, _, _ = constraint.compute_tt(a2, b2, vvi, vvj, vvk)
    
    tt1 = unit(tt1)
    tt2 = unit(tt2)
    # # Compute norms
    # print("tt1 norms:", np.sum(np.linalg.norm(tt1, axis=1))/len(tt1))
    # print("tt2 norms:", np.sum(np.linalg.norm(tt2, axis=1))/len(tt2))
    
    # Barycenter on both envelopes
    vc = (vi + vj + vk)/3
    vvc = (vvi + vvj + vvk)/3
    ec = vvc - vc
   
    # Compute analytic torsal directions
    att1 = unit(aa1[:, None]*vvij + bb[:, None]*vvik)
    att2 = unit(aa2[:, None]*vvij + bb[:, None]*vvik)

    att1 /= np.linalg.norm(att1, axis=1)[:, None]
    att2 /= np.linalg.norm(att2, axis=1)[:, None]
    
  
    # normal planes to att
    ant1 = unit(np.cross(at1, ec))
    ant2 = unit(np.cross(at2, ec))

    # Compute planarity
    planar_t1 = planarity_check(nt1, t1, tt1, ec)
    planar_t2 = planarity_check(nt2, t2, tt2, ec)

    aplanar_t1 = planarity_check(ant1, at1, att1, ec)
    aplanar_t2 = planarity_check(ant2, at2, att2, ec)


    planarity = (planar_t1 + planar_t2)/2
    

    analytic_planar = (aplanar_t1 + aplanar_t2)/2
    

    # Cross field Error
    cf_error = cross_field_error(t1, t2, at1, at2)

    # Angle nt1 nt2
    angle = np.arccos(abs(vec_dot(nt1, nt2)))*180/np.pi

    # Visualization
    ps.init()
    ps.remove_all_structures()
    
    #ps.register_point_cloud("Points", ptvv, radius=0.0005, enabled=True, color=(1.0, 0.0, 0.0))
    # Create mesh
    triangle = ps.register_surface_mesh("T1", tv, tf)
    triangle2 = ps.register_surface_mesh("T2", vv, tf)
    triangle.add_scalar_quantity("Planarity", planarity, defined_on='faces', enabled=True, cmap="viridis")
    triangle.add_scalar_quantity("Analytic Planarity", analytic_planar, defined_on='faces', enabled=True, cmap="viridis")
    triangle.add_scalar_quantity("Angle Torsal Planes", angle, defined_on='faces', enabled=True, cmap="viridis")
    triangle.add_vector_quantity("LC", e, defined_on='vertices', enabled=True, radius=0.001, length=2.0, color=(0.0, 1.0, 0.0))

    
    #add_cross_field(triangle, "Planes normals", nt1, nt2, 0.0002, 0.007, (0.0, 0.0, 1.0))
    add_cross_field(triangle, "Torsal Directions", t1, t2, 0.0003, 0.008, (1.0, 1.0, 1.0))

    # l = 0.08
    # for _ in range(20):
    #     f = np.random.randint(0, len(tf))
    #     ps.register_surface_mesh("Torsal plane "+str(f), np.array([vc[f]- l*t1[f], vc[f]+ l*t1[f], vc[f] + l*tt1[f] + l*ec[f], vc[f] - l*tt1[f] + l*ec[f] ]), [[0,1,2,3]], color=(0.2, 0.2, 0.2), transparency=0.5)
    #     ps.register_surface_mesh("Torsal plane 2 "+str(f), np.array([vc[f]- l*t2[f], vc[f]+ l*t2[f], vc[f] + l*tt2[f] + l*ec[f], vc[f] - l*tt2[f] + l*ec[f] ]), [[0,1,2,3]], color=(0.2, 0.2, 0.2), transparency=0.5)
    #add_cross_field(triangle, "Init Torsal Directions", T1, T2, 0.0003, 0.007, (0.2, 0.0, 0.8))
    #add_cross_field(triangle, "Analytical Torsal Directions", at1, at2, 0.0002, 0.008, (0.0, 1.0, 0.0))

    ps.show()


def main():

    run_optimization(It, Data)


main()