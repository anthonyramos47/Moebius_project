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

# Define paths
dir_path = os.getcwd()
data_path = dir_path+"/approximation/data/" # data path
out_path = dir_path+"/approximation/outputs/" # output path
math_path = dir_path+"/approximation/mathematica/" # mathematica path


# Iterations
It = 100
Data = 1


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

    return tv, tf, ct, nt, li, inner_vertices, e_i, dual_tf, dual_top, r 



def run_optimization(it, data):
    
    # Init data 
    tv, tf, bt, nt, df, inner_vertices, e_i, dual_tf, dual_top, r = init_test_data(data)

    # Correct normals
    nt = - nt

    # Get number of vertices and faces
    nV = len(tv)
    nF = len(tf)

    # Define variable indices
    var_idx = {     "e"  : np.arange( 0            , 3*nV), 
                    "a1" : np.arange( 3*nV        , 3*nV +    nF),
                    "b1" : np.arange( 3*nV +    nF, 3*nV +  2*nF),
                    "nt1": np.arange( 3*nV +  2*nF, 3*nV +  5*nF),
                    "a2" : np.arange( 3*nV +  5*nF, 3*nV +  6*nF),
                    "b2" : np.arange( 3*nV +  6*nF, 3*nV +  7*nF),
                    "nt2": np.arange( 3*nV +  7*nF, 3*nV + 10*nF),
                    "df" : np.arange( 3*nV + 10*nF, 3*nV + 11*nF),
                    "u" : np.arange( 3*nV + 11*nF, 3*nV + 12*nF),
                    "v"  : np.arange( 3*nV + 12*nF, 3*nV + 13*nF),
            }


    # Compute the circumcircle
    bf, _, ncf = circle_3pts(tv[tf[:,0]], tv[tf[:,1]], tv[tf[:,2]])

    # Init X 
    X = np.zeros(3*len(tv) + 13*len(tf))

    X[var_idx["e"]] = e_i.flatten() 
    X[var_idx["df"]] = df
    X[var_idx["u"]] = 10
    X[var_idx["v"]] = 0.5


    # Init LineCong
    linecong = LineCong()
    linecong.initialize_constraint(X, var_idx, len(tv), bf, nt, len(tf), dual_tf, inner_vertices)
    linecong.set_weigth(1)

    # Init Torsal 
    torsal = Torsal()
    torsal.initialize_constraint(X, var_idx, tv, tf, bf, nt)
    torsal.set_weigth(0)

    # Init optimizer
    optimizer = Optimizer()
    optimizer.initialize_optimizer(X, var_idx, "LM", 0.5)

    
    for _ in range(it):
        
        optimizer.unitize_variable("nt1", 3)
        optimizer.unitize_variable("nt2", 3)
        optimizer.unitize_variable("e", 3)

        optimizer.get_gradients(linecong)

        optimizer.optimize()
   
    visualization(torsal, optimizer, tv, tf, inner_vertices)


def fix_boundary_cross_field(v, f, t1, t2):

    # Make Mesh
    mesh = Mesh()
    mesh.make_mesh(v, f)

    # Get Face face adjacency
    f_f_adj = mesh.face_face_adjacency_list()

    # Get boundary faces
    b_faces = mesh.boundary_faces()

    # Loop over boundary faces
    for bf in b_faces:

        # # Get adjacent faces
        # adj_faces = f_f_adj[bf]
        
        # # Get t1 and t2
        # t1[bf] = np.sum(t1[adj_faces], axis=0)/len(adj_faces)
        # t2[bf] = np.sum(t2[adj_faces], axis=0)/len(adj_faces)

        # Project onto triangle
        vi, vj, vk = v[f[bf,0]], v[f[bf,1]], v[f[bf,2]]

        # Get face normal
        n = np.cross(vj - vi, vk - vi)

        # Project orthogonally onto triangle
        t1[bf] = unit(vj -vi)
        t2[bf] = unit(np.cross(n, t1[bf]))

        # print("boundary face: ",bf)   
        # print(t1[bf])
        # print(t2[bf])

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


   
def visualization(constraint, optimizer, tv, tf, inner_vertices):

    # Get variables
    e = constraint.uncurry_X(optimizer.X, "e")

    # Reshape variables
    e   = unit(e.reshape(-1,3))
   
    # LoadGround truth line congruence
    # Analytic torsal directions
    edir = np.loadtxt(math_path+"edir.dat")
    edir = unit(edir)

    # Check dimensions
    assert len(edir) == len(e), "Not same dimensions in e_i"
        

    # Compute error as the angle between the analytic and the computed edge directions
    error_lc = 0.6*np.ones(len(tv))
    error_lc[inner_vertices] = np.arccos(np.sum(e[inner_vertices]*edir[inner_vertices], axis=1)) * 180/np.pi

    
    # Visualization
    ps.init()

    ps.remove_all_structures()

    # Create mesh
    tm = ps.register_surface_mesh("T1", tv, tf)
    tm.add_vector_quantity("LC", e, defined_on='vertices', enabled=True, radius=0.001, length=2.0, color=(0.0, 1.0, 0.0))
    tm.add_vector_quantity("GT LC", edir, defined_on='vertices', enabled=True, radius=0.001, length=2.0, color=(0.0, 0.0, 1.0))
    tm.add_scalar_quantity("LC error", error_lc, defined_on='vertices', enabled=True, cmap="viridis")

    
    ps.show()


def main():

    run_optimization(It, Data)


main()