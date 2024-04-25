# Import the necessary libraries
import os
import sys
from pathlib import Path
import argparse
from pickle import load

# Obtain the path HananLab; this is the parent directory of the hananLab/hanan directory
# <Here you can manually add the path direction to the hananLab/hanan directory>
# Linux 
path = os.getenv('HANANLAB_PATH')
if not path:
    raise EnvironmentError("HANANLAB_PATH environment variable not set")
sys.path.append(path)

# Verification of the path
print(path)


# Import the necessary libraries for visualization and computation
import igl
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import json
import pickle



# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *

# Local files
from utils.bsplines_functions import *
from utils.visualization import *

# Optimization classes
from energies.Support import Supp
from energies.Sphericity import Sphericity
from energies.Proximity import Proximity
from energies.QM_Fairness import QM_Fairness
from optimization.Optimizer import Optimizer



# Create the parser
parser = argparse.ArgumentParser(description="Visualizer Parser")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
name = parser.parse_args().file_name
pickle_name = name+'.pickle'
save = 1

# Working directory
working_path = os.getcwd()

# Experiments folder
exp_dir = os.path.join(working_path, 'experiments')

# Remeshing data folder
remeshing_dir = os.path.join(working_path, 'data', 'Remeshing', name)

# Frame Field remeshed obj
remeshed_obj = os.path.join( remeshing_dir,  name+'_backmapped.obj')

# Read remeshed mesh
ffV, ffF = read_obj(remeshed_obj)

# Load picke information
def load_data():
    """ Function to load the data from a pickle file
    """
    with open(os.path.join(exp_dir, pickle_name), 'rb') as f:
        data = load(f)
    return data

data = load_data()

# Get u and v points 
u_pts = data['u_pts']
v_pts = data['v_pts']

# Get the B-spline and its derivatives
BSurf = data['surf']
rsurf = data['r_uv']

# Sample size of the B-spline
sample = (len(u_pts), len(v_pts))   

# Evaluate the B-spline at the u and v points
V, F = Bspline_to_mesh(BSurf, u_pts, v_pts)

# Get the vertices and faces of the mesh
ref_u = np.linspace(0, 1, 300)
ref_v = np.linspace(0, 1, 300)

# Get the vertices and faces of the mesh

ref_V, ref_F = Bspline_to_mesh(BSurf, ref_u, ref_v)

ref_F = np.array(triangulate_quads(ref_F))


# Compute footpoints (u,v) coordinates of remeshed mesh onto the B-spline
foot_pts = foot_points(ffV, V, u_pts, v_pts, BSurf)
foot_pts = foot_pts.reshape(-1, 2)

# Evaluate the B-spline functions at the foot points bsurf(u, v), r(u, v) and n(u, v)
f_pts = np.zeros((len(foot_pts), 3))
r_pts = np.zeros((len(foot_pts), 3))
n_dir = np.zeros((len(foot_pts), 3))
for i in range(len(foot_pts)):
    f_pts[i] = BSurf(foot_pts[i, 0], foot_pts[i, 1])
    n_dir[i] = BSurf.normal(foot_pts[i, 0], foot_pts[i, 1])
    r_pts[i] = bisplev(foot_pts[i, 0], foot_pts[i, 1], rsurf)

# Compute the vertices of the mid mesh
VR = f_pts + r_pts[:,None]*n_dir
VR = VR.reshape(-1, 3)


# Create mesh for Mid mesh (sphere centers)
mesh = Mesh()
mesh.make_mesh(VR, ffF)

# Get the face-face adjacency list 
f_f_adj = mesh.face_face_adjacency_list()
dual_top = mesh.dual_top()

# Faces of each edge
e_f_f = mesh.edge_faces()

# Vertices of each edge
e_v_v = mesh.edge_vertices()



# Get inner faces indices
inn_f = mesh.inner_faces()

# Get inner vertices
inn_v = mesh.inner_vertices()

# Get adjacent vertices
adj_v = mesh.vertex_adjacency_list()

# Get adjacent faces
ffF = mesh.faces()

# print(mesh.boundaries())

# vb = f_pts[mesh.boundary_vertices()]

# Compute the nodes of the mid mesh
c0, c1, c2, c3 = VR[ffF[:, 0]], VR[ffF[:, 1]], VR[ffF[:, 2]], VR[ffF[:, 3]]

# Compute the baricenter of the mid mesh
cc = (c0 + c1 + c2 + c3)/4

# Compute the barycenter at the remeshed mesh
vc = np.sum(f_pts[ffF], axis=1)/4

# Compute radius of spheres
rads = np.linalg.norm(vc - cc, axis=1)

# dual_top = []
# for f_i in inn_f:
#     dual_top.append(f_f_adj[f_i])
#     if len(f_f_adj[f_i]) != 4:
#         print(len(f_f_adj[f_i]))

# OPTIMIZATION ===============================================================

# Creat a set of 8 random centers
# cc = np.array([ 
#     [0,0,1],  #0
#     [1,0,0],  #1
#     [1,1,0],  #2
#     [0,1,0],  #3
#    [-1,1,0],  #4
#     [-1,0,0], #5
#     [-1,-1,0],#6
#     [0,-1,0], #7
#     [1,-1,0]  #8
# ])

# dual_top = np.array([
#     [0, 1, 2, 3],
#     [0, 3, 4, 5],
#     [0, 5, 6, 7],
#     [0, 7, 8, 1]
# ])
        

# dual_mesh = Mesh()
# dual_mesh.make_mesh(cc, dual_top)

# in_v_c = dual_mesh.inner_vertices()
# ad_v_c = dual_mesh.vertex_adjacency_list()

# cc = dual_mesh.vertices
# dual_top = dual_mesh.faces()

# # Normals of dual faces
nd = np.zeros((len(dual_top), 3))

for i, f_i in enumerate(dual_top):
    if len(f_i) >= 4:
        nd[i] = np.cross(cc[f_i[2]] - cc[f_i[0]], cc[f_i[1]] - cc[f_i[3]])
        nd[i] /= np.linalg.norm(nd[i])
    elif len(f_i) == 3:
        nd[i] = np.cross(cc[f_i[1]] - cc[f_i[0]], cc[f_i[2]] - cc[f_i[1]])
        nd[i] /= np.linalg.norm(nd[i])




# Find zero nd
#print(nd[np.where(np.linalg.norm(nd, axis=1) == 0)[0]])


# idx_dual_top = np.unique(fla_dual_top)

# # Network to visualzie nd
# nodes = np.hstack((vc[idx_dual_top], vc[idx_dual_top] + nd))
# nodes = nodes.reshape(-1, 3)

# edges = []
# for i in range(len(vc)):
#     edges.append([i, i+len(vc)])
        

# opt = Optimizer()

# # Add variables to the optimizer
# opt.add_variable("c" , len(vc)*3   ) # Centers of spheres
# opt.add_variable("nd", len(nd)*3   ) # Normals of dual faces
# opt.add_variable("v" , len(f_pts)*3) # Vertices of mid mesh
# opt.add_variable("r" , len(rads)   ) # Radii of spheres

# # Initialize Optimizer ("Method", step, verbosity)
# opt.initialize_optimizer("LM", 0.6, 1)

# # Initialize variables
# opt.init_variable("c"  , cc.flatten()) 
# opt.init_variable("nd" , nd.flatten())
# opt.init_variable("v"  , f_pts.flatten())
# opt.init_variable("r"  , rads)



# # Constraints ==========================================
# # Line congruence l.cu, l.cv = 0
# Supp_E = Supp()
# opt.add_constraint(Supp_E, args=(dual_top, 2), w=1)

# # Sphericity
# Sph_E = Sphericity()
# opt.add_constraint(Sph_E, args=(ffF, 2), w=1)

# # Fairness
# Fair_M = QM_Fairness()
# opt.add_constraint(Fair_M, args=(inn_v, adj_v, "v", 3), w=1)

# # Proximity
# Prox_M = Proximity()
# opt.add_constraint(Prox_M, args=(ref_V, ref_F, 0.001), w=0.5)

# # Fair_C = QM_Fairness()
# # opt.add_constraint(Fair_C, args=(in_v_c, ad_v_c, "c", 3), w=0.002)

# # Define unit variables
# opt.unitize_variable("nd", 3, 10)

# for _ in range(100):
#     # Get gradients
#     opt.get_gradients() # Compute J and residuals
#     opt.optimize() # Solve linear system and update variables


# opt.get_energy_per_constraint()


# if save:
#     # save mesh
#     write_obj(os.path.join(remeshing_dir, name+'_mesh_opt.obj'), vk, ffF)

#     # Save spheres ======================================================================
#     # Open file
#     exp_file = open(os.path.join(remeshing_dir, name+'_sphs_opt.dat'), 'w')

#     # Write inn_f in the first line
#     for i in inn_f:
#         exp_file.write(f"{i} ")
#     exp_file.write("\n")

#     for i in range(len(ffF)):

#         exp_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} {vc[i][0]} {vc[i][1]} {vc[i][2]} ")

#         for j in f_f_adj[i]:
#             exp_file.write(f"{j} ")

#         exp_file.write("\n")

#     exp_file.close()

w_proximity = 0.5
w_fairness = 1
w_sphericity = 1
w_supp = 1
iter_per_opt = 10
state = 0
counter = 0
name_saved = "Post_optimization"
def optimization():

    global w_proximity, w_fairness, w_sphericity, w_supp, iter_per_opt, init_opt, state, counter, vc, nd, f_pts, rads, dual_top, ffF, ref_V, ref_F, ref_u, ref_v, cc, vc, opt, inn_v, adj_v, name_saved,e_f_f, e_v_v

    # Title
    psim.TextUnformatted("Post Optimization GUI")

    if psim.Button("Stop"):
        state = 0
        state2 = 0

    psim.PushItemWidth(150)
    changed, iter_per_opt = psim.InputInt("Num of Iterations per Run: ", iter_per_opt)
    psim.Separator()
       
    # Inputs Opt 1
    if psim.CollapsingHeader("Optimization 1:"):
        changed, w_proximity  = psim.InputFloat("Proximity W", w_proximity)
        changed, w_fairness   = psim.InputFloat("Fairness W", w_fairness)
        changed, w_sphericity = psim.InputFloat("Sphericity W", w_sphericity)
        changed, w_supp       = psim.InputFloat("Support W", w_supp)

        # State handler
        if counter%iter_per_opt == 0:
            state = 0

        if psim.Button("Init First opt"):

            # Set init to True
            init_opt = True
                        
            opt = Optimizer()

            # Add variables to the optimizer
            opt.add_variable("c" , len(vc)*3   ) # Centers of spheres
            opt.add_variable("nd", len(nd)*3   ) # Normals of dual faces
            opt.add_variable("v" , len(f_pts)*3) # Vertices of mid mesh
            opt.add_variable("r" , len(rads)   ) # Radii of spheres

            # Initialize Optimizer ("Method", step, verbosity)
            opt.initialize_optimizer("LM", 0.6, 1)

            # Initialize variables
            opt.init_variable("c"  , cc.flatten()) 
            opt.init_variable("nd" , nd.flatten())
            opt.init_variable("v"  , f_pts.flatten())
            opt.init_variable("r"  , rads)

            # Constraints ==========================================
            # Line congruence l.cu, l.cv = 0
            Supp_E = Supp()
            opt.add_constraint(Supp_E, args=(dual_top, 2), w=1)

            # Sphericity
            Sph_E = Sphericity()
            opt.add_constraint(Sph_E, args=(ffF, 2), w=1)

            # Fairness
            Fair_M = QM_Fairness()
            opt.add_constraint(Fair_M, args=(inn_v, adj_v, "v", 3), w=1)

            # Proximity
            Prox_M = Proximity()
            opt.add_constraint(Prox_M, args=(ref_V, ref_F, 0.001), w=0.5)

            # Fair_C = QM_Fairness()
            # opt.add_constraint(Fair_C, args=(in_v_c, ad_v_c, "c", 3), w=0.002)

            # Define unit variables
            opt.unitize_variable("nd", 3, 10)

            ps.info("Finished Initialization of Optimization 1")


        if psim.Button("Optimize 1"):
            if init_opt:
                for _ in range(iter_per_opt):
                    # Optimize
                    opt.get_gradients() # Compute J and residuals
                    opt.optimize() # Solve linear system and update variables
                
                # Get Line congruence
                vk, sph_c, _, rf = opt.uncurry_X("v", "c", "nd", "r")
                sph_c = sph_c.reshape(-1, 3)
                vk = vk.reshape(-1,3)

                

                mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
                mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
                ps.register_surface_mesh("Opt_C", sph_c, dual_top)

                
            else:
                ps.warning("First Optimization not initialized")
                state = 0

    
    psim.Separator()


    if psim.Button("Report"):
        opt.get_energy_per_constraint()

    psim.Separator()

    if psim.Button("Draw Support Structure"):

        # Get results
        vk, sph_c, _, rf = opt.uncurry_X("v", "c", "nd", "r")
        sph_c = sph_c.reshape(-1, 3)
        vk = vk.reshape(-1,3)

        # Get vertices per edge
        vi, vj = vk[e_v_v[0]], vk[e_v_v[1]]

        # Get sphere indices per edge
        c_i, c_j = sph_c[e_f_f[0]], sph_c[e_f_f[1]]

        dir = c_j - c_i
        d = np.linalg.norm(dir, axis=1)
        dir /= d[:, None]

        ri, rj = rf[e_f_f[0]], rf[e_f_f[1]]

        mc = c_i + ((ri**2 - rj**2 + d**2)/(2*d))[:, None]*dir
        # # Dir vector between spheres
        # dij  = c_j - c_i
        # dij /= np.linalg.norm(dij, axis=1)[:, None]

        # # Point of bisecting plane between spheres
        # mc =  c_i +  np.einsum('ij,ij->i', (vi - c_i), dij)[:, None]*dij

        v1mc = vi - mc
        v2mc = vj - mc

        supp_obj = open(os.path.join(remeshing_dir, name+'_supp_opt.obj'), 'w')

        for i in range(len(mc)):
            verts = np.array([mc[i] + 0.98*v1mc[i], mc[i] + 0.98*v2mc[i], mc[i] + 1.02*v2mc[i], mc[i] + 1.02*v1mc[i]])
            #draw_polygon(verts, (0.1, 0, 0.2), name="plane"+str(i))

            
            supp_obj.write(f"v {verts[0][0]} {verts[0][1]} {verts[0][2]}\n")
            supp_obj.write(f"v {verts[1][0]} {verts[1][1]} {verts[1][2]}\n")
            supp_obj.write(f"v {verts[2][0]} {verts[2][1]} {verts[2][2]}\n")
            supp_obj.write(f"v {verts[3][0]} {verts[3][1]} {verts[3][2]}\n")

            supp_obj.write(f"f {4*i + 1} {4*i + 2} {4*i + 3} {4*i + 4} \n")

        supp_obj.close()
        
        mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
        mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
        ps.register_surface_mesh("Opt_C", sph_c, dual_top)

    psim.Separator()
    psim.TextUnformatted("Save Results")
    if psim.Button("Save"):

        # Get results
        vk, sph_c, rf  = opt.uncurry_X("v", 'c', 'r')
        vk = vk.reshape(-1,3)
        sph_c = sph_c.reshape(-1, 3)

        # save mesh
        write_obj(os.path.join(remeshing_dir, name+'_mesh_opt.obj'), vk, ffF)

        # Save spheres ======================================================================
        # Open file
        exp_file = open(os.path.join(remeshing_dir, name+'_sphs_opt.dat'), 'w')

        # Write inn_f in the first line
        for i in inn_f:
            exp_file.write(f"{i} ")
        exp_file.write("\n")

        for i in range(len(ffF)):

            exp_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} {vc[i][0]} {vc[i][1]} {vc[i][2]} ")

            for j in f_f_adj[i]:
                exp_file.write(f"{j} ")

            exp_file.write("\n")

        exp_file.close()


         
ps.init()
ps.register_surface_mesh("mesh", ffV, ffF)
ps.register_surface_mesh("S_uv", ref_V, ref_F)


ps.set_user_callback(optimization)
ps.show()



