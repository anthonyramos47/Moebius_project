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
V, F = Bspline_to_mesh(BSurf, u_pts, v_pts, sample)

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

# Compute the nodes of the mid mesh
c0, c1, c2, c3 = VR[ffF[:, 0]], VR[ffF[:, 1]], VR[ffF[:, 2]], VR[ffF[:, 3]]

# Compute the baricenter of the mid mesh
cc = (c0 + c1 + c2 + c3)/4


# Create mesh for Mid mesh (sphere centers)
mesh = Mesh()
mesh.make_mesh(VR, ffF)


# Get the face-face adjacency list 

f_f_adj = mesh.face_face_adjacency_list()
dual_top = mesh.dual_top()

# Get inner faces indices
inn_f = mesh.inner_faces()

# Get inner vertices
inn_v = mesh.inner_vertices()


ffF = mesh.faces()

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

# Define unit variables
opt.unitize_variable("nd", 3, 10)

for _ in range(100):
    # Get gradients
    opt.get_gradients() # Compute J and residuals
    opt.optimize() # Solve linear system and update variables


opt.get_energy_per_constraint()

vk, sph_c, n_dual, rf = opt.uncurry_X("v", "c", "nd", "r")

sph_c = sph_c.reshape(-1, 3)
vk = vk.reshape(-1,3)


vc = np.sum(f_pts[ffF], axis=1)/4

if save:
    # save mesh
    write_obj(os.path.join(remeshing_dir, name+'_mesh_opt.obj'), vk, ffF)

    # save spheres

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
or_mesh = ps.register_surface_mesh("mesh", ffV, ffF)

i_v = ps.register_point_cloud("Inner V", ffV[inn_v])
i_v.add_vector_quantity("Normals", nd)

#ps.register_curve_network("normals", nodes, edges)

#ps.register_surface_mesh("foot_pts", f_pts, ffF)
ps.register_surface_mesh("C_uv", VR, ffF)
ps.register_surface_mesh("Opt_Vertices", vk, ffF)
ps.register_surface_mesh("Dual V", vc, dual_top)
ps.register_surface_mesh("Optimized", sph_c, dual_top)
ps.register_surface_mesh("Dual C", sph_c, dual_top)

ps.show()



