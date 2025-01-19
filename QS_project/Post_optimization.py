# Import the necessary libraries
import os
import sys
from pathlib import Path
import argparse
import time
from pickle import load
import subprocess

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
#from energies.Support import Supp
from energies.Sphere import Sphere
from energies.QM_Fairness import QM_Fairness
from energies.Proximity_Gen import Proximity
from energies.Proximity_C import Proximity_C
from energies.Support import Supp
from energies.Support_Fix import Supp_F
from energies.Sphere_Fix_V import Sphere_Fix
from energies.Sph_unit import Sph_Unit
from energies.Tor_Planarity import Tor_Planarity
from energies.Reg_E import Reg_E

from optimization.Optimizer import Optimizer




# Create the parser
parser = argparse.ArgumentParser(description="Post Optimization")

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

# Reports folder
reports_dir = os.path.join(working_path, "data", "Reports")


# Load picke information
def load_data():
    """ Function to load the data from a pickle file
    """
    with open(os.path.join(exp_dir, pickle_name), 'rb') as f:
        data = load(f)
    return data

data = load_data()

if "projected" not in data.keys():
    print("Projecting the mesh")
    # Replace 'script_to_run.py' with the path to your Python script
    script_path = 'foot_point_bspline.py '+name+' 0'
    # Run the script using subprocess
    subprocess.run(['python', script_path])



# Get u and v points 
u_pts = data['u_pts']
v_pts = data['v_pts']

# Get the B-spline and its derivatives
BSurf = data['surf']
rsurf = data['r_uv']

# Get information from the pickle file
f_f_adj = data['f_f_adj']
e_f_f = data['e_f_f']
e_v_v = data['e_v_v']
adj_v = data['adj_v']
inn_f = data['inn_f']
inn_v = data['inn_v']
bd_v = data['bd_v']
cc = data['cc']
vc = data['vc']
nd = data['nd']
f_pts = data['f_pts']
rads = data['rads']
dual_top = data['dual_top']
ffF = data['ffF']
ffV = data['ffV']
ref_V = data['ref_V']
ref_F = data['ref_F']
ref_C = data['ref_C']
VR = data['VR']
l_f = data['l_dir']

A, B, C = get_Implicit_sphere(cc, rads)
dual_edges = np.array(list(extract_edges(dual_top)))
mesh_edges = e_v_v
tuples_edges = np.vstack((mesh_edges[0], mesh_edges[1])).T

w_proximity = 0.01
w_proximity_c = 0.01
w_fairness = 0.2
w_sphericity = 2
w_supp = 0.1
#w_lap = 0.01
iter_per_opt = 1
state = 0
state2 = 0
counter = 0
init_opt = False
w_tor = 0.01
idx_sph = 0
times = []
step = 0.5


#print("Mesh", ffV[:10])
opt = Optimizer()

# Add variables to the optimizer
opt.add_variable("A" , len(vc)  ) # Centers of spheres
opt.add_variable("B" , len(vc)*3  ) # Centers of spheres
opt.add_variable("C" , len(vc)  ) # Centers of spheres

# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", step, 1)

# Initialize variables
opt.init_variable("A"  , A) 
opt.init_variable("B"  , B.flatten())
opt.init_variable("C"  , C)

# # Line congruence l.cu, l.cv = 0
Supp_E = Supp_F()
opt.add_constraint(Supp_E, args=(dual_top, l_f), w=0.001, ce=1)

# Sphericity
Sph_E = Sphere_Fix()
opt.add_constraint(Sph_E, args=(ffF, f_pts), w=10, ce=1)

Sph_U = Sph_Unit()
opt.add_constraint(Sph_U, args=(), w=10, ce=1)

Prox_C = Proximity_C()
opt.add_constraint(Prox_C, args=(ref_C, ref_F, 0.0001), w=0.1)


for _ in range(30):
    # Get gradients
    opt.get_gradients()
    opt.optimize()

opt.get_energy_per_constraint()

A, B, C = opt.uncurry_X("A", "B", "C")

opt = Optimizer()

# Add variables to the optimizer
opt.add_variable("A" , len(vc)  ) # Centers of spheres
opt.add_variable("B" , len(vc)*3  ) # Centers of spheres
opt.add_variable("C" , len(vc)  ) # Centers of spheres

# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", step, 0)

# Initialize variables
opt.init_variable("A"  , A) 
opt.init_variable("B"  , B.flatten())
opt.init_variable("C"  , C)

opt.add_variable("nd", len(l_f)*3   ) # Normals of dual faces
opt.add_variable("v" , len(f_pts)*3) # Vertices of mid mesh
opt.add_variable("n_l" , len(mesh_edges[0])*3   ) # Normal or the supp planes
#opt.add_variable("r" , len(rads)   ) # Radii of spheres
#opt.add_variable("mu", len(dual_edges))


# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", step, 1)

# Initialize variables
opt.init_variable("A"  , A) 
opt.init_variable("B"  , B)
opt.init_variable("C"  , C)
opt.init_variable("v"  , f_pts.flatten())
opt.init_variable("nd" , l_f.flatten())

            
# Constraints ==========================================
# # Line congruence l.cu, l.cv = 0
Supp_E = Supp()
opt.add_constraint(Supp_E, args=(dual_top, inn_v), w=w_supp, ce=1)

# Sphericity
Sph_E = Sphere()
opt.add_constraint(Sph_E, args=(ffF, 2), w=w_sphericity, ce=1)

Sph_U = Sph_Unit()
opt.add_constraint(Sph_U, args=(), w=10)

# # Fairness
Fair_M = QM_Fairness()
opt.add_constraint(Fair_M, args=(adj_v, "v", 3), w=w_fairness)

# Torosal Planarity
Tor_P = Tor_Planarity()
opt.add_constraint(Tor_P, args=(mesh_edges, inn_v), w=w_tor, ce=1)

# # Proximity
Prox_M = Proximity()
opt.add_constraint(Prox_M, args=("v", ref_V, ref_F, 0.0001), w=w_proximity, ce =1)

Prox_C = Proximity_C()
opt.add_constraint(Prox_C, args=(ref_C, ref_F, 0.0001), w=w_proximity_c, ce =1)

# # Define unit variables
opt.unitize_variable("nd", 3, 10)
opt.unitize_variable("n_l", 3, 10)

opt.control_var("v" , 0.4)
opt.control_var("nd", 0.5)



for _ in range(200):
    # Get gradients
    i_t = time.time()
    opt.get_gradients()
    opt.optimize_step()
    f_t = time.time()
    times.append(f_t - i_t)


# Report
opt.get_norm_energy_per_constraint()


# Report data save json
report_data = {
    "Vertices": len(ffV),
    "Faces": len(ffF),
    "W_proximity": w_proximity,
    "W_proximity_c": w_proximity_c,
    "W_fairness": w_fairness,
    "W_sphere": w_sphericity,
    "W_supp": w_supp,
    "W_tor": w_tor,
    "Iterations": opt.it,
    "Time": np.mean(times),
    "Energy": sum(e_i for e_i in opt.norm_energy_dic.values())
}

# Open file to write
file_path = os.path.join(reports_dir, name + "_Final_Report.json")

# Save the variable to a json
with open(file_path, 'w') as file:
    json.dump(report_data, file)


# Get Variables
vk, oA, oB, oC, nl, nd = opt.uncurry_X("v", "A", "B", "C", "n_l", "nd")


vk = vk.reshape(-1,3)
oB = oB.reshape(-1,3)
nl = nl.reshape(-1,3)
nd = nd.reshape(-1,3)

# Save opt into pickle
data['vk'] = vk
data['oA'] = oA
data['oB'] = oB
data['oC'] = oC
data['nl'] = nl
data['nd'] = nd
with open(os.path.join(exp_dir, pickle_name), 'wb') as f:
    pickle.dump(data, f)

# Draw the spheres

sph_c, rf = Implicit_to_CR(oA,oB,oC)

v0, v1, v2, v3 = vk[ffF[:, 0]], vk[ffF[:, 1]], vk[ffF[:, 2]], vk[ffF[:, 3]]

# save mesh
write_obj(os.path.join(remeshing_dir, name+'_mesh_opt.obj'), vk, ffF)

# Save spheres ======================================================================
# Open file
sph_file = open(os.path.join(remeshing_dir, name+'_sphs_opt.dat'), 'w')
sph_cut_file = open(os.path.join(remeshing_dir, name+'_sphs_p_cut.dat'), 'w')
annuli_file = open(os.path.join(remeshing_dir, name+'_annuli.dat'), 'w')

# Vedo data 
sph_panels = []

# Write inn_f in the first line
for i in inn_f:
    sph_file.write(f"{i} ")
sph_file.write("\n")

# Get list of faces of each edge
edges_faces = np.c_[e_f_f].flatten()
for i in edges_faces:
    annuli_file.write(f"{i} ")
annuli_file.write("\n")

for i in range(len(e_f_f[0])):

    # Get the normal of the edge
    n = nl[i]

    # Get the vertices of the edge
    vi, vj = vk[e_v_v[0][i]], vk[e_v_v[1][i]]

    # Compute the mid point of the edge
    mid = (vi + vj)/2
    annuli_file.write(f"{mid[0]} {mid[1]} {mid[2]} {n[0]} {n[1]} {n[2]} {vi[0]} {vi[1]} {vi[2]} {vj[0]} {vj[1]} {vj[2]}\n")

# ffF are the faces of the mesh
for i in range(len(ffF)):
    
    # Write the sphere data
    sph_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} {vc[i][0]} {vc[i][1]} {vc[i][2]} ")

    # Sphere 
    string_data_sph_panel = f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} "

    # Annuli data 
    annuli_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]}\n")

    # sphere Data 
    sphere_data = [] 

    sphere_data.append(sph_c[i])
    sphere_data.append(rf[i])

    planes_sphere = []

    for j in range(len(ffF[i])):

        #edge
        e = [ffF[i][j], ffF[i][(j+1) % len(ffF[i])]]

        # find index of edge
        idx = search_edge(tuples_edges, e)

        # Get the normal of the edge
        n = nl[idx]

        # Get the vertices of the edge
        vi, vj = vk[e[0]], vk[e[1]]

        # Compute the mid point of the edge
        mid = (vi + vj)/2
        string_data_sph_panel += f"{vi[0]} {vi[1]} {vi[2]} {mid[0]} {mid[1]} {mid[2]} {n[0]} {n[1]} {n[2]} "

        planes_sphere.append([vi, n])
    
    sph_panels.append([sphere_data, planes_sphere])

    sph_cut_file.write(string_data_sph_panel + "\n")
    

    for j in f_f_adj[i]:
        sph_file.write(f"{j} ")

    sph_file.write("\n")

    
annuli_file.close()
sph_cut_file.close()
sph_file.close()




# Save Optimized line congruence
Node_path = os.path.join(remeshing_dir, name+'_opt_NodesAxis.obj')

# Compute line congruence
v_lc = np.vstack((vk - 0.3*nd, vk + 0.3*nd))

f_lc = np.array([[i, i+len(vk)] for i in range(len(vk))])
# Write info
with open(Node_path, 'w') as f:
    for v in v_lc:
        f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    for l in f_lc:
        f.write('l {} {}\n'.format(l[0]+1, l[1]+1))


