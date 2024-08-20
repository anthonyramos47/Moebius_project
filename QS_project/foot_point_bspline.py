import numpy as np
#import pymeshlab
import polyscope as ps
import argparse
import time
from pickle import load, dump


# Import the necessary libraries
import os
import sys
from pathlib import Path

hanan_path = os.getenv('HANANLAB_PATH')
if not hanan_path:
    raise EnvironmentError("HANANLAB_PATH environment variable not set")
sys.path.append(hanan_path)

from geometry.utils import *
from geometry.mesh import Mesh
from optimization.Optimizer import Optimizer
from energies.Proximity_Gen import Proximity
from energies.QM_Fairness import QM_Fairness
from utils.bsplines_functions import *




# Create the parser
parser = argparse.ArgumentParser(description="Visualizer Parser")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')
parser.add_argument('vis', type=int, help='Visualization on or off')

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
remeshed_obj = os.path.join( remeshing_dir,  name+'_Remeshed.obj')

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

init_l = data['init_l'].reshape(-1, 3)
opt_l  = data['l'].reshape(-1, 3)
lc  = data['lc'].reshape(-1, 3)

V = data['V']


print("Topology cleaning")
new_ffF = []
for f in ffF:
    if len(f) != 4:
        print(f)
    else:
        new_ffF.append(f)

ffF = np.array(new_ffF)



# Get the B-splines
BSurf = data['surf']
rsurf = data['r_uv']


# Sample size of the B-spline
sample = (len(u_pts), len(v_pts))   

# Evaluate the B-spline at the u and v points
V, F = Bspline_to_mesh(BSurf, u_pts, v_pts)

range_u = (0, 1)
range_v = (0, 1)

# Define the finer reference mesh
ref_u = np.linspace(range_u[0],  range_u[1], 300)
ref_v = np.linspace(range_v[0],  range_v[1], 300)
# Compute the vertices and faces of the finer reference mesh
ref_V, ref_F = Bspline_to_mesh(BSurf, ref_u, ref_v)
# Triangulate the quads



# Create mesh for Mid mesh (sphere centers)
aux_mesh = Mesh()
aux_mesh.make_mesh(ffV, ffF)

# Get adjacent vertices
adj_v = aux_mesh.vertex_adjacency_list()



# Foot Point optimization initializer
opt = Optimizer()
 

# Add variables to the optimizer
opt.add_variable("v" , len(ffV)*3) # Vertices of mid mesh
#opt.add_variable("r" , len(rads)   ) # Radii of spheres
#opt.add_variable("mu", len(dual_edges))

# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", 0.65, 1)

# Initialize variables
opt.init_variable("v"  , ffV.flatten())

# # Fairness
Fair_M = QM_Fairness()
opt.add_constraint(Fair_M, args=(adj_v, "v", 3), w=3, ce=1)

# # Proximity
Prox_M = Proximity()
opt.add_constraint(Prox_M, args=("v", ref_V, ref_F, 0.01), w=5, ce=1)

opt.control_var("v", 0.2)

for i in range(50):

    opt.get_gradients() # Compute J and residuals
    opt.optimize() # Solve linear system and update variables

opt.get_energy_per_constraint()


orffV = ffV.reshape(-1, 3)

ffV = opt.uncurry_X("v")

ffV = ffV.reshape(-1, 3)



# Compute footpoints (u,v) coordinates of remeshed mesh onto the B-spline
foot_pts, cls_pts = foot_points(ffV, ref_V, ref_F, ref_u, ref_v, BSurf, u_range=range_u, v_range=range_v)
foot_pts = foot_pts.reshape(-1, 2)



# Evaluate the B-spline functions at the foot points bsurf(u, v), r(u, v) and n(u, v)
f_pts = np.zeros((len(foot_pts), 3))
r_pts = np.zeros((len(foot_pts)))

n_dir = np.zeros((len(foot_pts), 3))
for i in range(len(foot_pts)):
    #l_dir[i] = sph_ln_cong_at_pt(BSurf, rsurf, foot_pts[i, 0], foot_pts[i, 1])[1]
    n_dir[i] = BSurf.normal(foot_pts[i, 0], foot_pts[i, 1])
    f_pts[i] =   BSurf(foot_pts[i, 0], foot_pts[i, 1])
    r_pts[i] = bisplev(foot_pts[i, 0], foot_pts[i, 1], rsurf)

# Compute the vertices of the mid mesh C(u,v)
VR = f_pts + r_pts[:,None]*n_dir
VR = VR.reshape(-1, 3)

# Interpolate line congruence
TF, EV = triangulate_quads(F, V)

TF = np.array(TF)


# Extend opt_l by interpolation at barycenter
opt_l_ext = np.vstack((opt_l, lc))


l_dir = interpolate_lc(f_pts, EV, TF, opt_l_ext)


# Topological information ===============================================================

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

# Boundary vertices
bd_v = mesh.boundary_vertices()

# Get adjacent vertices
adj_v = mesh.vertex_adjacency_list()

# Get adjacent faces
ffF = mesh.faces()

# Topological information ===============================================================

# Compute dual faces vertices and normals

# Compute the nodes of the mid mesh
c0, c1, c2, c3 = VR[ffF[:, 0]], VR[ffF[:, 1]], VR[ffF[:, 2]], VR[ffF[:, 3]]

# Compute the baricenter of the mid mesh
cc = (c0 + c1 + c2 + c3)/4

# Compute the barycenter at the remeshed mesh
vc = np.sum(f_pts[ffF], axis=1)/4

# Compute radius of spheres
#rads = np.linalg.norm(vc - cc, axis=1)
rads =  np.sum(r_pts[ffF ], axis=1)/4

# # Normals of dual faces
nd = np.zeros((len(dual_top), 3))

for i, f_i in enumerate(dual_top):
    if len(f_i) >= 4:
        nd[i] = np.cross(cc[f_i[2]] - cc[f_i[0]], cc[f_i[1]] - cc[f_i[3]])
        nd[i] /= np.linalg.norm(nd[i])
    elif len(f_i) == 3:
        nd[i] = np.cross(cc[f_i[1]] - cc[f_i[0]], cc[f_i[2]] - cc[f_i[1]])
        nd[i] /= np.linalg.norm(nd[i])

ref_F = triangulate_quads_diag(ref_F)

ref_F = np.array(ref_F)

# Get vertices of C(u, v) 
# Evaluate r(u,v) in ref_u ref_v
ref_r_uv = bisplev(ref_u, ref_v, rsurf)
# Compute the normals of the reference mesh
ref_n = BSurf.normal(ref_u, ref_v)
ref_rn = ref_r_uv[:, :, None]*ref_n

ref_v_aux = BSurf(ref_u, ref_v)
ref_C = ref_v_aux + ref_rn
ref_C = ref_C.reshape(-1, 3)


# Write relevant information as obj 
# =================================================================================================

# reference surfaces save
# Surface
ref_mesh = os.path.join(remeshing_dir, name+'_Ref_surf.obj')
write_obj(ref_mesh, ref_V, ref_F)
# Centers of sphere congruences
ref_c_mesh = os.path.join(remeshing_dir, name+'_Ref_C.obj')
write_obj(ref_c_mesh, ref_C, ref_F)


# Initial line congruence
LC_path = os.path.join(remeshing_dir, name+'_init_LC.obj')

# Compute line congruence

v_lc = np.vstack((V, V + init_l))

f_lc = np.array([[i, i+len(V)] for i in range(len(V))])

# Write info
with open(LC_path, 'w') as f:
    for v in v_lc:
        f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    for l in f_lc:
        f.write('l {} {}\n'.format(l[0]+1, l[1]+1))


# Optimized line congruence
LC_path = os.path.join(remeshing_dir, name+'_opt_LC.obj')

# Compute line congruence
v_lc = np.vstack((V, V + opt_l))
# Write info
with open(LC_path, 'w') as f:
    for v in v_lc:
        f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    for l in f_lc:
        f.write('l {} {}\n'.format(l[0]+1, l[1]+1))



# =================================================================================================


# Update the data dictionary
data["projected"] = True
data["dual_top"] = dual_top
data["f_f_adj"] = f_f_adj
data["e_f_f"] = e_f_f
data["e_v_v"] = e_v_v
data["inn_f"] = inn_f
data["inn_v"] = inn_v
data["bd_v"] = bd_v
data["adj_v"] = adj_v
data["ffV"] = ffV
data["ffF"] = ffF
data["rads"] = rads
data["cc"] = cc
data["vc"] = vc
data["f_pts"] = f_pts
data["ref_V"] = ref_V
data["ref_C"] = ref_C
data["ref_F"] = ref_F
data["nd"] = nd
data["VR"] = VR
data["l_dir"] = l_dir

# Step 4: Dump the updated data back into the pickle file
with open(os.path.join(exp_dir, pickle_name), 'wb') as file:
    dump(data, file)


if parser.parse_args().vis == 1:
        
    ps.init()

    ps.remove_all_structures()

    start_m = ps.register_surface_mesh("S_uv", V, F, enabled=False)
    
    start_m.add_vector_quantity("l", opt_l, enabled=True)
    or_mesh = ps.register_surface_mesh("foot_pts", f_pts, ffF)
    ps.register_surface_mesh("Or Remeshed", orffV, ffF)
    or_mesh.add_vector_quantity("l", l_dir, enabled=True)
    ps.register_surface_mesh("C_uv", VR, ffF, enabled=False)
    ps.register_surface_mesh("ref_mesh", ref_V, ref_F, enabled=False)
    ps.register_surface_mesh("ref_C", ref_C, ref_F, enabled=False)

    #ps.register_point_cloud("C centers", cc)
    #ps.register_point_cloud("V centers", vc)


    ps.show()