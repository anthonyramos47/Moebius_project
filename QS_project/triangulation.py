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
from utils.bsplines_functions import *

path = os.getcwd()
print("Path triangulation:", path)

# experiment dir
exp_dir = os.path.join(path, 'QS_project', 'experiments')
print("Experiment dir:", exp_dir)


# Create the parser
parser = argparse.ArgumentParser(description="Visualizer Parser")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
file_name = parser.parse_args().file_name

# Save dir
save_dir = os.path.join(path, 'QS_project', 'data', 'Remeshing', file_name)

print("Save dir:", save_dir)

file_name += '.pickle'


def load_data():
    """ Function to load the data from a pickle file
    """
    with open(os.path.join(exp_dir, file_name), 'rb') as f:
        data = load(f)
    return data


data = load_data()

V = data['V']  # Vertices
F = data['F']  # Faces

TF = np.array(triangulate_quads_diag(F))


or_l = data['l']  # Line congruence
lc = data['lc']  # Line congruence at the center of the face
or_l = or_l.reshape(-1, 3)

u_pts = data['u_pts']  # U points
v_pts = data['v_pts']  # V points

ot1 = data['t1']
ot2 = data['t2']

ot1 = ot1.reshape(-1, 3)
ot2 = ot2.reshape(-1, 3)

# v0, v1, v2, v3 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]], V[F[:, 3]]
# l0, l1, l2, l3 = or_l[F[:, 0]], or_l[F[:, 1]], or_l[F[:, 2]], or_l[F[:, 3]]

# lu = l2 - l0
# lv = l1 - l3

# du = v2 - v0
# dv = v3 - v0

# bc = ( v0 + v1 + v2 + v3 ) / 4
# lc = ( l0 + l1 + l2 + l3 ) / 4

# # ct1, ct2, _, _, _, _, idx =  torsal_directions(lc, lu, lv, du, dv)


# valid = np.zeros(len(F))
# valid[idx] = 1

# BSurf = data['surf']
# rsurf = data['r_uv']
# Assuming V and F are your vertices and faces arrays
# For demonstration, let's create some dummy data
# V: 3D positions of vertices (N x 3 numpy array)
# F: Indices of vertices that compose each face (M x 3 numpy array for triangles)


# # Create a new MeshSet instance
# m = pymeshlab.Mesh(V, F)

# # create a new MeshSet
# ms = pymeshlab.MeshSet()

# # add the mesh to the MeshSet
# ms.add_mesh(m, "Mesh")

# # Perform isotropic remeshing
# ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(5))

# # You can extract the remeshed mesh's vertices and faces if needed
# remeshed_mesh = ms.current_mesh()
# remeshed_V = remeshed_mesh.vertex_matrix()
# remeshed_F = remeshed_mesh.face_matrix()

# print("V size", remeshed_V.shape)


# # Footpoints of remeshed mesh
# i_t = time.time()

# ui_vj = foot_points(V, V, u_pts, v_pts, BSurf)

# f_t = time.time()

# print("\n\nTime to compute foot points:", f_t - i_t)



# for i in range(len(ui_vj)):

#     V_R[i] = BSurf(ui_vj[i][0], ui_vj[i][1])

#     C_uv[i], l_uv[i]  = sph_ln_cong_at_pt(BSurf, rsurf, ui_vj[i][0], ui_vj[i][1])
ot1 /= np.linalg.norm(ot1, axis=1)[:, None]
ot2 /= np.linalg.norm(ot2, axis=1)[:, None]

int1, int2, nbc =interpolate_torsal_Q_tri(ot1, ot2, V, F)
tF= triangulate_quads_diag(F)

tV = V

# even tf
eve_T = tF[::2]

# Get baricenter of each face
ev_c = np.mean(tV[eve_T], axis=1)

tF = np.array(tF)

ext_l = np.vstack((or_l, lc))

v0, v1, v2, v3 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]], V[F[:, 3]]
vc  = (v0 + v1 + v2 + v3)/4

t1, t2, bc, idx = get_torsal_Mesh(tV, tF, ext_l)



# _, _, vc  = get_torsal_QMesh(V, F,  or_l)

# save_torsal(vc, ot1, ot2, path=save_dir, type=2)

# _, _, vc  = get_torsal_Mesh(V, TF,  or_l)
# t1 = ot1.repeat(2, axis=0)
# t2 = ot2.repeat(2, axis=0)

# print(len(TF))
# print(len(F))

# #ot1, ot2, bc = get_torsal_Mesh(V, F, or_l)

# Update the data dictionary
data["int_t2"] = int2
data["int_t1"] = int1
data["nbc"] = nbc
data["TV"] = tV
data["TF"] = tF


# Step 4: Dump the updated data back into the pickle file
with open(os.path.join(exp_dir, file_name), 'wb') as file:
    dump(data, file)

# # V_R = 
# # C_uv = np.zeros((len(ui_vj), 3))
# # l_uv = np.zeros((len(ui_vj), 3))
valid = np.zeros(len(tF))

valid[idx] = 1

# # # # If you want to save the remeshed mesh to a file
# ps.init()

# ps.remove_all_structures()

# tri = ps.register_surface_mesh("tri_mesh", tV, tF)
# # # tri.add_scalar_quantity("valid", valid, defined_on="faces", enabled=True, cmap="blues")
# # # or_mesh = ps.register_surface_mesh("mesh", V, F)

# # # or_mesh.add_scalar_quantity("valid", valid, defined_on="faces", enabled=True, cmap="blues")
# # # or_mesh.add_vector_quantity("l", or_l, defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.0, 1.0, 0.0))

# # # ps.register_point_cloud("vc", vc, color=(1, 0, 0), radius=0.001)

# torsal_dir_show(nbc, int1, int2, size=0.02, rad=0.0005,  color=(1,0,0), name="Tri")
# torsal_dir_show(vc, ot1, ot2, size=0.02, rad=0.0005,  color=(1,1,1), name="")

# # # ps.register_surface_mesh("Proj", V_R, TF)
# # # ps.register_surface_mesh("C_uv", C_uv, TF)
# ps.show()

save_torsal(vc, ot1, ot2, path=save_dir, type=2)
save_torsal(nbc, int1, int2, path=save_dir)
save_torsal(ev_c, ot1,  ot2, path=save_dir, type=3)


