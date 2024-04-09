import numpy as np
import pymeshlab
import polyscope as ps
import argparse
import time
from pickle import load


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
print(path)

# experiment dir
exp_dir = os.path.join(path, 'experiments')

# Create the parser
parser = argparse.ArgumentParser(description="Visualizer Parser")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
file_name = parser.parse_args().file_name

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
or_l = data['l']  # Line congruence
or_l = or_l.reshape(-1, 3)

u_pts = data['u_pts']  # U points
v_pts = data['v_pts']  # V points

BSurf = data['surf']
rsurf = data['r_uv']
# Assuming V and F are your vertices and faces arrays
# For demonstration, let's create some dummy data
# V: 3D positions of vertices (N x 3 numpy array)
# F: Indices of vertices that compose each face (M x 3 numpy array for triangles)


# Create a new MeshSet instance
m = pymeshlab.Mesh(V, F)

# create a new MeshSet
ms = pymeshlab.MeshSet()

# add the mesh to the MeshSet
ms.add_mesh(m, "Mesh")

# Perform isotropic remeshing
ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(5))

# You can extract the remeshed mesh's vertices and faces if needed
remeshed_mesh = ms.current_mesh()
remeshed_V = remeshed_mesh.vertex_matrix()
remeshed_F = remeshed_mesh.face_matrix()

print("V size", remeshed_V.shape)


# Footpoints of remeshed mesh
i_t = time.time()

ui_vj = foot_points(remeshed_V, V, u_pts, v_pts, BSurf)

f_t = time.time()

print("\n\nTime to compute foot points:", f_t - i_t)

V_R = np.zeros((len(ui_vj), 3))
C_uv = np.zeros((len(ui_vj), 3))
l_uv = np.zeros((len(ui_vj), 3))


for i in range(len(ui_vj)):

    V_R[i] = BSurf(ui_vj[i][0], ui_vj[i][1])

    C_uv[i], l_uv[i]  = sph_ln_cong_at_pt(BSurf, rsurf, ui_vj[i][0], ui_vj[i][1])


t1, t2, vc = get_torsal_Mesh(remeshed_V, remeshed_F,  l_uv)

ot1, ot2, bc = get_torsal_Mesh(V, F, or_l)
    

# If you want to save the remeshed mesh to a file
ps.init()

ps.remove_all_structures()

or_mesh = ps.register_surface_mesh("mesh", V, F)
or_mesh.add_vector_quantity("l", or_l, defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.0, 1.0, 0.0))

torsal_dir_show(vc, t1, t2, size=0.02, rad=0.0005,  color=(1,1,1), name="")
torsal_dir_show(bc, ot1, ot2, size=0.02, rad=0.0005,  color=(1,1,1), name="Or")

mesh = ps.register_surface_mesh("remeshed", remeshed_V, remeshed_F)
mesh.add_vector_quantity("l_uv", l_uv, defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.1, 0.0, 0.0))
ps.register_surface_mesh("Proj", V_R, remeshed_F)
ps.register_surface_mesh("C_uv", C_uv, remeshed_F)

ps.show()
