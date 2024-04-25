import numpy as np
#import pymeshlab
import polyscope as ps
import argparse
import pymeshlab
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

BSurf = data['surf']
rsurf = data['r_uv']


# Get the vertices and faces of the mesh
u_pts = np.linspace(0, 1, 300)
v_pts = np.linspace(0, 1, 300)

# Get the vertices and faces of the mesh

V, F = Bspline_to_mesh(BSurf, u_pts, v_pts)

TF = np.array(triangulate_quads(F))

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
# ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(0.8))

# # You can extract the remeshed mesh's vertices and faces if needed
# remeshed_mesh = ms.current_mesh()
# remeshed_V = remeshed_mesh.vertex_matrix()
# remeshed_F = remeshed_mesh.face_matrix()

# print("V size", remeshed_V.shape)



#ot1, ot2, bc = get_torsal_Mesh(V, F, or_l)


# V_R = 
# C_uv = np.zeros((len(ui_vj), 3))
# l_uv = np.zeros((len(ui_vj), 3))


# If you want to save the remeshed mesh to a file
ps.init()

ps.remove_all_structures()

ps.register_surface_mesh("mesh", V, F)
ps.register_surface_mesh("Remeshed", V, TF)

#ps.register_surface_mesh("Proj", V_R, TF)
#ps.register_surface_mesh("C_uv", C_uv, TF)
ps.show()

#save_torsal(vc, t1, t2, path=save_dir)
