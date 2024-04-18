import numpy as np
#import pymeshlab
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
from geometry.mesh import Mesh
from utils.bsplines_functions import *



# Create the parser
parser = argparse.ArgumentParser(description="Visualizer Parser")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
name = parser.parse_args().file_name
pickle_name = name+'.pickle'

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

# Compute the barycenter at the remeshed mesh
vc = np.sum(f_pts[ffF], axis=1)/4

# Compute radius of spheres
rads = np.linalg.norm(vc - cc, axis=1)

# Create HE mesh 
mesh = Mesh()
mesh.make_mesh(f_pts, ffF)

# Get the face-face adjacency list 
f_f_adj = mesh.face_face_adjacency_list()
# Get inner faces indices
inn_f = mesh.inner_faces()

# Open file
exp_file = open(os.path.join(remeshing_dir, name+'_sphs.dat'), 'w')

# Write inn_f in the first line
for i in inn_f:
    exp_file.write(f"{i} ")
exp_file.write("\n")

for i in range(len(ffF)):

    exp_file.write(f"{cc[i][0]} {cc[i][1]} {cc[i][2]} {rads[i]} {vc[i][0]} {vc[i][1]} {vc[i][2]} ")

    for j in f_f_adj[i]:
        exp_file.write(f"{j} ")

    exp_file.write("\n")

exp_file.close()



ps.init()

ps.remove_all_structures()

for idx in [62,64,65,132,128]:
    #idx = np.random.randint(0, len(ffF))
    sph = ps.register_point_cloud(f"s_"+str(idx), np.array([cc[idx]]), transparency=0.4, color=(0.1, 0.1, 0.1))
    r = np.linalg.norm(vc[idx] - cc[idx])
    sph.set_radius(r, relative=False)
    

or_mesh = ps.register_surface_mesh("mesh", ffV, ffF)
ps.register_surface_mesh("S_uv", V, F)
ps.register_surface_mesh("foot_pts", f_pts, ffF)
ps.register_surface_mesh("C_uv", VR, ffF)

ps.register_point_cloud("C centers", cc)
ps.register_point_cloud("V centers", vc)


ps.show()
