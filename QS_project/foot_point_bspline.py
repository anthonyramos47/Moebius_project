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
from utils.bsplines_functions import *


path = os.getcwd()
print(path)

# experiment dir
exp_dir = os.path.join(path, 'experiments')

objs_dir = '/Users/cisneras/hanan/hananLab'

# Create the parser
parser = argparse.ArgumentParser(description="Visualizer Parser")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
name = parser.parse_args().file_name
pickle_name = name+'.pickle'

# Frame Field remeshed obj
remeshed_obj = name+'_FFR.obj'

ffV, ffF = read_obj(os.path.join(objs_dir, remeshed_obj))


def load_data():
    """ Function to load the data from a pickle file
    """
    with open(os.path.join(exp_dir, pickle_name), 'rb') as f:
        data = load(f)
    return data


data = load_data()

u_pts = data['u_pts']
v_pts = data['v_pts']

BSurf = data['surf']
rsurf = data['r_uv']


sample = (len(u_pts), len(v_pts))   
# End of constraints ===================================
V, F = Bspline_to_mesh(BSurf, u_pts, v_pts, sample)



p_q, _ = closest_grid_points(ffV, V)

foot_pts = foot_points(ffV, V, u_pts, v_pts, BSurf)

foot_pts = foot_pts.reshape(-1, 2)

# Evaluate 
f_pts = np.zeros((len(foot_pts), 3))
r_pts = np.zeros((len(foot_pts), 3))
n_dir = np.zeros((len(foot_pts), 3))

for i in range(len(foot_pts)):
    f_pts[i] = BSurf(foot_pts[i, 0], foot_pts[i, 1])
    n_dir[i] = BSurf.normal(foot_pts[i, 0], foot_pts[i, 1])
    
    r_pts[i] = bisplev(foot_pts[i, 0], foot_pts[i, 1], rsurf)

VR = f_pts + r_pts[:,None]*n_dir

VR = VR.reshape(-1, 3)

c0, c1, c2, c3 = VR[ffF[:, 0]], VR[ffF[:, 1]], VR[ffF[:, 2]], VR[ffF[:, 3]]

cc = (c0 + c1 + c2 + c3)/4

vc = np.sum(f_pts[ffF], axis=1)/4




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
