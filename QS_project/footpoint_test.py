# Import the necessary libraries
import os
import sys
from pathlib import Path

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
# Import the necessary libraries for visualization and computation

import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
from scipy.optimize import minimize
from scipy.spatial import KDTree


from time import sleep

# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *

# Local files
from utils.bsplines_functions import *
from utils.visualization import *

# Optimization classes
from energies.BS_LineCong import BS_LC
from energies.BS_LineCong_Orth import BS_LC_Orth
from energies.BS_Torsal import BS_Torsal
from energies.BS_Torsal_Angle import BS_Torsal_Angle

from optimization.Optimizer import Optimizer

# Directory where the data is stored ======================================

# Here you can add the directory where you want to save the results
dir_path = os.getcwd()
print(dir_path)

# Define Bsplines Surface directory
surface_dir = os.path.join(dir_path, "data", "Bsplines_Surfaces")
print("surface dir:", surface_dir)

experiment_dir = os.path.join(dir_path, "experiments")


def closest_grid_points(p, grid_v):
    """
    Find the closest grid point to a given point p
    """
    kd_tree = KDTree(grid_v)

    _, i = kd_tree.query(p)

    return V[i], i

def foot_point(p, V, u_pts, v_pts, Bsp):
    """
    Find the foot point of a point p in the B-spline surface
    """
    #n = len(u_pts)
    m = len(v_pts)
    # Find the closest grid point
    _, idx = closest_grid_points(p, V)

    u_idx = idx//m
    v_idx = idx%m

    x0 = np.hstack((u_pts[u_idx], v_pts[v_idx]))

    # Find the closest point in the B-spline surface
    # Define function to minimize 
    def f(x):
        x = x.reshape(-1, 2)
        xu = x[:, 0]
        xv = x[:, 1]
        
        ev_b = np.zeros((len(xu), 3))
        for i in range(len(xu)):
            ev_b[i] = Bsp(xu[i], xv[i])
        
        
        return np.linalg.norm(ev_b - p)
    
    # Minimize the function
    res = minimize(f, x0)

    print("res: ", res)

    return res.x
# Optimization options ======================================

# Name surface file

# Rhino Test  1
#bspline_surf_name, dir = "Complex_test_S", -1

# Rhino Test 2
#bspline_surf_name, dir = "Complex_test_S2", 1

# Rhino Bad test 
#bspline_surf_name, dir = "Surfjson", -1

# Florian Non change mean Curvature
bspline_surf_name, dir = "Florian", 1

# Sample size
sample = (22, 22)
delta = 0.1
choice_data = 0 # 0: Json , 1: data_hyp2.dat
mid_init = 0  # 0: central_sphere, 1: offset_surface
delta = 0.1

bsp1 = get_spline_data(choice_data, surface_dir, bspline_surf_name)

# Get Grid Information
u_pts, v_pts = sample_grid(sample[0], sample[1], delta=delta)
n_squares = (len(u_pts)-1)*(len(v_pts)-1)

# r_H, n = init_sphere_congruence(mid_init, bsp1, u_pts, v_pts, sample)

# # Fit r_H to a B-spline surface r(u,v)
# r_uv = r_uv_fitting(u_pts, v_pts, r_H)

# Querry point
q = np.array([[1, 1, 1],[1,0,0.2]])

# End of constraints ===================================
V, F = Bspline_to_mesh(bsp1, u_pts, v_pts, sample)


p_q, _ = closest_grid_points(q, V)

foot_pts = foot_point(q, V, u_pts, v_pts, bsp1)

foot_pts = foot_pts.reshape(-1, 2)

# Evaluate 
f_pts = np.zeros((len(foot_pts), 3))

for i in range(len(foot_pts)):
    f_pts[i] = bsp1(foot_pts[i, 0], foot_pts[i, 1])

ps.init()
# Surface
surf = ps.register_surface_mesh("S_uv", V, F)
# INITIAL LC
ps.register_point_cloud("q", np.array(q), radius = 0.007)
ps.register_point_cloud("closest", np.array(p_q), radius = 0.007)
ps.register_point_cloud("foot", f_pts, radius = 0.008)
ps.show()

