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
import igl
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import json
import pickle


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

# Optimization options ======================================

# Name surface file

# Rhino Test  1
#bspline_surf_name, dir = "Complex_test_S", -1

# Rhino Test 2
#bspline_surf_name, dir = "Complex_test_S2", 1

# Rhino Bad test 
#bspline_surf_name, dir = "Surfjson", -1

# Florian Non change mean Curvature
#bspline_surf_name, dir = "Florian", 1

# Dat name
#bspline_surf_name, dir = "inv_1", 1
bspline_surf_name, dir = "data_hyp", 1

# Sample size
sample = (25, 25)
delta = 0.3
choice_data = 1 # 0: Json , 1: data_hyp.dat
mid_init = 0  # 0: central_sphere, 1: offset_surface
angle = 25 # Angle threshold with surface
tangle = 45 # Torsal angle threshold for torsal planes
weights = {
    "LC": [1,1], # Line congruence l.cu = 0, l.cv = 0
    "LC_Orth": [2,1], # Line congruence orthogonality with surface
    "Torsal": 1, # Torsal constraint
    "Torsal_Angle": 3 # Torsal angle constraint
}


# Gobal variables ======================================
ang_normal = np.zeros((sample[0], sample[1]))
state = 0
state2 = 0  
counter = 1
init_opt_1 = False
init_opt_2 = False
name_saved = "Results"
iter_per_opt = 20
step_1 = 0.5
step_2 = 0.5


bsp1 = get_spline_data(choice_data, surface_dir, bspline_surf_name)

# Get Grid Information
u_pts, v_pts = sample_grid(sample[0], sample[1], delta=delta)
n_squares = (len(u_pts)-1)*(len(v_pts)-1)

r_H, n = init_sphere_congruence(mid_init, bsp1, u_pts, v_pts, sample)

# Fit r_H to a B-spline surface r(u,v)
r_uv = r_uv_fitting(u_pts, v_pts, r_H)

# Compute the line congruence
l = line_congruence_uv(bsp1, r_uv, u_pts, v_pts)
l = flip(l, n)

# Store initial line congruence for visualization
init_l = l.copy()

# Get the number of control points
cp = r_uv[2].copy()


# End of constraints ===================================

V, F = Bspline_to_mesh(bsp1, u_pts, v_pts)

ps.init()
# Surface
surf = ps.register_surface_mesh("S_uv", V, F)
# INITIAL LC
surf.add_vector_quantity("init_l", init_l.reshape(-1, 3), vectortype='ambient', enabled=False, color=(0.0, 0.0, 0.1))
ps.set_user_callback(optimization)
ps.show()

