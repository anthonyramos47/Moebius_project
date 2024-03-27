# This file is a template for creating a new python file in the hananLab/hanan directory
# Import the necessary libraries
import os
import sys
from pathlib import Path


path = "/Users/cisneras/hanan/hananLab/hanan"
sys.path.append(path)

# Verification of the path
print(path)


# Import the necessary libraries for visualization and computation
import igl
import polyscope as ps
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import json
from scipy.interpolate import bisplev
from geomdl.fitting import approximate_surface

# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *
from geometry.bsplines_functions import *

# Optimization classes
from energies.BS_LineCong import BS_LC
from energies.BS_LineCong_Orth import BS_LC_Orth
from optimization.Optimizer import Optimizer


# Here you can add the directory where you want to save the results
dir_path = os.getcwd()
print(dir_path)

# Define Bsplines Surface directory
surface_dir = os.path.join(dir_path, "data", "Bsplines_Surfaces")

print("surface dir:", surface_dir)
# Name surface file

# Hyperboloid
data_bspline = "data_hyp.dat"

data_bspline = os.path.join(surface_dir, data_bspline)


bsp = approx_surface_from_data(data_bspline)


# Sample size
sample = (20, 20)

# Angle threshold
angle = 25

u_pts, v_pts = sample_grid(sample[0], sample[1], delta=0.3)


# Compute central spheres radius and normals
c, r_H, H, n = central_spheres(bsp, u_pts, v_pts) 


# Compute grid points
s_uv = bsp(u_pts, v_pts)

S_flat = s_uv.reshape(-1, 3)

# Mesh Visualization ========================== 

# Get Grid as quad mesh V and F
V = S_flat
# Faces F_i = [i, i+1, sample[1]*i + i, sample[1]*i + i]
F = np.array([  (sample[1]*j) +np.array([ i, i+1, sample[1] + i + 1, sample[1] + i])   for j in range(sample[1] - 1)for i in range(sample[0] - 1)] )


ps.init()
surf = ps.register_surface_mesh("S_uv", V, F)

surf.add_scalar_quantity("H", H.flatten(), enabled=True)

ps.show()



