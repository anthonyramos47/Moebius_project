# This file is a template for creating a new python file in the hananLab/hanan directory

# Import the necessary libraries
import os
import sys
from pathlib import Path

# Obtain the path HananLab; this is the parent directory of the hananLab/hanan directory
# <Here you can manually add the path direction to the hananLab/hanan directory>
path = "/Users/cisneras/hanan/hananLab/hanan"
sys.path.append(path)

# Verification of the path
print(path)

# Import the necessary libraries for visualization and computation
import igl
import polyscope as ps
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, bisplev, bisplrep


# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *

# Optimization classes
from energies.BS_approx import BS_approx
from optimization.Optimizer import Optimizer


# Here you can add the directory where you want to save the results
dir_path = os.getcwd()
print(dir_path)


# Create a Bspline
# Define control points for the B-spline surface
data_points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                        [0, 1, 1], [1, 1, 2], [2, 1, -1],
                        [0, 2, 0], [1, 2, 0], [2, 2, 0]])

# Create a B-spline surface representation
kx, ky = 2, 2  # Degree of the B-spline in u and v directions
s = 0.0  # Smoothing factor (0 for interpolation, higher values for smoother surfaces)

# Compute the B-spline representation
bsp1, _, _, _ = bisplrep(data_points[:, 0], data_points[:, 1], data_points[:, 2], kx=kx, ky=ky, s=s, full_output=True)

bsp2 = bsp1.copy()
bsp2[2] = bsp2[2] + 4


# Define the grid intervals
grid_intervals = [[0, 2], [0, 2]]
grid_size = [100, 100]

# Initialize Optimizer
opt = Optimizer()

# Add viariables
opt.add_variable("cp", len(bsp2[2]) )

# Initialize optimizer
opt.initialize_optimizer("LM", 0.8, 1)

# Add contraint
bs_approx = BS_approx()
opt.add_constraint(bs_approx, args=(grid_intervals, grid_size, bsp1, bsp2))

# Optimize
for i in range(10):
    opt.get_gradients()
    opt.optimize()

opt.get_energy_per_constraint()


u_vals = bs_approx.u_pts
v_vals = bs_approx.v_pts


surface_points = bisplev(u_vals[:,0], v_vals[0,:], bsp1)

bsp2[2] = opt.X[opt.var_idx["cp"]]
surface_points2 = bisplev(u_vals[:,0], v_vals[0,:], bsp2)

# Plot the B-spline surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2], c='r', marker='o', label='Control Points')

ax.plot_surface(u_vals, v_vals, surface_points, cmap='viridis', alpha=0.8)
ax.plot_surface(u_vals, v_vals, surface_points2, alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
