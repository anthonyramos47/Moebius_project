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
from matplotlib.colors import ListedColormap, Normalize
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


# def f(u,v):

#     lam = 2
#     # Parametric equations for the surface
#     x = u
#     y = v
#     z = 0.5 * lam * u - 0.5 * np.arctan(lam * v)

#     return [x, y, z]


# # Create a Bspline
# # Define control points for the B-spline surface
# data_points = np.array([f(u, v) for u in np.linspace(-5, 5, 20) for v in np.linspace(-5, 5, 20)])

v, _ = igl.read_triangle_mesh("/Users/cisneras/hanan/hananLab/QS_project/data/Roof.obj")

data_points = v

fstd = 1.4
min_x, max_x = np.mean(data_points[:,0]) - fstd*np.std(data_points[:,0]), np.mean(data_points[:,0]) + fstd*np.std(data_points[:,0])
min_y, max_y = np.mean(data_points[:,1]) - fstd*np.std(data_points[:,1]), np.mean(data_points[:,1]) + fstd*np.std(data_points[:,1])

# Create a B-spline surface representation
kx, ky = 5, 5 # Degree of the B-spline in u and v directions
s = 1.0  # Smoothing factor (0 for interpolation, higher values for smoother surfaces)

# Compute the B-spline representation
bsp1, _, _, _ = bisplrep(data_points[:, 0], data_points[:, 1], data_points[:, 2], kx=kx, ky=ky, s=s, full_output=True)


# Create the grid
u_int = np.linspace(min_x , max_x, 20)
v_int = np.linspace(min_y , max_y, 20)
u_vals, v_vals = np.meshgrid(u_int, v_int, indexing='ij')

surface_points = bisplev(u_vals[:,0], v_vals[0,:], bsp1)

# Compute Mean Curvature
GK, H = curvatures(bsp1, u_vals, v_vals)

# Get shape of the grid
m, n = u_vals.shape[0], v_vals.shape[1]

# Central sphere radius
c_bsp = GK

# Fit a B-spline to the central spheres 
kx, ky = 5, 5  # Degree of the B-spline in u and v directions
s = 2.0  # Smoothing factor (0 for interpolation, higher values for smoother surfaces)

# Compute the B-spline representation
bsp2, _, _, _ = bisplrep(u_vals.flatten(), v_vals.flatten(), c_bsp, kx=kx, ky=ky, s=s, full_output=True)

# Evaluate the B-spline surface
c_bsp = bisplev(u_vals[:,0], v_vals[0,:], bsp2)
#c_bsp = np.random.rand(m, n)


cmap = plt.cm.RdBu # You can choose any existing colormap as a starting point
norm = Normalize(vmin=c_bsp.min(), vmax=c_bsp.max())


# Set the limits of the scalar values
vmin = c_bsp.min()
vmax = c_bsp.max()


print("Cuvatures ", max(GK), min(GK))
print("c_bsp", np.max(c_bsp))
print("vmin", vmin)
print("vmax", vmax)
print("av ", c_bsp.mean())

# Plot the B-spline surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2], c='r', marker='o', label='Control Points')

surf = ax.plot_surface(u_vals, v_vals, surface_points, cmap =plt.cm.plasma,  facecolors=cmap(norm(c_bsp)), alpha=0.8, vmin=vmin, vmax=vmax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

xmin, xmax = np.min(u_vals), np.max(u_vals)
ymin, ymax = np.min(v_vals), np.max(v_vals)
zmin, zmax = np.min(surface_points), np.max(surface_points)

# Fix box aspect
ax.set_box_aspect([xmax-xmin,ymax - ymin, zmax- zmin]/np.max([xmax-xmin, ymax-ymin, zmax-zmin]))
ax.set_title('B-spline surface')


# Add colorbar for reference
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_array(c_bsp)
colorbar = plt.colorbar(mappable, ax=ax, orientation='vertical', shrink=0.6)
colorbar.set_label('Scalar Values')

plt.show()