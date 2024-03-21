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

# Rhino Test  1
#bspline_surf_name, dir = "Complex_test_S", -1

# Rhino Test 2
#bspline_surf_name, dir = "Complex_test_S2", 1

# Rhino Bad test 
#bspline_surf_name, dir = "Surfjson", -1

# # Ellipsoid
# bspline_surf_name, dir = "ellipsoid", -1

# # Define the path to the B-spline surface
# bspline_surf_path = os.path.join(surface_dir, bspline_surf_name + ".json")
# print("bspline_surf_path:", bspline_surf_path)


# # Load the B-spline surface
# control_points, knots_u, knots_v, order_u, order_v = read_bspline_json(bspline_surf_path)

# # Create the B-splines basis
# basis_u = sp.BSplineBasis(order_u, knots_u) 
# basis_v = sp.BSplineBasis(order_v, knots_v) 


# Mean curvature test

# Load a file
data = np.loadtxt("/Users/cisneras/sph_dt.dat", dtype=float)

surface = approximate_surface(data, 32, 32, 4, 5)

#data = data.reshape(32, 32, 3)

ord_u = surface.degree_u + 1
ord_v = surface.degree_v + 1
ct_pts = surface.ctrlpts
knots_u = surface.knotvector_u
knots_v = surface.knotvector_v

# Create the B-splines basis
basis_u = sp.BSplineBasis(ord_u, knots_u) 
basis_v = sp.BSplineBasis(ord_v, knots_v) 

# Create the B-spline surface
bsp1 = sp.Surface(basis_u, basis_v, ct_pts)

# Create the B-spline surface
#bsp1 = sp.Surface(basis_u, basis_v, control_points)

# Sample size
sample = (30, 30)

# Angle threshold
angle = 45

u_pts, v_pts = sample_grid(sample[0], sample[1], delta=0.2)

# Compute central spheres radius and normals
c, r_H, H, n = central_spheres(bsp1, u_pts, v_pts) 

#plot_scalar_value(plt, c, r_H, "Mean Curvature")

#n = bsp1.normal(u_pts, v_pts)

#r_H = np.array([ np.cos(u) + np.sin(v) + 2 for u in u_pts for v in v_pts]).reshape(sample[0], sample[1]) 

# Add noise
#r_H += np.random.normal(0, 0.1, r_H.shape)

# Fit r_H to a B-spline surface r(u,v)
r_uv = r_uv_fitting(u_pts, v_pts, r_H)
#r_uv = r_uv_fitting(aux_x, aux_y, r_H)

# Get the number of control points
cp = r_uv[2].copy()

# Create the optimizer
opt = Optimizer()

# Add variables to the optimizer
opt.add_variable("rij", len(cp)) # Control points
opt.add_variable("l", 3*len(u_pts)*len(v_pts))
# Dummy variables
opt.add_variable("mu" , len(u_pts)*len(v_pts)) 

# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", 0.7, 1)

# Initialize variables
opt.init_variable("rij", cp)
opt.init_variable("mu", 0.5)


# Add the constraint to the optimizer
LC = BS_LC()
# add_constraint(name, args, weight)
opt.add_constraint(LC, args=(bsp1, r_uv, u_pts, v_pts), w=1.0)

LC_orth = BS_LC_Orth()
opt.add_constraint(LC_orth, args=(bsp1, r_uv, u_pts, v_pts, angle), w=4.0)


print("Lc:", opt.uncurry_X("l"))

opt.unitize_variable("l", 3)

# Optimize
for i in range(1):
    # Get gradients
    opt.get_gradients() # Compute J and residuals
    opt.optimize() # Solve linear system and update variables

# Print in command line the energy per constraint and best energy
opt.get_energy_per_constraint()

l, u, cp = opt.uncurry_X("l", "mu", "cp")
# Visualization

l = l.reshape(len(u_pts), len(v_pts), 3)


l /= np.linalg.norm(l, axis=2)[:,:,None]

# Compute angles with n
angles = np.arccos(np.sum(abs(l*n), axis=2))*180/np.pi

# Histogram of angles
plt.hist(angles.flatten(), bins=20)
plt.show()

# Get grid points
u_grid, v_grid = np.meshgrid(u_pts, v_pts, indexing='ij')

r_uv[2] = cp

r_uv_surf = bisplev(u_pts, v_pts, r_uv)
r_u = bisplev(u_pts, v_pts, r_uv, dx=1, dy=0)
r_v = bisplev(u_pts, v_pts, r_uv, dx=0, dy=1)

M_ruv = np.stack((u_grid, v_grid, r_uv_surf), axis=2)

init_pts = np.stack((u_grid, v_grid, r_H), axis=2)


X = bsp1(u_pts, v_pts)

print("H shape:", H.shape)
plt.hist(H.flatten(), bins=60, color='r')
plt.show()

# Init figure
fig     = plt.figure()
surface_ax = fig.add_subplot(1,2,1, projection='3d') 
#central_ax = fig.add_subplot(1,2,2, projection='3d')


# Plot the B-spline surface
plot_surface(surface_ax, M_ruv,  "B-spline Surface")
#plot_scalar_value(surface_ax, X,  H, "Mean_Curvature")

# print("data shape", data.shape)

#add_control_points(surface_ax, data)
#plot_surface(surface_ax, M_ruv,  "r(u,v) Surface")
#plot_surface(surface_ax, c,  "Central Surface")


# r_pts_flat = M_ruv.reshape(-1,3)
# r_u_flat = r_u.reshape(-1)
# r_v_flat = r_v.reshape(-1)

# for i in range(len(r_pts_flat)):

    
#     print("r_u_flat[i]:", r_u_flat[i])
#     dir_u = unit(np.array([1, 0, r_u_flat[i]]))
#     dir_v = unit(np.array([0, 1, r_v_flat[i]]))


#     surface_ax.quiver(r_pts_flat[i,0], r_pts_flat[i,1], r_pts_flat[i,2], dir_u[0], dir_u[1], dir_u[2], color='r')

#     surface_ax.quiver(r_pts_flat[i,0], r_pts_flat[i,1], r_pts_flat[i,2], dir_v[0], dir_v[1], dir_v[2], color='b')

#surface_ax.scatter(init_pts[:,:,0], init_pts[:,:,1], init_pts[:,:,2],  c='r', s=10)

plt.show()