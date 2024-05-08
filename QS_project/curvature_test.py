# This file is a template for creating a new python file in the hananLab/hanan directory

# Import the necessary libraries
import os
import sys
import argparse
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

# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *
from utils.bsplines_functions import *

# Optimization classes
from energies.BS_approx import BS_approx
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

#bspline_surf_name, dir = "Test_SubD", 1
#bspline_surf_name, dir = "Sph_inv_2", 1


# Rhino Bad test 
#bspline_surf_name, dir = "Surfjson", -1
#bspline_surf_name, dir = "Sample_C", 1
#bspline_surf_name, dir = "Tunel", -1


# Create the parser
parser = argparse.ArgumentParser(description="Curvature Vis")
# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

parser.add_argument('deltaumin', type=float, help='delta value')
parser.add_argument('deltaumax', type=float, help='delta value')
parser.add_argument('deltavmin', type=float, help='delta value')
parser.add_argument('deltavmax', type=float, help='delta value')

bspline_surf_name = parser.parse_args().file_name

# Define the path to the B-spline surface
bspline_surf_path = os.path.join(surface_dir, bspline_surf_name + ".json")
print("bspline_surf_path:", bspline_surf_path)

# Sample in each direction
sample = 1000

# Load the B-spline surface
control_points, knots_u, knots_v, order_u, order_v = read_bspline_json(bspline_surf_path)


# Create the B-splines basis
basis_u = sp.BSplineBasis(order_u, knots_u) 
basis_v = sp.BSplineBasis(order_v, knots_v) 

# Create the B-spline surface
bsp1 = sp.Surface(basis_u, basis_v, control_points)

# Sample the grid points to evaluate the B-spline surface
u_vals, v_vals = sample_grid(sample, sample, deltaum=parser.parse_args().deltaumin, deltauM=parser.parse_args().deltaumax, deltavm = parser.parse_args().deltavmin, deltavM = parser.parse_args().deltavmax)


# Evaluate the B-spline surface
eval_surf = bsp1(u_vals, v_vals)

# Compute the curvature
K, H, _ = curvatures_par(bsp1, u_vals, v_vals)


hist_fig = plt.figure()
hist = hist_fig.add_subplot(1,1,1)
hist.set_title("Mean Curvature")
hist.hist(H.flatten(), bins=20)


# # Init figure
fig       = plt.figure()
surface_ax = fig.add_subplot(1,2,1, projection='3d') 
central_ax = fig.add_subplot(1,2,2, projection='3d')

# # Compute central spheres centers and radius
# c_z, r_z, H, _= central_spheres(bsp1, u_vals, v_vals)


# # Create r(u,v)  Bspline surface
# r_uv = r_uv_fitting(u_vals, v_vals, r_z)

# # Get grid points
# u_pts, v_pts = np.meshgrid(u_vals, v_vals, indexing='ij')

# r_uv_surf = bisplev(u_vals, v_vals, r_uv)

# X_ruv = np.stack((u_pts, v_pts, r_uv_surf), axis=2)

# nu, nv = normal_derivatives_uv(bsp1, u_vals, v_vals)

# cu, cv = sphere_congruence_derivatives(bsp1, r_uv, u_vals, v_vals)


# # Line congruence of offset surfaces
# #cof, rof = offset_spheres(bsp1, u_vals, v_vals, 5)

# # Fit rof to a B-spline surface
# # r_uv_of = r_uv_fitting(u_vals, v_vals, rof)
# l_uv = line_congruence_uv(bsp1, r_uv, u_vals, v_vals)


# eval_surf_flat = eval_surf.reshape(-1,3)
# l_uv_flat = l_uv.reshape(-1,3)

# # # Offset surface
# # s_n = bsp1.normal(u_vals, v_vals)
# # offset_surf = eval_surf  + dir*5*s_n

# # Plot the B-spline surface
# plot_surface(surface_ax, eval_surf,  "B-spline Surface")


# #add_control_points(surface_ax, control_points)

# # Plot the central spheres
# #plot_surface(central_ax, c_z, "Central Surface")

# #add_control_points(surface_ax, control_points)

# # Plot the offset surface
# #plot_surface(/Users/cisneras/Downloads/Assignment_3.pdfsurface_ax, offset_surf, None, "B-spline Surface Offset")
plot_scalar_value(surface_ax, eval_surf,  H, "Mean")


plt.show()
# # Plot 
# for _ in range(100):
    
#     id = np.random.randint(0, len(l_uv_flat))

#     l = l_uv_flat[id]
#     p1 = eval_surf_flat[id]
#     p2 = dir*5*l

#     #sph_x, sph_y, sph_z = drawSphere(c_z_flat[id][0], c_z_flat[id][1], c_z_flat[id][2],  abs(r_z_flat[id]))
#     #    surface_ax.plot_surface(sph_x, sph_y, sph_z, color='r', alpha=0.3)
#     #surface_ax.quiver(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], color='black', alpha=0.5, pivot='tail')
#     central_ax.quiver(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], color='black', alpha=0.5, pivot='tail')

# plt.show()
