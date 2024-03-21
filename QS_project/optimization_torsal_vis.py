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

# Florian Non change mean Curvature
bspline_surf_name, dir = "Florian", 1

# Ellipsoid
#bspline_surf_name, dir = "ellipsoid", -1

# Define the path to the B-spline surface
bspline_surf_path = os.path.join(surface_dir, bspline_surf_name + ".json")
print("bspline_surf_path:", bspline_surf_path)


# Load the B-spline surface
control_points, knots_u, knots_v, order_u, order_v = read_bspline_json(bspline_surf_path)

# # Create the B-splines basis
# basis_u = sp.BSplineBasis(order_u, knots_u) 
# basis_v = sp.BSplineBasis(order_v, knots_v) 


# Mean curvature test
# # Load a file
# data = np.loadtxt("/Users/cisneras/sph_dt.dat", dtype=float)

# surface = approximate_surface(data, 32, 32, 4, 5)

# #data = data.reshape(32, 32, 3)

# order_u = surface.degree_u + 1
# order_v = surface.degree_v + 1
# control_points = surface.ctrlpts
# knots_u = surface.knotvector_u
# knots_v = surface.knotvector_v


# Scale control points
ctrl_pts_shape = control_points.shape
flat_ctrl_pts  = control_points.reshape(-1,3)
norm_ctrl_pts  = normalize_vertices(flat_ctrl_pts, 2)
control_points = norm_ctrl_pts.reshape(ctrl_pts_shape)

# Create the B-splines basis
basis_u = sp.BSplineBasis(order_u, knots_u) 
basis_v = sp.BSplineBasis(order_v, knots_v) 



# Create the B-spline surface
bsp1 = sp.Surface(basis_u, basis_v, control_points)

# Create the B-spline surface
#bsp1 = sp.Surface(basis_u, basis_v, control_points)

# Sample size
sample = (50, 50)

# Angle threshold
angle = 25

u_pts, v_pts = sample_grid(sample[0], sample[1], delta=0.1)


# Compute central spheres radius and normals
c, r_H, H, n = central_spheres(bsp1, u_pts, v_pts) 

n *= dir

#plot_scalar_value(plt, c, r_H, "Mean Curvature")

#n = bsp1.normal(u_pts, v_pts)
#r_H = np.array([ np.cos(u) + np.sin(v) + 2 for u in u_pts for v in v_pts]).reshape(sample[0], sample[1]) 
#r_H = 5*np.ones((sample[0], sample[1]))

# Fit r_H to a B-spline surface r(u,v)
r_uv = r_uv_fitting(u_pts, v_pts, r_H)

# Compute the line congruence
l = dir*line_congruence_uv(bsp1, r_uv, u_pts, v_pts)

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
opt.initialize_optimizer("LM", 0.4, 1)

# Initialize variables
opt.init_variable("rij", cp)
opt.init_variable("mu", 1)
opt.init_variable("l", l.flatten())


# Add the constraint to the optimizer
LC = BS_LC()
# add_constraint(name, args, weight)
opt.add_constraint(LC, args=(bsp1, r_uv, u_pts, v_pts), w=1)

LC_orth = BS_LC_Orth()
opt.add_constraint(LC_orth, args=(bsp1, r_uv, u_pts, v_pts, dir, angle), w=5)

opt.unitize_variable("l", 3, 10)

# Optimize
for i in range(300):
    # Get gradients
    opt.get_gradients() # Compute J and residuals
    opt.optimize() # Solve linear system and update variables

# Get energy per constraint
opt.get_energy_per_constraint()

# Get Line congruence
l, cp = opt.uncurry_X("l", "rij")

# Reshape Line congruence
l = l.reshape(len(u_pts), len(v_pts), 3)
l /= np.linalg.norm(l, axis=2)[:,:,None]

# Angle with normal
ang_normal = np.arccos( np.sum( l*n, axis=2))*180/np.pi

# Histogram
plt.hist(ang_normal.flatten(), bins=50)
plt.show()

# Surface points
S_uv = bsp1(u_pts, v_pts)

# Get quadrilateral grid points
v0 = S_uv[:-1,:-1]
v1 = S_uv[:-1,1:]
v2 = S_uv[1:,1:]
v3 = S_uv[1:,:-1]


# Diagonal vectors
du = v2 - v0
dv = v1 - v3

# Compute baricenter per quadrilateral
baricenter = (v0 + v1 + v2 + v3)/4

sign_l = np.sign(np.sum(l*n, axis=2))
l = sign_l[:,:,None]*l

# Get line congruence points
l0 = l[:-1,:-1]
l1 = l[:-1,1:]
l2 = l[1:,1:]
l3 = l[1:,:-1]


# Get Diagonal vectors
lu = l2 - l0
lv = l1 - l3

# Compute baricenter per quadrilateral
l_c = (l0 + l1 + l2 + l3)/4
# Normalize it
l_c /= np.linalg.norm(l_c, axis=2)[:,:,None]

# Get grid points
u_grid, v_grid = np.meshgrid(u_pts, v_pts, indexing='ij')

r_uv[2] = cp

r_uv_surf = bisplev(u_pts, v_pts, r_uv)
M_ruv = np.stack((u_grid, v_grid, r_uv_surf), axis=2)


lc = l_c.reshape(-1,3)
lu = lu.reshape(-1,3)
lv = lv.reshape(-1,3)
du = du.reshape(-1,3)
dv = dv.reshape(-1,3)
vc_flat = baricenter.reshape(-1,3)
S_flat = S_uv.reshape(-1,3)

t1, t2, _, _, _, _ = torsal_directions(lc, lu, lv, du, dv)

# Torsal
t1 = unit(t1)
t2 = unit(t2)

# Normals of torsal planes
nt1 = np.cross(t1, lc)
nt2 = np.cross(t2, lc)

# Compute angle between nt1, nt2
torsal_angles= np.arccos(np.sum(abs(nt1*nt2),axis=1))*180/np.pi

# Histogram
plt.hist(torsal_angles, bins=50)
plt.show()


# Init figure
fig     = plt.figure()
surface_ax = fig.add_subplot(1,2,1, projection='3d') 
central_ax = fig.add_subplot(1,2,2, projection='3d')

plot_surface(central_ax, M_ruv,  "r(u, v)")
# Plot the B-spline surface
plot_surface(surface_ax, S_uv,  "B-spline Surface Vis")
#plot_scalar_value(surface_ax, S_uv, H, "Mean_Curvature")

t1 = t1
t2 = t2

nt1 = nt1
nt2 = nt2

# Get Grid as quad mesh V and F
V = S_flat
# Faces F_i = [i, i+1, sample[1]*i + i, sample[1]*i + i]
F = np.array([  (sample[1]*j) +np.array([ i, i+1, sample[1] + i + 1, sample[1] + i])   for j in range(sample[1] - 1)for i in range(sample[0] - 1)] )


V_R = V + r_uv_surf.flatten()[:,None]*n.reshape(-1,3)

ps.init()
surf = ps.register_surface_mesh("S_uv", V, F)
central = ps.register_surface_mesh("Mid Surf", V_R, F)
#surf.add_vector_quantity("lc", lc.reshape(-1, 3), defined_on="faces", length=1,  enabled=True, color=(0.5, 0, 0))
#surf.add_vector_quantity("l", l.reshape(-1, 3), defined_on="vertices", length=1,  enabled=True, color=(0.5, 0.5, 0.5))
surf.add_scalar_quantity("Angles", ang_normal.flatten(), defined_on="vertices", enabled=True)
surf.add_scalar_quantity("Torsal Angles", torsal_angles, defined_on="faces", enabled=True)
#surf.add_vector_quantity("n", n.reshape(-1,3), defined_on="vertices", length=0.1,  enabled=True, color=(0, 0, 0))
surf.add_vector_quantity("t1", t1, defined_on="faces", length=0.01,  enabled=True, color=(0, 0, 0))
surf.add_vector_quantity("t2", t2, defined_on="faces", length=0.01,  enabled=True, color=(0, 0, 0))


ps.show()



