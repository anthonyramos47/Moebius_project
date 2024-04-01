# This file is a template for creating a new python file in the hananLab/hanan directory

# Ideas reorient before second optimization
# Verify n is passed to bouth lc constriant wiht orientation
# Check if reoreintation change energy 
# Implement energy for positivity ln



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
from energies.BS_Torsal import BS_Torsal
from energies.BS_Torsal_Angle import BS_Torsal_Angle
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


# Parameters optimization

# Sample size
sample = (20, 20)
delta = 0.1
angle = 25 # Angle threshold with surface
tangle = 60 # Torsal angle threshold for torsal planes
choice_data = 0 # 0: data_hyp.dat, 1: data_hyp2.dat
mid_init = 0  # 0: central_sphere, 1: offset_surface
opt_1_it = 200 # Number of iterations first optimization
opt_2_it = 2 # Number of iterations second optimization
weights = {
    "LC": [0.00001,1], # Line congruence l.cu = 0, l.cv = 0
    "LC_Orth": [10,0], # Line congruence orthogonality with surface
    "Torsal": 0, # Torsal constraint
    "Torsal_Angle": 0 # Torsal angle constraint
}


if choice_data == 0:
    # Define the path to the B-spline surface
    bspline_surf_path = os.path.join(surface_dir, bspline_surf_name + ".json")
    print("bspline_surf_path:", bspline_surf_path)

    # Load the B-spline surface
    control_points, knots_u, knots_v, order_u, order_v = read_bspline_json(bspline_surf_path)

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
else:
    data_bspline = "data_hyp.dat"
    dir = 1
    data_bspline = os.path.join(surface_dir, data_bspline)
    bsp1 = approx_surface_from_data(data_bspline)


u_pts, v_pts = sample_grid(sample[0], sample[1], delta=delta)
# Number of squares
n_squares = (len(u_pts)-1)*(len(v_pts)-1)


if mid_init == 0:
    # Compute central spheres radius and normals
    c, r_H, H, n = central_spheres(bsp1, u_pts, v_pts) 
else: 
    n = bsp1.normal(u_pts, v_pts)
    r_H = 5*np.ones((sample[0], sample[1])) 


# Fit r_H to a B-spline surface r(u,v)
r_uv = r_uv_fitting(u_pts, v_pts, r_H)

r_uv_0 = bisplev(u_pts, v_pts, r_uv)

# Compute the line congruence
l = line_congruence_uv(bsp1, r_uv, u_pts, v_pts)

# Store initial line congruence for visualization
init_l = l.copy()

init_flipped = np.sum(np.sum(l*n, axis=2) < 0)

# Fix sign with normal
l = np.sign(np.sum(l*n, axis=2))[:,:,None]*l

after_flippint = np.sum(np.sum(l*n, axis=2) < 0)

# Get the number of control points
cp = r_uv[2].copy()

# Create the optimizer
opt = Optimizer()

# Add variables to the optimizer
opt.add_variable("rij", len(cp)) # Control points
opt.add_variable("l"  , 3*len(u_pts)*len(v_pts))
# Dummy variables                              
opt.add_variable("mu" , len(u_pts)*len(v_pts)  )


# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", 0.7, 0)

# Initialize variables
opt.init_variable("rij" ,         cp )
opt.init_variable("mu"  ,         2 )
opt.init_variable("l"   , l.flatten())


# Constraints ==========================================

# Line congruence l.cu, l.cv = 0
LC = BS_LC()
opt.add_constraint(LC, args=(bsp1, r_uv, u_pts, v_pts), w=weights["LC"][0])

# Line cong orthgonality with surface s(u,v)
LC_orth = BS_LC_Orth()
opt.add_constraint(LC_orth, args=(bsp1, r_uv, u_pts, v_pts, angle), w=weights["LC_Orth"][0])

# Define unit variables
opt.unitize_variable("l", 3, 10)

# Optimize
for i in range(opt_1_it):
    # Get gradients
    opt.get_gradients() # Compute J and residuals
    opt.optimize() # Solve linear system and update variables


# PRINT RESULTS FIRST OPTIMIZATION
print("First Optimization")
# Get energy per constraint
opt.get_energy_per_constraint()


# Copy previous X from optimization
f_l, f_cp, f_mu = opt.uncurry_X("l", "rij", "mu")

f_l = f_l.reshape(-1,3)
n_flat = n.reshape(-1,3)

after_opt_l =  np.sum(np.sum(f_l*n_flat, axis=1) < 0)

# Fix direction with normal
f_l = np.sign(np.sum(f_l*n_flat, axis=1))[:,None]*f_l

after_flip_opt_l = np.sum(np.sum(f_l*n_flat, axis=1) < 0)




# Create the optimizer
opt = Optimizer()

# Add variables to the optimizer
opt.add_variable("rij" , len(cp)) # Control points
opt.add_variable("l"   , 3*len(u_pts)*len(v_pts))
# Dummy variables
opt.add_variable("mu"    , len(u_pts)*len(v_pts)) 
opt.add_variable("nt1"   , 3*n_squares  ) 
opt.add_variable("nt2"   , 3*n_squares  )
opt.add_variable("u1"    , n_squares    )
opt.add_variable("u2"    , n_squares    )
opt.add_variable("v1"    , n_squares    )
opt.add_variable("v2"    , n_squares    )
opt.add_variable("theta" , n_squares    )


# Initialize Optimizer
opt.initialize_optimizer("LM", 0.7, 1)

# Init variables 
opt.init_variable("theta" , 5)
opt.init_variable("l"     , f_l.flatten())  
opt.init_variable("rij"   , f_cp)
opt.init_variable("mu"    , f_mu)

r_uv[2] = f_cp

# Line congruence l.cu, l.cv = 0
LC = BS_LC()
opt.add_constraint(LC, args=(bsp1, r_uv, u_pts, v_pts), w=weights["LC"][0])

# Line cong orthgonality with surface s(u,v)
LC_orth = BS_LC_Orth()
opt.add_constraint(LC_orth, args=(bsp1, r_uv, u_pts, v_pts, angle), w=weights["LC_Orth"][1])


# Torsal constraint 
LC_torsal = BS_Torsal()
opt.add_constraint(LC_torsal, args=(bsp1, u_pts, v_pts, n, sample), w=weights["Torsal"])

# Torsal angle constraint
LC_torsal_ang = BS_Torsal_Angle()
opt.add_constraint(LC_torsal_ang, args=(tangle, 0), w=weights["Torsal_Angle"])


opt.unitize_variable("l", 3, 10)
opt.unitize_variable("nt1", 3, 10)
opt.unitize_variable("nt2", 3, 10)

# Optimize
for i in range(opt_2_it):
    # Get gradients
    opt.get_gradients() # Compute J and residuals
    opt.optimize() # Solve linear system and update variables

# PRINT RESULTS FIRST OPTIMIZATION
print("Second Optimization")
# Get energy per constraint
opt.get_energy_per_constraint()


# Get Line congruence
l, cp, tu1, tu2, tv1, tv2, nt1, nt2, theta = opt.uncurry_X("l", "rij", "u1", "u2", "v1", "v2", "nt1", "nt2", "theta")

# Reshape Line congruence
l = l.reshape(len(u_pts), len(v_pts), 3)
l /= np.linalg.norm(l, axis=2)[:,:,None]
# Reortient line congruence
final_flipped = np.sum(np.sum(l*n, axis=2) < 0)

# Print details about flipped lines
print("Line Flip report: \n")
print("Initial flipped: ", init_flipped, "After flipped: ", after_flippint, "\nAfter optimization flipped: ", after_opt_l, "After flip opt: ", after_flip_opt_l, "\nFinal flipped:", final_flipped)
print("\n===============================\n")


#l = np.sign(np.sum(l*n, axis=2))[:,:,None]*l

# Angle with normal
ang_normal = np.arccos( np.sum( l*n, axis=2) )*180/np.pi

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

# Unpdate control points of r(u,v)
r_uv[2] = cp
r_uv_surf = bisplev(u_pts, v_pts, r_uv)

# Get line congruence at barycenter and derivatives
lc, lu, lv= lc_info_at_grid_points(l)

# Reshape quantities
lc = lc.reshape(-1,3)
lu = lu.reshape(-1,3)
lv = lv.reshape(-1,3)
du = du.reshape(-1,3)
dv = dv.reshape(-1,3)
vc_flat = baricenter.reshape(-1,3)
S_flat = S_uv.reshape(-1,3)


# OPTIMIZED TORSAL DIR
opt1 = tu1[:,None]*du + tv1[:,None]*dv
opt2 = tu2[:,None]*du + tv2[:,None]*dv

olt1 = tu1[:,None]*lu + tv1[:,None]*lv
olt2 = tu2[:,None]*lu + tv2[:,None]*lv

planarity_opt = 0.5*(planarity_check(opt1, olt1, lc) + planarity_check(opt2, olt2, lc))


# ANALYTICAL TORSAL DIR
t1, t2, u1, v1, u2, v2, neg = torsal_directions(lc, lu, lv, du, dv)
# Get number of valid torsal directions 
valid_torsal = np.zeros((len(u_pts)-1)* (len(v_pts)-1))
valid_torsal[neg] = 1

lt1 = u1[:,None]*lu + v1[:,None]*lv
lt2 = u2[:,None]*lu + v2[:,None]*lv

planarity_anal = 0.5*(planarity_check(t1, lt1, lc) + planarity_check(t2, lt2, lc))

# Normalize torsal directions
# Torsal
t1 = unit(t1)
t2 = unit(t2)
nt1 = unit( nt1.reshape(-1,3) )
nt2 = unit( nt2.reshape(-1,3) )


# Compute angle between nt1, nt2
torsal_angles= np.arccos(np.abs( np.sum(nt1*nt2,axis=1) ) )*180/np.pi

print("Torsal energy:", np.linalg.norm(np.sum(nt1*nt2,axis=1)**2 - np.cos(tangle*np.pi/180)**2 - theta**2))
# Histograms: ======= 

# Histogram
fig, (norm_hist, angle_hist) = plt.subplots(2)
fig.tight_layout(pad=3.0)
norm_hist.hist(ang_normal.flatten(), bins=50)
norm_hist.set_title("Surface Normal Angles")
angle_hist.hist(torsal_angles, bins=50)
angle_hist.set_title("Torsal Angles")
angle_hist.set_xlabel("Angle (Degrees)")
plt.show()


# Mesh Visualization ========================== 

# Get Grid as quad mesh V and F
V = S_flat
# Faces F_i = [i, i+1, sample[1]*i + i, sample[1]*i + i]
F = np.array([  (sample[1]*j) +np.array([ i, i+1, sample[1] + i + 1, sample[1] + i])   for j in range(sample[1] - 1)for i in range(sample[0] - 1)] )

V_R = V + r_uv_surf.flatten()[:,None]*n.reshape(-1,3)


# Visualize r(u,v)
u_grid, v_grid = np.meshgrid(u_pts, v_pts, indexing='ij')
R_vis_ruv = np.stack((u_grid, v_grid, r_uv_surf), axis=2) # Optimized r(u,v)
R0_vis_ruv = np.stack((u_grid, v_grid, r_uv_0), axis=2) # Initial r(u,v)

# Get min and max values of R(u,v)
min_r = np.min(r_uv_surf)
max_r = np.max(r_uv_surf)

fig     = plt.figure()
surface_ax = fig.add_subplot(1,1,1, projection='3d') 
#plot_surface(surface_ax, R0_vis_ruv,  "r_0(u,v)")
plot_surface(surface_ax, R_vis_ruv,   "r(u,v)")
plt.title("R(u,v)")
# Set limits
surface_ax.set_xlim([-0.1, 1.1])
surface_ax.set_ylim([-0.1, 1.1])
surface_ax.set_zlim([min_r, 1.1*max_r])
plt.show()


ps.init()
surf = ps.register_surface_mesh("S_uv", V, F)
central = ps.register_surface_mesh("Mid Surf", V_R, F)

# LC 
surf.add_vector_quantity("lc", lc.reshape(-1, 3), defined_on="faces", length=1,  enabled=False, color=(0, 0.2, 0))
# INITIAL LC
surf.add_vector_quantity("init_l", init_l.reshape(-1, 3), vectortype='ambient', enabled=False, color=(0.0, 0.0, 0.1))

# OPTIMIZED LC
surf.add_vector_quantity("l", l.reshape(-1, 3), defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.1, 0.0, 0.0))

# ANGLES WITH NORMAL SCALAR FIELD
surf.add_scalar_quantity("Angles", ang_normal.flatten(), defined_on="vertices", enabled=True)

# VALID TORSAL DIRECTIONS SCALAR FIELD
surf.add_scalar_quantity("Valid Torsal", valid_torsal.flatten(), defined_on="faces", enabled=True)

# OPTIMIZED PLANARITY SCALAR FIELD
surf.add_scalar_quantity("Planarity", planarity_opt, defined_on="faces", enabled=True)

# ANALYTICAL PLANARITY SCALAR FIELD
surf.add_scalar_quantity("Planarity Anal", planarity_anal, defined_on="faces", enabled=True)

# MEAN CURVATURE SCALAR FIELD
if mid_init == 0:
    surf.add_scalar_quantity("Mean Curvature", H.flatten(), defined_on="vertices", enabled=True)


# TORSAL PLANES SCALAR FIELD
surf.add_scalar_quantity("Torsal Angles", torsal_angles, defined_on="faces", enabled=True) 
# INITIAL RADIUS SCALAR FIELD
#surf.add_scalar_quantity("Initial Radius", init_r_uv.flatten(), defined_on="vertices", enabled=True)

# OPTIMIZED RADIUS SCALAR FIELD 
surf.add_scalar_quantity("Radius", r_uv_surf.flatten(), defined_on="vertices", enabled=True)

# Add torsal planes directions
surf.add_vector_quantity("PT 1", nt1, defined_on="faces", length=0.01,  enabled=True, color=(0.4, 0.0, 0.0))
surf.add_vector_quantity("PT 2", nt2, defined_on="faces", length=0.01,  enabled=True, color=(0.0, 0.4, 0.0))

                        
torsal_dir_show(baricenter, opt1, opt2)
#torsal_dir_show(baricenter, t1, t2, (0,0,0), "Analytical ")


ps.show()


