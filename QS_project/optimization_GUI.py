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
import polyscope.imgui as psim
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import json
from scipy.interpolate import bisplev
from geomdl.fitting import approximate_surface
from time import sleep

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

# Directory where the data is stored ======================================

# Here you can add the directory where you want to save the results
dir_path = os.getcwd()
print(dir_path)

# Define Bsplines Surface directory
surface_dir = os.path.join(dir_path, "data", "Bsplines_Surfaces")
print("surface dir:", surface_dir)



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
angle = 25 # Angle threshold with surface
tangle = 45 # Torsal angle threshold for torsal planes
choice_data = 0 # 0: Json , 1: data_hyp2.dat
mid_init = 0  # 0: central_sphere, 1: offset_surface
delta = 0.1
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
init_opt_2 = 0


def flip(l, n ):
    l = np.sign(np.sum(l*n, axis=2))[:,:,None]*l
    return l

# Optimization functions ====================================
def optimization():

    global state, state2, counter, opt, init_opt_2

    if psim.Button("Optimize 1"):
        state = 1
        counter = 1
    
    if psim.Button("Optimize 2"):
        state2 = 1
        counter = 1

    if counter%20 == 0:
        state = 0
        state2 = 0

    if state:
        counter += 1
        # Optimize
        
        # Get gradients
        opt.get_gradients() # Compute J and residuals
        opt.optimize() # Solve linear system and update variables

        # l = opt.uncurry_X("l")
        # l = l.reshape(len(u_pts), len(v_pts), 3)
        # l = flip(l, n)
        # opt.init_variable("l", l.flatten())

        # Get Line congruence
        l, cp = opt.uncurry_X("l", "rij" )

        visualize_LC(surf, r_uv, l, n, u_pts, v_pts, V, F,  cp)
    
    if psim.Button("Init Second opt:"):

        init_opt_2 = 1

        # Copy previous X from optimization
        f_l, f_cp, f_mu = opt.uncurry_X("l", "rij", "mu")

        f_l = f_l.reshape(-1,3)
        n_flat = n.reshape(-1,3)

        # Fix direction with normal
        f_l = np.sign(np.sum(f_l*n_flat, axis=1))[:,None]*f_l

        # Create the optimizer
        opt = Optimizer()

        # Add variables to the optimizer
        opt.add_variable("rij" , len(f_cp)) # Control points
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
        opt.initialize_optimizer("LM", 0.5, 1)

        # Init variables 
        opt.init_variable("theta" , 10)
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
    
    if state2 and init_opt_2:

        counter += 1

        # Optimize
        opt.get_gradients() # Compute J and residuals
        opt.optimize() # Solve linear system and update variables

        # Flip line congruence if needed
        l = opt.uncurry_X("l")
        l = l.reshape(len(u_pts), len(v_pts), 3)
        l = flip(l, n)
        opt.init_variable("l", l.flatten())


        visualization_LC_Torsal(surf, opt, r_uv, u_pts, v_pts, n, V, F)
        
    if psim.Button("Flip 1:"):
    
        # Get Line congruence
        l = opt.uncurry_X("l")


        # Reshape Line congruence
        l = l.reshape(len(u_pts), len(v_pts), 3)
        l /= np.linalg.norm(l, axis=2)[:,:,None]

        # FIx sign with normal
        l = np.sign(np.sum(l*n, axis=2))[:,:,None]*l

        opt.init_variable("l", l.flatten())

        # Angle with normal
        ang_normal = np.arccos( np.sum( l*n, axis=2) )*180/np.pi


        # OPTIMIZED LC
        surf.add_vector_quantity("l", l.reshape(-1, 3), defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.1, 0.0, 0.0))


        # ANGLES WITH NORMAL SCALAR FIELD
        surf.add_scalar_quantity("Angles", ang_normal.flatten(), defined_on="vertices", enabled=True)

    if psim.Button("Report"):
        opt.get_energy_per_constraint()

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

# Create the optimizer
opt = Optimizer()

# Add variables to the optimizer
opt.add_variable("rij", len(cp)) # Control points
opt.add_variable("l"  , 3*len(u_pts)*len(v_pts))
# Dummy variables                              
opt.add_variable("mu" , len(u_pts)*len(v_pts)  )

# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", 0.5, 1)

# Initialize variables
opt.init_variable("rij" ,         cp )
opt.init_variable("mu"  ,         0.1 )
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

# End of constraints ===================================

V, F = Bspline_to_mesh(bsp1, u_pts, v_pts, sample)


ps.init()
# Surface
surf = ps.register_surface_mesh("S_uv", V, F)
# INITIAL LC
surf.add_vector_quantity("init_l", init_l.reshape(-1, 3), vectortype='ambient', enabled=False, color=(0.0, 0.0, 0.1))
ps.set_user_callback(optimization)
ps.show()


