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
#bspline_surf_name, dir = "Sph_inv1", 1
bspline_surf_name, dir = "Sph_inv_2", 1

# Rhino Bad test 
#bspline_surf_name, dir = "Surfjson", -1

# Florian Non change mean Curvature
#bspline_surf_name, dir = "Florian", 1

# Dat name
#bspline_surf_name, dir = "inv_1", 1
#bspline_surf_name, dir = "data_hyp", 1 # delta = 0.3
#bspline_surf_name, dir = "Sample_C", 1
#bspline_surf_name, dir = "Neg_Surf", 1
#bspline_surf_name, dir = "Tunel", -1

# Sample size
sample = (15, 15)
delta = 0.2
choice_data = 0 # 0: Json , 1: data_hyp.dat
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
counter = 0
init_opt_1 = False
init_opt_2 = False
name_saved = "Results"
iter_per_opt = 20
step_1 = 0.5
step_2 = 0.5


# Optimization functions ====================================
def optimization():

    global state, state2, counter, opt, cp, l, init_opt_2, init_opt_1, angle, tangle, name_saved,iter_per_opt, step_1, step_2

    # Title
    psim.TextUnformatted("Sphere and Lince Congruence Optimization")


    if psim.Button("Stop"):
        state = 0
        state2 = 0


    psim.PushItemWidth(150)
    changed, iter_per_opt = psim.InputInt("Num of Iterations per Run: ", iter_per_opt)
    psim.SameLine()
    changed, angle  = psim.InputFloat("Angle: ", angle)
    changed, tangle = psim.InputFloat("Torsal Angle: ", tangle)
    psim.Separator()

       
    # Inputs Opt 1
    if psim.CollapsingHeader("Optimization 1:"):

        changed, weights["LC"][0] = psim.InputFloat("Line Congruence", weights["LC"][0])
        changed, weights["LC_Orth"][0] = psim.InputFloat("LC Surf Orth", weights["LC_Orth"][0])
        changed, step_1 = psim.InputFloat("Optimization step size", step_1)

        # State handler
        if counter%iter_per_opt == 0:
            state = 0
            state2 = 0


        if psim.Button("Init First opt"):

            # Set init to True
            init_opt_1 = True

            # Create the optimizer
            opt = Optimizer()

            # Add variables to the optimizer
            opt.add_variable("rij", len(cp)) # Control points
            opt.add_variable("l"  , 3*sample[0]*sample[1])
            # Dummy variables                              
            opt.add_variable("mu" , sample[0]*sample[1])

            # Initialize Optimizer ("Method", step, verbosity)
            opt.initialize_optimizer("LM", step_1, 1)

            # Initialize variables
            opt.init_variable("rij" ,         cp )
            opt.init_variable("mu"  ,         50  )
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

            ps.info("Finished Initialization of Optimization 1")


        if psim.Button("Optimize 1"):
            state = 1
            counter = 1

    
    psim.Separator()

    if psim.CollapsingHeader("Optimization 2:"):

        changed, weights["LC"][1] = psim.InputFloat("Line Congruence W", weights["LC"][1])
        changed, weights["LC_Orth"][1] = psim.InputFloat("LC Surf Orth W", weights["LC_Orth"][1])
        changed, weights["Torsal"] = psim.InputFloat("Torsal W", weights["Torsal"])
        changed, weights["Torsal_Angle"] = psim.InputFloat("Torsal Angle W", weights["Torsal_Angle"])
        changed, step_2 = psim.InputFloat("Optimization step size", step_2)
        
        if psim.Button("Init Second opt"):

            init_opt_2 = True

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
            opt.initialize_optimizer("LM", step_2, 1)

            # Init variables 
            opt.init_variable("theta" , 50)
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

            ps.info("Finished Initialization of Optimization 2")

        # if psim.Button("Optimize 2"):
        #     state2 = 1
        #     counter = 1


    if state:
            if init_opt_1:

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
            else:
                ps.warning("First Optimization not initialized")
                state = 0
    
    
    if psim.Button("Optimize 2"):
        if init_opt_2:
            for _ in range(iter_per_opt):
                # Optimize
                opt.get_gradients() # Compute J and residuals
                opt.optimize() # Solve linear system and update variables

            # Flip line congruence if needed
            l = opt.uncurry_X("l")
            l = l.reshape(len(u_pts), len(v_pts), 3)
            l = flip(l, n)
            opt.init_variable("l", l.flatten())

            visualization_LC_Torsal(surf, opt, r_uv, u_pts, v_pts, n, V, F)
        # else:
        #     ps.warning("Second Optimization not initialized")
        #     state2 = 0
                            

    psim.Separator()
        
    if psim.Button("Flip Line Congruence"):
    
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
    
    psim.Separator()

    psim.TextUnformatted("Save Results")

    changed, name_saved = psim.InputText("Save File Name", name_saved)    

    if psim.Button("Save"):

         # Get RESULTS
        l, cp, tu1, tu2, tv1, tv2, nt1, nt2 = opt.uncurry_X("l", "rij", "u1", "u2", "v1", "v2", "nt1", "nt2")

        # Reshape Torsal normals
        nt1 = nt1.reshape(-1,3)
        nt2 = nt2.reshape(-1,3)

        # Update control points of r(u,v) spline surface
        r_uv[2] = cp  
        r_uv_surf = bisplev(u_pts, v_pts, r_uv)

        # Reshape Line congruence
        l = l.reshape(len(u_pts), len(v_pts), 3)
        l /= np.linalg.norm(l, axis=2)[:,:,None]

        # Angle with normal
        ang_normal = np.arccos( np.sum( l*n, axis=2) )*180/np.pi

        # Get vertices
        v0, v1, v2, v3 = V[F[:,0]], V[F[:,1]], V[F[:,2]], V[F[:,3]]

        # l (-1,3)
        l_f = l.reshape(-1,3)

        l0, l1, l2, l3 = l_f[F[:,0]], l_f[F[:,1]], l_f[F[:,2]], l_f[F[:,3]]

        lu = l2 - l0
        lv = l1 - l3

        # Compute tangents
        du = v2 - v0
        dv = v1 - v3

        # Compute barycenters
        barycenters = (v0 + v1 + v2 + v3)/4
        lc = (l0 + l1 + l2 + l3)/4

        # Get torsal directions
        t1 = unit(tu1[:,None]*du + tv1[:,None]*dv)
        t2 = unit(tu2[:,None]*du + tv2[:,None]*dv)

        lt1 = unit(tu1[:,None]*lu + tv1[:,None]*lv)
        lt2 = unit(tu2[:,None]*lu + tv2[:,None]*lv)


        V_R = V + r_uv_surf.flatten()[:,None]*n.reshape(-1,3)
        
        # Variable to save
        save_data = {
                    'surf': bsp1,
                    'r_uv': r_uv,
                    'u_pts': u_pts,
                    'v_pts': v_pts,
                    'V': V,
                    'V_R': V_R,
                    'F': F,
                    'l': l,
                    'n': n,
                    't1': t1,
                    't2': t2,
                    'nt1': nt1,
                    'nt2': nt2,
                    'lt1': lt1,
                    'lt2': lt2,
                    'bar': barycenters,
                    'lc': lc,
                    'r_uv_surf': r_uv_surf
                      }
        
        save_file_path = os.path.join(experiment_dir, name_saved + ".pickle")

        # Save the variable to a file
        with open(save_file_path, 'wb') as file:
            pickle.dump(save_data, file)

        ps.warning("Results saved in: " + save_file_path)


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

