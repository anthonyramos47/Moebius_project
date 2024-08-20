# This file is a template for creating a new python file in the hananLab/hanan directory

# Import the necessary libraries
import os
import sys
import argparse
import pickle
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
from scipy.optimize import minimize
from scipy.interpolate import bisplev

# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *
from utils.bsplines_functions import *

# Optimization classes
from energies.BS_Repair import BS_Repair
from optimization.Optimizer import Optimizer


# Here you can add the directory where you want to save the results
dir_path = os.getcwd()
print(dir_path)

# Define Bsplines Surface directory
surface_dir = os.path.join(dir_path, "data", "Bsplines_Surfaces")

print("surface dir:", surface_dir)

experiment_dir = os.path.join(dir_path, "experiments")


# Create the parser
parser = argparse.ArgumentParser(description="Curvature Vis")
# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

bspline_surf_name = parser.parse_args().file_name

# Define the path to the B-spline surface
bspline_surf_path = os.path.join(surface_dir, bspline_surf_name + ".json")
print("bspline_surf_path:", bspline_surf_path)

# Sample in each direction
sample = 45

# Load the B-spline surface
control_points, knots_u, knots_v, order_u, order_v = read_bspline_json(bspline_surf_path)



# Create the B-splines basis
basis_u = sp.BSplineBasis(order_u, knots_u) 
basis_v = sp.BSplineBasis(order_v, knots_v) 



# Create the B-spline surface
bsp_opt = sp.Surface(basis_u, basis_v, control_points)


# Sample the grid points to evaluate the B-spline surface
u_vals, v_vals = sample_grid(sample, sample, deltaum=0.01, deltauM=0.01, deltavm = 0.01, deltavM = 0.05)
# Compute the curvature
_, H, _ = curvatures_par(bsp_opt, u_vals, v_vals)
H = H.flatten()

# Create the optimizer
opt = Optimizer()

# Add variables to the optimizer
opt.add_variable("cp" , len(control_points)*3) # Control points


# Initialize Optimizer
opt.initialize_optimizer("LM", 0.1, 1)

# Init variables 
opt.init_variable("cp" , control_points.flatten())  


# Line congruence l.cu, l.cv = 0
BS_rep = BS_Repair()
opt.add_constraint(BS_rep, args=(order_u, order_v, knots_u, knots_v, control_points, u_vals, v_vals), w=0.1, ce=1)

opt.control_var("cp", 1)

open_menu = False
iterations = 20

run_opt = False

iteration = 0

def optimization():
    global opt, sample, delta_u_m, delta_u_M, delta_v_m, delta_v_M, u_vals, v_vals, surf, bsp_opt, open_menu, iterations, run_opt, iteration

    delta_u_m = 0.05 
    delta_u_M = 0.05  
    delta_v_m = 0.05
    delta_v_M = 0.05


    _, iterations = psim.InputInt("Iterations", iterations)

    if psim.Button("Optimize") and not opt.stop:
        run_opt = True

    if iteration == iterations:
        run_opt = False
        iteration = 0

    if run_opt:
        iteration += 1
        opt.get_gradients()
        opt.optimize_step()
        
        opt.stop_criteria()

        optimized_control_points = opt.uncurry_X("cp")
        #optimized_control_points = result.x.reshape(control_points.shape)
        # Create the B-spline surface
        bsp_opt = sp.Surface(basis_u, basis_v, optimized_control_points.reshape(-1,3))


        V,F =Bspline_to_mesh(bsp_opt, u_vals, v_vals)
        surf = ps.register_surface_mesh("mesh", V, F)

         # Compute the curvature
        K, H, _ = curvatures_par(bsp_opt, u_vals, v_vals)

        H = H.flatten()
        K = K.flatten()
        valid = np.zeros_like(H)

        idx = np.where(H < 0)[0]

        valid[idx] = 1
        
        surf.add_scalar_quantity("Mean Curvature", H, enabled=True)
        surf.add_scalar_quantity("Gaussian Curvature", K, enabled=False)
        surf.add_scalar_quantity("Near Vanishing Curv", valid, enabled=True)

        ps.register_curve_network("control_points mesh", optimized_control_points.reshape(-1,3), np.array(edges_net), enabled=True, radius=0.001, color=(1.0, 0.0, 0.0))

        ps.register_point_cloud("control_points", optimized_control_points.reshape(-1,3), radius=0.003, color=(0.0, 0.0, 0.0))

        # # Variable to save
        # save_data = {
        #             'surf': bsp_opt,
        #             'o_u_pts': u_vals,
        #             'o_v_pts': v_vals,
        #             }
        
        # save_file_path = os.path.join(experiment_dir, bspline_surf_name  + "_init.pickle")

        # # Save the variable to a file
        # with open(save_file_path, 'wb') as file:
        #     pickle.dump(save_data, file)

        # ps.warning("Results saved in: " + save_file_path)

    if psim.Button("Save"):
        optimized_control_points = opt.uncurry_X("cp")
        bsp_opt = sp.Surface(basis_u, basis_v, optimized_control_points.reshape(-1,3))
        # Variable to save
        save_data = {
                    'surf': bsp_opt,
                    }
        
        save_file_path = os.path.join(experiment_dir, bspline_surf_name  + "_init.pickle")

        # Save the variable to a file
        with open(save_file_path, 'wb') as file:
            pickle.dump(save_data, file)

        ps.warning("Results saved in: " + save_file_path)

    
    if psim.Button("Export"):

        open_menu = True

        

    _, delta_u_m = psim.SliderFloat("deltaumin", delta_u_m, 0.01, 1)
    _, delta_u_M = psim.SliderFloat("deltaumax", delta_u_M, 0.01, 1)
    _, delta_v_m = psim.SliderFloat("deltavmin", delta_v_m, 0.01, 1)
    _, delta_v_M = psim.SliderFloat("deltavmax", delta_v_M, 0.01, 1)

        
    if open_menu:
        if psim.Button("Draw Surface"):
            optimized_control_points = opt.uncurry_X("cp")
            bsp_opt = sp.Surface(basis_u, basis_v, optimized_control_points.reshape(-1,3))
            # Sample the grid points to evaluate the B-spline surface
            u_vals, v_vals = sample_grid(sample, sample, deltaum=delta_u_m, deltauM=delta_u_M, deltavm = delta_v_m, deltavM = delta_v_M)
        

            V,F = Bspline_to_mesh(bsp_opt, u_vals, v_vals)
            surf = ps.register_surface_mesh("mesh", V, F)

            # Compute the curvature
            K, H, _ = curvatures_par(bsp_opt, u_vals, v_vals)

            H = H.flatten()
            K = K.flatten()
            valid = np.zeros_like(H)

            idx = np.where(H < 0)[0]

            valid[idx] = 1
            
            surf.add_scalar_quantity("Mean Curvature", H, enabled=True)
            surf.add_scalar_quantity("Gaussian Curvature", K, enabled=False)
            surf.add_scalar_quantity("Near Vanishing Curv", valid, enabled=True)


        if psim.Button("Save"):
            optimized_control_points = opt.uncurry_X("cp")
            bsp_opt = sp.Surface(basis_u, basis_v, optimized_control_points.reshape(-1,3))
            # Variable to save
            save_data = {
                        'surf': bsp_opt,
                        }
            
            save_file_path = os.path.join(experiment_dir, bspline_surf_name  + "_init.pickle")

            # Save the variable to a file
            with open(save_file_path, 'wb') as file:
                pickle.dump(save_data, file)

            ps.warning("Results saved in: " + save_file_path)




# Evaluate the B-spline surface
#eval_surf = bsp_opt(u_vals, v_vals)


# def curvature_sel():
#     global bsp_opt, sample, delta_u_m, delta_u_M, delta_v_m, delta_v_M, u_vals, v_vals, surf


# Get only control points
flat_ctrl_pts = control_points.reshape(-1,3)
print("flat_ctrl_pts:", flat_ctrl_pts.shape)

# Create netwrok for the control points
edges_net= []
for i in range(39):
    for j in range(39):
        edges_net.append([i*40 + j, i*40 + j + 1])
        edges_net.append([i*40 + j, (i+1)*40 + j])



ps.init()

V,F =Bspline_to_mesh(bsp_opt, u_vals, v_vals)
surf = ps.register_surface_mesh("mesh", V, F)

ps.register_curve_network("control_points mesh", control_points.reshape(-1,3), np.array(edges_net), enabled=True, radius=0.001, color=(1.0, 0.0, 0.0))

ps.register_point_cloud("control_points", control_points.reshape(-1,3), radius=0.003, color=(0.0, 0.0, 0.0))

# Compute the curvature
K, H, _ = curvatures_par(bsp_opt, u_vals, v_vals)

H = H.flatten()
K = K.flatten()
valid = np.zeros_like(H)

idx = np.where(H < 0)[0]

valid[idx] = 1

surf.add_scalar_quantity("Mean Curvature", H, enabled=True)
surf.add_scalar_quantity("Gaussian Curvature", K, enabled=False)
surf.add_scalar_quantity("Near Vanishing Curv", valid, enabled=True)
ps.set_user_callback(optimization)
ps.show()
