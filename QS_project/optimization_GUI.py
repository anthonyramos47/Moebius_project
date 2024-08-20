# Import the necessary libraries
import os
import sys
from pathlib import Path
import argparse

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
import time 
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
from energies.Lap_Fairness import Lap_Fairness

from optimization.Optimizer import Optimizer

# Directory where the data is stored ======================================

# Here you can add the directory where you want to save the results
dir_path = os.getcwd()
print(dir_path)

# Define Bsplines Surface directory
surface_dir = os.path.join(dir_path, "data", "Bsplines_Surfaces")
print("surface dir:", surface_dir)
experiment_dir = os.path.join(dir_path, "experiments")
reports_dir = os.path.join(dir_path, "data", "Reports")

# Optimization options ======================================

# Create the parser
parser = argparse.ArgumentParser(description="Optimization")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

parser.add_argument('deltaumin', type=float, help='delta value')
parser.add_argument('deltaumax', type=float, help='delta value')
parser.add_argument('deltavmin', type=float, help='delta value')
parser.add_argument('deltavmax', type=float, help='delta value')

parser.add_argument('type', type=int, help='Read 1 pickle or 2 json file')

bspline_surf_name = parser.parse_args().file_name
dir =  1




# Sample size
sample = (20, 20
          )
choice_data = 0 # 0: Json , 1: data_hyp.dat
mid_init = 0  # 0: central_sphere, 1: offset_surface
angle = 25 # Angle threshold with surface
tangle = 45 # Torsal angle threshold for torsal planes
weights = {
    "LC": [1,1], # Line congruence l.cu = 0, l.cv = 0
    "LC_Orth": [2,1], # Line congruence orthogonality with surface
    "Torsal": 1, # Torsal constraint
    "Torsal_Angle": 3, # Torsal angle constraint
    "Fairness": 0.1 # Fairness constraint
}


# Gobal variables ======================================
ang_normal = np.zeros((sample[0], sample[1]))
state = 0
state2 = 0  
counter = 0
init_opt_1 = False
init_opt_2 = False
name_saved = "Results"
name_report = ""
iter_per_opt = 20
per_cop_it = 25
step_1 = 0.5
step_2 = 0.5
time_1 = []
time_2 = []
report_data = {}
lc_rep = False
tor_rep = False
final_rep = False


# Optimization functions ====================================
def optimization():

    global state, state2, counter, opt, cp, l, init_opt_2, init_opt_1, angle, tangle, name_saved,iter_per_opt, step_1, step_2, bsp1, adj_v, time_1, time_2, name_report, report_data, lc_rep, tor_rep, final_rep, per_cop_it

    # Title
    psim.TextUnformatted("Sphere and Lince Congruence Optimization")


    if psim.Button("Stop"):
        state = 0
        state2 = 0


    psim.PushItemWidth(150)
    _, iter_per_opt = psim.InputInt("Num of Iterations per Run: ", iter_per_opt)

    _, angle  = psim.InputFloat("Angle: ", angle)
    _, tangle = psim.InputFloat("Torsal Angle: ", tangle)
    psim.Separator()

       
    # Inputs Opt 1
    if psim.CollapsingHeader("Optimization 1:"):

        _, weights["LC"][0] = psim.InputFloat("Line Congruence", weights["LC"][0])
        _, weights["LC_Orth"][0] = psim.InputFloat("LC Surf Orth", weights["LC_Orth"][0])
        _, step_1 = psim.InputFloat("Optimization step size", step_1)

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
            opt.add_constraint(LC, args=(bsp1, r_uv, u_pts, v_pts), w=weights["LC"][0], ce=1)

            # Line cong orthgonality with surface s(u,v)
            LC_orth = BS_LC_Orth()
            opt.add_constraint(LC_orth, args=(bsp1, r_uv, u_pts, v_pts, angle), w=weights["LC_Orth"][0], ce=1)

            # Define unit variables
            opt.unitize_variable("l", 3, 10)

            ps.info("Finished Initialization of Optimization 1")


        if psim.Button("Optimize 1"):
            state = 1
            counter = 1

    
    psim.Separator()

    if psim.CollapsingHeader("Optimization 2:"):

        _, weights["LC"][1] = psim.InputFloat("Line Congruence W", weights["LC"][1])
        _, weights["LC_Orth"][1] = psim.InputFloat("LC Surf Orth W", weights["LC_Orth"][1])
        _, weights["Torsal"] = psim.InputFloat("Torsal W", weights["Torsal"])
        _, weights["Torsal_Angle"] = psim.InputFloat("Torsal Angle W", weights["Torsal_Angle"])
        _, weights["Fairness"] = psim.InputFloat("Fairness W", weights["Fairness"])
        _, step_2 = psim.InputFloat("Optimization step size", step_2)
        
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
            opt.init_variable("theta" , 0.1)
            opt.init_variable("l"     , f_l.flatten())  
            opt.init_variable("rij"   , f_cp)
            opt.init_variable("mu"    , f_mu)

            r_uv[2] = f_cp

            # Line congruence l.cu, l.cv = 0
            LC = BS_LC()
            opt.add_constraint(LC, args=(bsp1, r_uv, u_pts, v_pts), w=weights["LC"][0], ce=1)

            # Line cong orthgonality with surface s(u,v)
            LC_orth = BS_LC_Orth()
            opt.add_constraint(LC_orth, args=(bsp1, r_uv, u_pts, v_pts, angle), w=weights["LC_Orth"][1], ce=1)

            # Torsal constraint 
            LC_torsal = BS_Torsal()
            opt.add_constraint(LC_torsal, args=(bsp1, u_pts, v_pts, n, sample), w=weights["Torsal"], ce=1)

            # Torsal angle constraint
            LC_torsal_ang = BS_Torsal_Angle()
            opt.add_constraint(LC_torsal_ang, args=(tangle, 0), w=weights["Torsal_Angle"], ce=1)

            # Fairness
            Fair_L = Lap_Fairness()
            opt.add_constraint(Fair_L, args=(adj_v, "l", 3), w=weights["Fairness"])

            opt.unitize_variable("l", 3, 10)
            opt.unitize_variable("nt1", 3, 10)
            opt.unitize_variable("nt2", 3, 10)

            opt.control_var("nt1", 0.05)
            opt.control_var("nt2", 0.05)
            #opt.control_var("l", 0.1)

            ps.info("Finished Initialization of Optimization 2")

        # if psim.Button("Optimize 2"):
        #     state2 = 1
        #     counter = 1


    if state:
            if init_opt_1 and not opt.stop:

                counter += 1
                # Optimize
                
                i_t  = time.time()
                # Get gradients
                opt.get_gradients() # Compute J and residuals
                opt.optimize_step() # Solve linear system and update variables
                f_t  = time.time()

                time_1.append(f_t - i_t)
                # l = opt.uncurry_X("l")
                # l = l.reshape(len(u_pts), len(v_pts), 3)
                # l = flip(l, n)
                # opt.init_variable("l", l.flatten())

                # Get Line congruence
                l, cp = opt.uncurry_X("l", "rij" )


                visualize_LC(surf, r_uv, l, n, u_pts, v_pts, V, F,  cp)
                #visualize_LC(surf, bsp1, r_uv, l, np.linspace(0, 1, 200), np.linspace(0, 1, 200), cp)
                opt.stop_criteria()
            elif not init_opt_1:
                ps.warning("First Optimization not initialized")
                state = 0
            else:
                ps.warning("Optimization 1 finished")
                state = 0
    
    
    if psim.Button("Optimize 2"):
        
        if init_opt_2:
            print("stop", opt.stop)
            it = 0
            while it < iter_per_opt and not opt.stop:
                # Optimize
                i_t = time.time()
                opt.get_gradients() # Compute J and residuals
                opt.optimize_step() # Solve linear system and update variables
                f_t = time.time()
                if it%per_cop_it == 0 and weights["Torsal_Angle"]!= 0:
                    opt.constraints[2].recompute(opt.X, opt.var_idx)

                time_2.append(f_t - i_t)
                it += 1
                opt.stop_criteria()
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


    psim.Separator()

    if psim.Button("Time"):  

        print("Time 1: ", np.array(time_1).mean())
        print("Time 2: ", np.array(time_2).mean())

        time_1 = []
        time_2 = []

    psim.Separator()

    psim.TextUnformatted("Save Results")

    _, name_saved = psim.InputText("Save File Name", name_saved)  

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
                    'init_l': init_l,
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

        # Save report 
        ps.warning("Results saved in: " + save_file_path)

    

    if psim.Button("Get Report LC"):
        # Show result per console
        opt.get_norm_energy_per_constraint()

        # Open file to write
        file_path = os.path.join(reports_dir, name_saved + "_LC.json")

        # Data to save
        
        report_data["Opt1 LC weights"] = weights["LC"][0]
        report_data["Opt1 LC Orth weights"] = weights["LC_Orth"][0]
        report_data["Opt1 it"] = opt.it
        report_data["Opt1 Energy"] = sum(e_i for e_i in opt.norm_energy_dic.values()) 
        report_data["Opt1 Time"] = np.array(time_1).mean()

        lc_rep = True

    
    psim.SameLine() 

    if psim.Button("Get Report Torsal"):
        # Show result per console
        opt.get_norm_energy_per_constraint()

        # Data to save
        report_data["Recomp steps"] = per_cop_it
        report_data["Opt2 LC weights"] = weights["LC"][1]
        report_data["Opt2 LC Orth weights"] = weights["LC_Orth"][1]
        report_data["Opt2 Torsal weights"] = weights["Torsal"]
        report_data["Opt2 Torsal Angle weights"] = weights["Torsal_Angle"]
        report_data["Opt2 it"] = opt.it
        report_data["Opt2 Energy"] = sum(e_i for e_i in opt.norm_energy_dic.values()) 
        report_data["Opt2 Time"] = np.array(time_2).mean()

        report_data["Angle"] = angle
        report_data["Torsal Angle"] = tangle
        report_data["Gird"] = sample
    
        tor_rep = True

        # Get RESULTS
        try:
            nt1, nt2, th = opt.uncurry_X("nt1", "nt2", "theta")

            nt1 = nt1.reshape(-1,3)
            nt2 = nt2.reshape(-1,3)

            cost1t2 = np.einsum("ij,ij->i", nt1, nt2)**2 

            cos_a_2 = np.cos(tangle*np.pi/180)**2

            idx = np.where(cost1t2 > np.cos(tangle*np.pi/180)**2)

            t_energy = np.mean((cost1t2[idx] - cos_a_2 )**2 )

            print("Torsal Angles: ", t_energy)
        except:
            pass
    
    psim.SameLine() 

    if psim.Button("Get Report Torsal Final"):
        # Show result per console
        opt.get_norm_energy_per_constraint()

        nt1, nt2 = opt.uncurry_X("nt1", "nt2")

        nt1 = nt1.reshape(-1,3)
        nt2 = nt2.reshape(-1,3)

        cost1t2 = np.einsum("ij,ij->i", nt1, nt2)**2 

        cos_a_2 = np.cos(tangle*np.pi/180)**2

        idx = np.where(cost1t2 > np.cos(tangle*np.pi/180)**2)

        if idx[0].shape[0] != 0:

            t_energy = np.mean((cost1t2[idx] - cos_a_2 )**2 )
        else:
            t_energy = 0

        print("Torsal Angles: ", t_energy)

        # Data to save
        report_data["Final LC weights"] = weights["LC"][1]
        report_data["Final LC Orth weights"] = weights["LC_Orth"][1]
        report_data["Final Torsal weights"] = weights["Torsal"]
        report_data["Final Torsal Angle weights"] = weights["Torsal_Angle"]
        report_data["Final it"] = opt.it
        report_data["Final Energy"] = sum(e_i for e_i in opt.norm_energy_dic.values()) + t_energy
        report_data["Final Time"] = np.array(time_2).mean()

        final_rep = True


    if psim.Button("Save Report"):
        
        # Open file to write
        file_path = os.path.join(reports_dir, name_saved + "_Report.json")

        # Save the variable to a json
        with open(file_path, 'w') as file:
            json.dump(report_data, file)
        

if parser.parse_args().type == 1:


    # Load picke information
    def load_data():
        """ Function to load the data from a pickle file
        """
        with open(os.path.join(experiment_dir, bspline_surf_name+'_init.pickle'), 'rb') as f:
            data = pickle.load(f)
        return data

    data = load_data()

    bsp1 = data["surf"]
    # o_v_pts = data["o_v_pts"]
    # o_u_pts = data["o_u_pts"]
elif parser.parse_args().type == 2:
    bsp1 = get_spline_data(choice_data, surface_dir, bspline_surf_name)
#bsp1 = get_spline_data(choice_data, surface_dir, bspline_surf_name)
        
#bsp1 = data["surf"]

# Get Grid Information
u_pts, v_pts = sample_grid(sample[0], sample[1], deltaum=parser.parse_args().deltaumin, deltauM=parser.parse_args().deltaumax, deltavm = parser.parse_args().deltavmin, deltavM = parser.parse_args().deltavmax)
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

#OV, OF = Bspline_to_mesh(bsp1, o_u_pts, o_v_pts)

# Compute the curvature
K, H, _ = curvatures_par(bsp1, u_pts, v_pts)

#OK, OH, _ = curvatures_par(bsp1, o_u_pts, o_v_pts)

H = H.flatten()
K = K.flatten()

# OH = OH.flatten()
# OK = OK.flatten()


valid = np.zeros_like(H)
# o_valid = np.zeros_like(OH)

idx = np.where(H < 0)[0]
# o_idx = np.where(OH < 0)[0]

valid[idx] = 1
#o_valid[o_idx] = 1


# GET TOPOLOGY INFO

# Initialize Mesh 
mesh = Mesh()
mesh.make_mesh(V, F)

# Get the vertex vertex adj list
adj_v = mesh.vertex_adjacency_list()

ps.init()
# Surface
surf = ps.register_surface_mesh("S_uv", V, F)


surf.add_scalar_quantity("Mean Curvature", H, enabled=True)
surf.add_scalar_quantity("Gaussian Curvature", K, enabled=False)
surf.add_scalar_quantity("Near Vanishing Curv", valid, enabled=True)


# INITIAL LC
surf.add_vector_quantity("init_l", init_l.reshape(-1, 3), vectortype='ambient', enabled=False, color=(0.0, 0.0, 0.1))
ps.set_user_callback(optimization)
ps.show()

