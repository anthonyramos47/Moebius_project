import pickle


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

# Define the optimization GUI
def optimization_gui(opt: Optimizer, weights)->None:
    opt.set_constraints_weights_dic(weights)

    opt.get_gradients() # Compute J and residuals
    opt.optimize_step() # Solve linear system and update variables
    opt.stop_criteria() # Check stopping criteria
            

def optimization_init(opt: Optimizer,  
                        bsp,
                        sample,
                        u_range,
                        v_range,
                        steps,
                        angle
                      )->None:

    # Get Grid points
    u_pts = np.linspace(u_range[0], u_range[1], sample[0])
    v_pts = np.linspace(v_range[0], v_range[1], sample[1])


    # Initialize the central spheres radius and normals
    r_H, n = init_sphere_congruence(0, bsp, u_pts, v_pts, sample)

    # Fit r_H to a B-spline surface r(u,v)
    r_uv = r_uv_fitting(u_pts, v_pts, r_H)

    # Compute the line congruence
    l = line_congruence_uv(bsp, r_uv, u_pts, v_pts)
    l = flip(l, n)

    # Get the number of control points
    cp = r_uv[2].copy()

    # Add variables to the optimizer
    opt.add_variable("rij", len(cp)) # Control points
    opt.add_variable("l"  , 3*sample[0]*sample[1])
    # Dummy variables                              
    opt.add_variable("mu" , sample[0]*sample[1])

    # Initialize Optimizer ("Method", step, verbosity)
    opt.initialize_optimizer("LM", steps, 1)

    # Initialize variables
    opt.init_variable("rij" ,         cp )
    opt.init_variable("mu"  ,         50  )
    opt.init_variable("l"   , l.flatten())
    # Constraints ==========================================

    # Line congruence l.cu, l.cv = 0
    LC = BS_LC()
    opt.add_constraint(LC, args=(bsp, r_uv, u_pts, v_pts), w=1, ce=1)
    BS_LC.name = "LC"

    # Line cong orthgonality with surface s(u,v)
    LC_orth = BS_LC_Orth()
    opt.add_constraint(LC_orth, args=(bsp, r_uv, u_pts, v_pts, angle), w=1, ce=1)
    BS_LC_Orth.name = "LC_Orth"

    # Define unit variables
    opt.unitize_variable("l", 3, 10)

    # Gvars
    gvars = {"r_uv": r_uv, "n": n, "init_l": l}

    return gvars


def optimization_torsal(opt: Optimizer,  
                        sample,
                        bsp,
                        r_uv,
                        n,
                        u_pts,
                        v_pts,
                        steps,
                        angle,
                        tangle,
                    )->None:
    
    V,F = Bspline_to_mesh(bsp, u_pts, v_pts)

    # Initialize Mesh 
    mesh = Mesh()
    mesh.make_mesh(V, F)

    # Get the vertex vertex adj list
    adj_v = mesh.vertex_adjacency_list()

    # Get Grid points
    n_squares = (len(u_pts)-1)*(len(v_pts)-1)
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
    opt.initialize_optimizer("LM", steps, 1)

    # Init variables 
    opt.init_variable("theta" , 0.1)
    opt.init_variable("l"     , f_l.flatten())  
    opt.init_variable("rij"   , f_cp)
    opt.init_variable("mu"    , f_mu)

    r_uv[2] = f_cp

    # Line congruence l.cu, l.cv = 0
    LC = BS_LC()
    opt.add_constraint(LC, args=(bsp, r_uv, u_pts, v_pts), w=1, ce=1)
    LC.name = "LC"

    # Line cong orthgonality with surface s(u,v)
    LC_orth = BS_LC_Orth()
    opt.add_constraint(LC_orth, args=(bsp, r_uv, u_pts, v_pts, angle), w=1, ce=1)
    LC_orth.name = "LC_Orth"

    # Torsal constraint 
    LC_torsal = BS_Torsal()
    opt.add_constraint(LC_torsal, args=(bsp, u_pts, v_pts, n, sample), w=1, ce=1)
    LC_torsal.name = "Torsal"

    # Torsal angle constraint
    LC_torsal_ang = BS_Torsal_Angle()
    opt.add_constraint(LC_torsal_ang, args=(tangle, 0), w=1, ce=1)
    LC_torsal_ang.name = "Torsal_Angle"

    # Fairness
    Fair_L = Lap_Fairness()
    opt.add_constraint(Fair_L, args=(adj_v, "l", 3), w=1)
    Lap_Fairness.name = "Fairness"

    opt.unitize_variable("l", 3, 10)
    opt.unitize_variable("nt1", 3, 10)
    opt.unitize_variable("nt2", 3, 10)

    opt.control_var("nt1", 0.05)
    opt.control_var("nt2", 0.05)
    #opt.control_var("l", 0.1)

    return opt