# Import the necessary libraries
import os
import sys
from pathlib import Path
import argparse
from pickle import load
import subprocess

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
import igl
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import json
import pickle



# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *

# Local files
from utils.bsplines_functions import *
from utils.visualization import *

# Optimization classes
#from energies.Support import Supp
from energies.Sphere import Sphere
from energies.QM_Fairness import QM_Fairness
from energies.Proximity_Gen import Proximity
from energies.Proximity_C import Proximity_C
from energies.Support import Supp
from energies.Support_Fix import Supp_F
from energies.Sphere_Fix_V import Sphere_Fix
from energies.Sph_unit import Sph_Unit
from energies.Tor_Planarity import Tor_Planarity
from energies.Reg_E import Reg_E

from optimization.Optimizer import Optimizer




# Create the parser
parser = argparse.ArgumentParser(description="Post Optimization")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
name = parser.parse_args().file_name
pickle_name = name+'.pickle'
save = 1

# Working directory
working_path = os.getcwd()

# Experiments folder
exp_dir = os.path.join(working_path, 'experiments')

# Remeshing data folder
remeshing_dir = os.path.join(working_path, 'data', 'Remeshing', name)


# Load picke information
def load_data():
    """ Function to load the data from a pickle file
    """
    with open(os.path.join(exp_dir, pickle_name), 'rb') as f:
        data = load(f)
    return data

data = load_data()

if "projected" not in data.keys():
    print("Projecting the mesh")
    # Replace 'script_to_run.py' with the path to your Python script
    script_path = 'foot_point_bspline.py '+name+' 0'
    # Run the script using subprocess
    subprocess.run(['python', script_path])



# Get u and v points 
u_pts = data['u_pts']
v_pts = data['v_pts']

# Get the B-spline and its derivatives
BSurf = data['surf']
rsurf = data['r_uv']

# Get information from the pickle file
f_f_adj = data['f_f_adj']
e_f_f = data['e_f_f']
e_v_v = data['e_v_v']
adj_v = data['adj_v']
inn_f = data['inn_f']
inn_v = data['inn_v']
bd_v = data['bd_v']
cc = data['cc']
vc = data['vc']
nd = data['nd']
f_pts = data['f_pts']
rads = data['rads']
dual_top = data['dual_top']
ffF = data['ffF']
ffV = data['ffV']
ref_V = data['ref_V']
ref_F = data['ref_F']
ref_C = data['ref_C']
VR = data['VR']
l_f = data['l_dir']


A, B, C = get_Implicit_sphere(cc, rads)

dual_edges = np.array(list(extract_edges(dual_top)))

mesh_edges = e_v_v

tuples_edges = np.vstack((mesh_edges[0], mesh_edges[1])).T







# clean up edges have only innver vertices
# # Sample size of the B-spline
# sample = (len(u_pts), len(v_pts))   

# # Evaluate the B-spline at the u and v points
# V, F = Bspline_to_mesh(BSurf, u_pts, v_pts)

# # Get the vertices and faces of the mesh
# ref_u = np.linspace(0,   1, 300)
# ref_v = np.linspace(0,   1, 300)

# # Get the vertices and faces of the mesh
# ref_V, ref_F = Bspline_to_mesh(BSurf, ref_u, ref_v)

# ref_F = np.array(triangulate_quads(ref_F))


# # Compute footpoints (u,v) coordinates of remeshed mesh onto the B-spline
# foot_pts = foot_points(ffV, V, u_pts, v_pts, BSurf)
# foot_pts = foot_pts.reshape(-1, 2)

# # Evaluate the B-spline functions at the foot points bsurf(u, v), r(u, v) and n(u, v)
# f_pts = np.zeros((len(foot_pts), 3))
# r_pts = np.zeros((len(foot_pts)))
# n_dir = np.zeros((len(foot_pts), 3))
# for i in range(len(foot_pts)):
#     n_dir[i] = BSurf.normal(foot_pts[i, 0], foot_pts[i, 1])
#     f_pts[i] =   BSurf(foot_pts[i, 0], foot_pts[i, 1])
#     r_pts[i] = bisplev(foot_pts[i, 0], foot_pts[i, 1], rsurf)

# # Compute the vertices of the mid mesh
# VR = f_pts + r_pts[:,None]*n_dir
# VR = VR.reshape(-1, 3)

# # Get vertices of C(u, v) 
# # Evaluate r(u,v) in ref_u ref_v
# ref_r_uv = bisplev(ref_u, ref_v, rsurf)
# # Compute the normals of the reference mesh
# ref_n = BSurf.normal(ref_u, ref_v)
# ref_rn = ref_r_uv[:, :, None]*ref_n

# ref_v = BSurf(ref_u, ref_v)
# ref_C = ref_v + ref_rn
# ref_C = ref_C.reshape(-1, 3)



# # Create mesh for Mid mesh (sphere centers)
# mesh = Mesh()
# mesh.make_mesh(VR, ffF)

# # Get the face-face adjacency list 
# f_f_adj = mesh.face_face_adjacency_list()
# dual_top = mesh.dual_top()

# # Faces of each edge
# e_f_f = mesh.edge_faces()

# # Vertices of each edge
# e_v_v = mesh.edge_vertices()

# # Get inner faces indices
# inn_f = mesh.inner_faces()

# # Get inner vertices
# inn_v = mesh.inner_vertices()

# bd_v = mesh.boundary_vertices()

# #print(inn_v)

# # Get adjacent vertices
# adj_v = mesh.vertex_adjacency_list()

# # Get adjacent faces
# ffF = mesh.faces()

# # Compute the nodes of the mid mesh
# c0, c1, c2, c3 = VR[ffF[:, 0]], VR[ffF[:, 1]], VR[ffF[:, 2]], VR[ffF[:, 3]]

# # Compute the baricenter of the mid mesh
# cc = (c0 + c1 + c2 + c3)/4

# # Compute the barycenter at the remeshed mesh
# vc = np.sum(f_pts[ffF], axis=1)/4

# # Compute radius of spheres
# #rads = np.linalg.norm(vc - cc, axis=1)
# rads =  np.sum(r_pts[ffF ], axis=1)/4

# # # Normals of dual faces
# nd = np.zeros((len(dual_top), 3))

# for i, f_i in enumerate(dual_top):
#     if len(f_i) >= 4:
#         nd[i] = np.cross(cc[f_i[2]] - cc[f_i[0]], cc[f_i[1]] - cc[f_i[3]])
#         nd[i] /= np.linalg.norm(nd[i])
#     elif len(f_i) == 3:
#         nd[i] = np.cross(cc[f_i[1]] - cc[f_i[0]], cc[f_i[2]] - cc[f_i[1]])
#         nd[i] /= np.linalg.norm(nd[i])



# # Find zero nd
# #print(nd[np.where(np.linalg.norm(nd, axis=1) == 0)[0]])


# # idx_dual_top = np.unique(fla_dual_top)

# # # Network to visualzie nd
# # nodes = np.hstack((vc[idx_dual_top], vc[idx_dual_top] + nd))
# # nodes = nodes.reshape(-1, 3)

# # edges = []
# # for i in range(len(vc)):
# #     edges.append([i, i+len(vc)])
        

# # opt = Optimizer()

# # # Add variables to the optimizer
# # opt.add_variable("c" , len(vc)*3   ) # Centers of spheres
# # opt.add_variable("nd", len(nd)*3   ) # Normals of dual faces
# # opt.add_variable("v" , len(f_pts)*3) # Vertices of mid mesh
# # opt.add_variable("r" , len(rads)   ) # Radii of spheres

# # # Initialize Optimizer ("Method", step, verbosity)
# # opt.initialize_optimizer("LM", 0.6, 1)

# # # Initialize variables
# # opt.init_variable("c"  , cc.flatten()) 
# # opt.init_variable("nd" , nd.flatten())
# # opt.init_variable("v"  , f_pts.flatten())
# # opt.init_variable("r"  , rads)



# # # Constraints ==========================================
# # # Line congruence l.cu, l.cv = 0
# # Supp_E = Supp()
# # opt.add_constraint(Supp_E, args=(dual_top, 2), w=1)

# # # Sphericity
# # Sph_E = Sphericity()
# # opt.add_constraint(Sph_E, args=(ffF, 2), w=1)

# # # Fairness
# # Fair_M = QM_Fairness()
# # opt.add_constraint(Fair_M, args=(inn_v, adj_v, "v", 3), w=1)

# # # Proximity
# # Prox_M = Proximity()
# # opt.add_constraint(Prox_M, args=(ref_V, ref_F, 0.001), w=0.5)

# # # Fair_C = QM_Fairness()
# # # opt.add_constraint(Fair_C, args=(in_v_c, ad_v_c, "c", 3), w=0.002)

# # # Define unit variables
# # opt.unitize_variable("nd", 3, 10)

# # for _ in range(100):
# #     # Get gradients
# #     opt.get_gradients() # Compute J and residuals
# #     opt.optimize() # Solve linear system and update variables


# # opt.get_energy_per_constraint()


# # if save:
# #     # save mesh
# #     write_obj(os.path.join(remeshing_dir, name+'_mesh_opt.obj'), vk, ffF)

# #     # Save spheres ======================================================================
# #     # Open file
# #     exp_file = open(os.path.join(remeshing_dir, name+'_sphs_opt.dat'), 'w')

# #     # Write inn_f in the first line
# #     for i in inn_f:
# #         exp_file.write(f"{i} ")
# #     exp_file.write("\n")

# #     for i in range(len(ffF)):

# #         exp_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} {vc[i][0]} {vc[i][1]} {vc[i][2]} ")

# #         for j in f_f_adj[i]:
# #             exp_file.write(f"{j} ")

# #         exp_file.write("\n")

# #     exp_file.close()

w_proximity = 0.5
w_proximity_c = 1e-7
w_fairness = 1
w_sphericity = 1
w_supp = 1
iter_per_opt = 1
w_lap = 0.1
state = 0
state2 = 0
counter = 0
counter2 = 0
init_opt = False
w_tor = 0.1
w_reg = 0.1
init_opt_2 = False
idx_sph = 0
step = 0.5
name_saved = "Post_optimization"
def optimization():

    global w_proximity, w_fairness, w_sphericity, w_supp, iter_per_opt, init_opt, state, counter, vc, nd, f_pts, rads, dual_top, ffF, ref_V, ref_F, ref_u, ref_v, cc, vc, opt, inn_v, bd_v, adj_v, name_saved,e_f_f, e_v_v, w_lap, state2, init_opt_2, A, B, C, w_proximity_c, idx_sph, w_tor, w_reg, step, l_f, counter2

    # Title
    psim.TextUnformatted("Post Optimization GUI")

    if psim.Button("Stop"):
        state = 0
        state2 = 0
        

    psim.PushItemWidth(150)
    changed, iter_per_opt = psim.InputInt("Num of Iterations per Run: ", iter_per_opt)
    changed, step = psim.InputFloat("Setp per Optimization: ", step)
    
    psim.Separator()

        # State handler
    if counter%iter_per_opt == 0 or counter2%iter_per_opt == 0:
        state = 0
        state2 = 0
       
    # Inputs Opt 1
    if psim.CollapsingHeader("Optimization 1:"):
        changed, w_sphericity = psim.InputFloat("Sphericity W", w_sphericity)
        changed, w_supp       = psim.InputFloat("Support W", w_supp)
        changed, w_proximity_c  = psim.InputFloat("Proximity C", w_proximity_c)
        #changed, w_lap        = psim.InputFloat("Laplacian W", w_lap)

  

        if psim.Button("Init First opt"):

            # Set init to True
            init_opt = True

            opt = Optimizer()

            # Add variables to the optimizer
            opt.add_variable("A" , len(vc)  ) # Centers of spheres
            opt.add_variable("B" , len(vc)*3  ) # Centers of spheres
            opt.add_variable("C" , len(vc)  ) # Centers of spheres
            
            
            

            # Initialize Optimizer ("Method", step, verbosity)
            opt.initialize_optimizer("LM", 0.65, 1)

            # Initialize variables
            opt.init_variable("A"  , A) 
            opt.init_variable("B"  , B.flatten())
            opt.init_variable("C"  , C)
            

    
            # Sphericity
            Sph_E = Sphere_Fix()
            opt.add_constraint(Sph_E, args=(ffF, f_pts), w=w_sphericity)
            Sph_U = Sph_Unit()
            opt.add_constraint(Sph_U, args=(), w=10)

            
            Supp_E = Supp_F()
            opt.add_constraint(Supp_E, args=(dual_top, l_f), w=w_supp)


            # Proximity
            Prox_C = Proximity_C()
            opt.add_constraint(Prox_C, args=(ref_C, ref_F, 0.001), w=w_proximity_c)

            #opt.unitize_variable("nd", 3, 10)
            #opt.control_var("v", 0.01)
            # opt.control_var("c", 0.001)

            ps.info("Finished Initialization of Optimization 1")

        if psim.Button("Optimize 1"):
            if init_opt:
                state = 1
            else:
                ps.warning("First Optimization not initialized")
                state = 0


    if psim.CollapsingHeader("Optimization 2:"):

        changed, w_proximity  = psim.InputFloat("Proximity M", w_proximity)
        changed, w_proximity_c  = psim.InputFloat("Proximity C", w_proximity_c)
        changed, w_fairness   = psim.InputFloat("Fairness W", w_fairness)
        changed, w_sphericity = psim.InputFloat("Sphericity W", w_sphericity)
        changed, w_supp       = psim.InputFloat("Support W", w_supp)
        changed, w_tor        = psim.InputFloat("Tor Planarity W", w_tor)
        changed, w_reg        = psim.InputFloat("Reg W", w_reg)

        if psim.Button("Init Second opt"):

            init_opt_2 = True

            A, B, C =opt.uncurry_X("A", "B", "C")

            opt = Optimizer()

            # Add variables to the optimizer
            opt.add_variable("A" , len(vc)  ) # Centers of spheres
            opt.add_variable("B" , len(vc)*3  ) # Centers of spheres
            opt.add_variable("C" , len(vc)  ) # Centers of spheres
            opt.add_variable("nd", len(l_f)*3   ) # Normals of dual faces
            opt.add_variable("v" , len(f_pts)*3) # Vertices of mid mesh
            opt.add_variable("n_l" , len(mesh_edges[0])*3   ) # Radii of spheres
            #opt.add_variable("r" , len(rads)   ) # Radii of spheres
            #opt.add_variable("mu", len(dual_edges))

            # Initialize Optimizer ("Method", step, verbosity)
            opt.initialize_optimizer("LM", step, 1)

            # Initialize variables
            opt.init_variable("A"  , A) 
            opt.init_variable("B"  , B)
            opt.init_variable("C"  , C)
            opt.init_variable("v"  , f_pts.flatten())
            opt.init_variable("nd" , l_f.flatten())
            
                        
            # Constraints ==========================================
            # # Line congruence l.cu, l.cv = 0
            Supp_E = Supp()
            opt.add_constraint(Supp_E, args=(dual_top, inn_v), w=w_supp)

            # Sphericity
            Sph_E = Sphere()
            opt.add_constraint(Sph_E, args=(ffF, 2), w=w_sphericity)

            Sph_U = Sph_Unit()
            opt.add_constraint(Sph_U, args=(), w=10)

            # # Fairness
            Fair_M = QM_Fairness()
            opt.add_constraint(Fair_M, args=(adj_v, "v", 3), w=w_fairness)

            # Torosal Planarity
            Tor_P = Tor_Planarity()
            opt.add_constraint(Tor_P, args=(mesh_edges, inn_v), w=w_tor)

            # # Proximity
            Prox_M = Proximity()
            opt.add_constraint(Prox_M, args=("v", ref_V, ref_F, 0.01), w=w_proximity)

            Prox_C = Proximity_C()
            opt.add_constraint(Prox_C, args=(ref_C, ref_F, 0.0001), w=w_proximity_c)
            
            # Reg 
            reg = Reg_E()
            opt.add_constraint(reg, args=(e_f_f, e_v_v), w=w_reg)

            # # Define unit variables
            opt.unitize_variable("nd", 3, 10)
            opt.unitize_variable("n_l", 3, 10)

            opt.control_var("v" , 0.2)
            opt.control_var("nd", 1)
            #opt.control_var("c", 0.0001)

            ps.info("Finished Initialization of Optimization 2")

        if psim.Button("Optimize 2"):
            if init_opt_2:
                state2 = 1
            else:
                ps.warning("First Optimization not initialized")
                state2 = 0

            
    if state:    
        counter += 1
        # Optimize
        opt.get_gradients() # Compute J and residuals
        opt.optimize() # Solve linear system and update variables
        
        # Get Variables
        oA, oB, oC = opt.uncurry_X("A", "B", "C")
        
        vk = f_pts
        oB = oB.reshape(-1,3)
        

        sph_c, rf = Implicit_to_CR(oA,oB,oC)

        v0, v1, v2, v3 = vk[ffF[:, 0]], vk[ffF[:, 1]], vk[ffF[:, 2]], vk[ffF[:, 3]]

        dif0 = np.linalg.norm(v0 - sph_c, axis=1)**2 - rf**2
        dif1 = np.linalg.norm(v1 - sph_c, axis=1)**2 - rf**2
        dif2 = np.linalg.norm(v2 - sph_c, axis=1)**2 - rf**2
        dif3 = np.linalg.norm(v3 - sph_c, axis=1)**2 - rf**2


        planarity1 = np.zeros(len(dual_top))
        planarity  = np.zeros(len(dual_top))
        for i in range(len(dual_top)):
            if len(dual_top[i]) == 4:
                c0, c1, c2, c3 = sph_c[dual_top[i]]
                planarity1[i] =  compute_volume_of_tetrahedron(c0, c1, c2, c3)
                planarity[i]  =  compute_planarity(c0, c1, c2, c3)


        sphericity =  dif0**2 + dif1**2 + dif2**2 + dif3**2

        idx_max = np.argmax(sphericity)
        #max_sph = ps.register_point_cloud("Max_Sphericity", np.array([sph_c[idx_max]]), enabled=True, transparency=0.2, color=(0.1, 0.1, 0.1))
        #max_sph.set_radius(rf[idx_max], relative=False)

        # for _ in range(10):
        #     i = np.random.randint(0, len(ffF))
        #     sph = ps.register_point_cloud(f"s_"+str(i), np.array([sph_c[i]]), enabled=True, transparency=0.2, color=(0.1, 0.1, 0.1))
        #     sph.set_radius(rf[i], relative=False)
    
        mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
        mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
        mesh.add_vector_quantity("lc", l_f, enabled=True, length=0.12)
        mesh.add_scalar_quantity("Sphericity", np.array(sphericity), defined_on="faces", enabled=True )
        c_surf = ps.register_surface_mesh("Opt_C", sph_c, dual_top)
        c_surf.add_scalar_quantity("Planarity", planarity, defined_on='faces', enabled=True)
        c_surf.add_scalar_quantity("Face_Vol" , planarity1, defined_on='faces', enabled=True)
        #c_surf.add_vector_quantity("Normals", nd[inn_v], defined_on='faces', enabled=True)
        ps.register_curve_network("Edges", sph_c, dual_edges, radius=0.001)
        ps.register_point_cloud("cc", sph_c, radius=0.002)

    if state2:    
        counter2 += 1
        # Optimize
        opt.get_gradients() # Compute J and residuals
        opt.optimize() # Solve linear system and update variables
        
        # Get Variables
        vk, oA, oB, oC, nd = opt.uncurry_X("v", "A", "B", "C", "nd")
        
        vk = vk.reshape(-1,3)
        oB = oB.reshape(-1,3)
        nd = nd.reshape(-1,3)

        vc = np.sum(vk[ffF], axis=1)/4

        sph_c, rf = Implicit_to_CR(oA,oB,oC)

        v0, v1, v2, v3 = vk[ffF[:, 0]], vk[ffF[:, 1]], vk[ffF[:, 2]], vk[ffF[:, 3]]

        dif0 = np.linalg.norm(v0 - sph_c, axis=1)**2 - rf**2
        dif1 = np.linalg.norm(v1 - sph_c, axis=1)**2 - rf**2
        dif2 = np.linalg.norm(v2 - sph_c, axis=1)**2 - rf**2
        dif3 = np.linalg.norm(v3 - sph_c, axis=1)**2 - rf**2


        planarity1 = np.zeros(len(dual_top))
        planarity  = np.zeros(len(dual_top))
        for i in range(len(dual_top)):

            if len(dual_top[i]) == 4:
                c0, c1, c2, c3 = sph_c[dual_top[i]]
                planarity1[i] =  compute_volume_of_tetrahedron(c0, c1, c2, c3)
                planarity[i]  =  compute_planarity(c0, c1, c2, c3)


        sphericity =  dif0**2 + dif1**2 + dif2**2 + dif3**2

        #idx_max = np.argmax(sphericity)

        
        #max_sph = ps.register_point_cloud("Max_Sphericity", np.array([sph_c[idx_max]]), enabled=True, transparency=0.2, color=(0.1, 0.1, 0.1))
        #max_sph.set_radius(rf[idx_max], relative=False)

        # for _ in range(10):
        #     i = np.random.randint(0, len(ffF))
        #     sph = ps.register_point_cloud(f"s_"+str(i), np.array([sph_c[i]]), enabled=True, transparency=0.2, color=(0.1, 0.1, 0.1))
        #     sph.set_radius(rf[i], relative=False)
    
        mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
        mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
        mesh.add_scalar_quantity("Sphericity", np.array(sphericity), defined_on="faces", enabled=True )
        mesh.add_vector_quantity("lc", nd, enabled=True, length=0.12)
        c_surf = ps.register_surface_mesh("Opt_C", sph_c, dual_top)
        c_surf.add_scalar_quantity("Planarity", planarity, defined_on='faces', enabled=True)
        # c_surf.add_scalar_quantity("Face_Vol" , planarity1, defined_on='faces', enabled=True)
        c_surf.add_vector_quantity("Normals", nd[inn_v], defined_on='faces', enabled=True)
        ps.register_curve_network("Edges", sph_c, dual_edges, radius=0.001)
        ps.register_point_cloud("cc", sph_c, radius=0.002)

        centers = ps.register_point_cloud("cc", sph_c[465:468])

        centers.add_vector_quantity("Dir", 1.2*(vc[465:468] - sph_c[465:468]), enabled=True, vectortype="ambient")

        # pc = ps.register_point_cloud("V_LC", vk[inn_v], radius=0.002)
        # pc.add_vector_quantity("LC", nd[inn_v], enabled=True)


    psim.Separator()


    if psim.Button("Report"):
        opt.get_energy_per_constraint()

    psim.Separator()

    psim.TextUnformatted("Visualization Sphere")

    change, idx_sph = psim.InputInt("Index", idx_sph)

    if psim.Button("Draw Sphere"):

        # Get Variables
        oA, oB, oC = opt.uncurry_X( "A", "B", "C")
        
        oB = oB.reshape(-1,3)
        
        sph_c, rf = Implicit_to_CR(oA,oB,oC)


        sph = ps.register_point_cloud("s_"+str(idx_sph), np.array([sph_c[idx_sph]]), enabled=True, transparency=0.3, color=(1, 0, 0))
        sph.set_radius(rf[idx_sph], relative=False)
    
    
    if psim.Button("Draw Support"):

        # # Get results
        vk, n_l = opt.uncurry_X("v",  "n_l")

        n_l = n_l.reshape(-1,3)
        vk  = vk.reshape(-1,3)

        
        
        # # Get vertices per edge
        vi, vj = vk[mesh_edges[0]], vk[mesh_edges[1]]


        vij = vj - vi
        dist = np.linalg.norm(vij, axis=1)

        vij = vij - np.einsum('ij,ij->i', vij, n_l)[:, None]*n_l
        b1 = vij/np.linalg.norm(vij, axis=1)[:, None]

        print(b1.shape, n_l.shape)

        # Compute orthogonal vector
        b2 = np.cross(n_l, b1)

       

        supp_obj = open(os.path.join(remeshing_dir, name+'_supp_opt.obj'), 'w')


        # delta value of widht for plane
        delta = 0.01


        for i in range(len(mesh_edges[0])):
            
            p0 = vi[i] - delta*b1[i] - delta*b2[i]
            p1 = vi[i] + (dist[i] + delta)*b1[i] - delta*b2[i]
            p2 = vi[i] + (dist[i] + delta)*b1[i] + (dist[i] + delta)*b2[i]
            p3 = vi[i] - delta*b1[i] + (dist[i] + delta)*b2[i]
            
            verts = np.array([p0,p1,p2,p3])
            #draw_polygon(verts, (0.1, 0, 0.2), name="plane"+str(i))

            supp_obj.write(f"v {p0[0]} {p0[1]} {p0[2]}\n")
            supp_obj.write(f"v {p1[0]} {p1[1]} {p1[2]}\n")
            supp_obj.write(f"v {p2[0]} {p2[1]} {p2[2]}\n")
            supp_obj.write(f"v {p3[0]} {p3[1]} {p3[2]}\n")
            supp_obj.write(f"f {4*i + 1} {4*i + 2} {4*i + 3} {4*i + 4} \n")

        supp_obj.close()
        
        # mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
        # mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
        # ps.register_surface_mesh("Opt_C", sph_c, dual_top)

    psim.Separator()
    psim.TextUnformatted("Save Results")
    if psim.Button("Save"):

        # Get results
                  
        # Get Variables
        vk, oA, oB, oC, nl = opt.uncurry_X("v", "A", "B", "C", "n_l")
        
        vk = vk.reshape(-1,3)
        oB = oB.reshape(-1,3)
        nl = nl.reshape(-1,3)

        sph_c, rf = Implicit_to_CR(oA,oB,oC)

        v0, v1, v2, v3 = vk[ffF[:, 0]], vk[ffF[:, 1]], vk[ffF[:, 2]], vk[ffF[:, 3]]

        # save mesh
        write_obj(os.path.join(remeshing_dir, name+'_mesh_opt.obj'), vk, ffF)

        # Save spheres ======================================================================
        # Open file
        exp_file = open(os.path.join(remeshing_dir, name+'_sphs_opt.dat'), 'w')

        exp_file2 = open(os.path.join(remeshing_dir, name+'_sphs_p_cut.dat'), 'w')

        # Write inn_f in the first line
        for i in inn_f:
            exp_file.write(f"{i} ")
        exp_file.write("\n")


        for i in range(len(ffF)):
            
            exp_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} {vc[i][0]} {vc[i][1]} {vc[i][2]} ")

            string_data_sph_panel = f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} "

            for j in range(len(ffF[i])):

                #edge
                e = [ffF[i][j], ffF[i][(j+1) % len(ffF[i])]]

                # find index of edge
                idx = search_edge(tuples_edges, e)

                # Get the normal of the edge
                n = nl[idx]

                # Get the vertices of the edge
                vi, vj = vk[e[0]], vk[e[1]]

                # Compute the mid point of the edge
                mid = (vi + vj)/2
                string_data_sph_panel += f"{vi[0]} {vi[1]} {vi[2]} {mid[0]} {mid[1]} {mid[2]} {n[0]} {n[1]} {n[2]} "
        

                

            exp_file2.write(string_data_sph_panel + "\n")
            

            for j in f_f_adj[i]:
                exp_file.write(f"{j} ")

            exp_file.write("\n")

            

        exp_file2.close()
        exp_file.close()


#print("Mesh", ffV[:10])
         
ps.init()

mesh = ps.register_surface_mesh("mesh", f_pts, ffF)
mesh.add_vector_quantity("lc", l_f, enabled=True)
#ps.register_surface_mesh("S_uv", ref_V, ref_F)
#ps.register_surface_mesh("Remeshed", ffV, ffF)
#ps.register_surface_mesh("Dual mesh", vc, dual_top)
#centers = ps.register_point_cloud("cc", cc[465:468])

#centers.add_vector_quantity("Dir", 1.2*(vc[465:468] - cc[465:468]), enabled=True, vectortype="ambient")


#ps.register_point_cloud("Boundary", f_pts[bd_v], enabled=True, color=(0.1, 0.1, 0.1))
ps.register_surface_mesh("C_uv", ref_C, ref_F, enabled=False)
#ps.register_surface_mesh("Mid_mesh", VR, ffF)


ps.set_user_callback(optimization)
ps.show()



