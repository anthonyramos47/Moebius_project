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
from energies.Support import Supp
from energies.Sphericity import Sphericity
from energies.Proximity_Gen import Proximity
from energies.Edge_Lenght import Edge_L
from energies.QM_Fairness import QM_Fairness
from energies.Laplacian import Laplacian
from optimization.Optimizer import Optimizer




# Create the parser
parser = argparse.ArgumentParser(description="Visualizer Parser")

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



dual_edges = np.array(list(extract_edges(dual_top)))

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
w_fairness = 1
w_sphericity = 1/max(rads)
w_supp = 1
iter_per_opt = 1
w_lap = 0.1
state = 0
counter = 0
init_opt = False
name_saved = "Post_optimization"
def optimization():

    global w_proximity, w_fairness, w_sphericity, w_supp, iter_per_opt, init_opt, state, counter, vc, nd, f_pts, rads, dual_top, ffF, ref_V, ref_F, ref_u, ref_v, cc, vc, opt, inn_v, bd_v, adj_v, name_saved,e_f_f, e_v_v, w_lap

    # Title
    psim.TextUnformatted("Post Optimization GUI")

    if psim.Button("Stop"):
        state = 0
        

    psim.PushItemWidth(150)
    changed, iter_per_opt = psim.InputInt("Num of Iterations per Run: ", iter_per_opt)
    psim.Separator()
       
    # Inputs Opt 1
    if psim.CollapsingHeader("Optimization 1:"):
        changed, w_proximity  = psim.InputFloat("Proximity W", w_proximity)
        changed, w_fairness   = psim.InputFloat("Fairness W", w_fairness)
        changed, w_sphericity = psim.InputFloat("Sphericity W", w_sphericity)
        changed, w_supp       = psim.InputFloat("Support W", w_supp)
        #changed, w_lap        = psim.InputFloat("Laplacian W", w_lap)

        # State handler
        if counter%iter_per_opt == 0:
            state = 0

        if psim.Button("Init First opt"):

            # Set init to True
            init_opt = True


            opt = Optimizer()

            # Add variables to the optimizer
            opt.add_variable("c" , len(vc)*3   ) # Centers of spheres
            opt.add_variable("nd", len(nd)*3   ) # Normals of dual faces
            opt.add_variable("v" , len(f_pts)*3) # Vertices of mid mesh
            opt.add_variable("r" , len(rads)   ) # Radii of spheres
            opt.add_variable("mu", len(dual_edges))

            # Initialize Optimizer ("Method", step, verbosity)
            opt.initialize_optimizer("LM", 0.45, 1)

            # Initialize variables
            opt.init_variable("c"  , cc.flatten()) 
            opt.init_variable("nd" , nd.flatten())
            opt.init_variable("v"  , f_pts.flatten())
            opt.init_variable("r"  , rads)
            opt.init_variable("mu" ,  0.1)

                        
            # Constraints ==========================================
            # Line congruence l.cu, l.cv = 0
            Supp_E = Supp()
            opt.add_constraint(Supp_E, args=(dual_top, 2), w=w_supp)

            # Sphericity
            Sph_E = Sphericity()
            opt.add_constraint(Sph_E, args=(ffF, 2), w=w_sphericity)

            # Fairness
            Fair_M = QM_Fairness()
            opt.add_constraint(Fair_M, args=(bd_v, adj_v, "v", 3), w=w_fairness)

            # Proximity
            Prox_M = Proximity()
            opt.add_constraint(Prox_M, args=("v", ref_V, ref_F, 0.001), w=w_proximity)

            # Edge Length
            Edge_l = Edge_L()
            opt.add_constraint(Edge_l, args=("c", dual_edges, 3), w=1)

            # Lap = Laplacian()
            #opt.add_constraint(Lap, args=(f_f_adj, "c", 3), w=w_lap)

            Prox_C = Proximity()
            opt.add_constraint(Prox_C, args=("c", ref_C, ref_F, 0.001), w=w_proximity)

            # Fair_C = QM_Fairness()
            # opt.add_constraint(Fair_C, args=(in_v_c, ad_v_c, "c", 3), w=0.002)

            # Define unit variables
            opt.unitize_variable("nd", 3, 10)

            #opt.control_var("v", 0.001)
            #opt.control_var("c", 0.001)

            ps.info("Finished Initialization of Optimization 1")


        if psim.Button("Optimize 1"):
            if init_opt:
                state = 1
            else:
                ps.warning("First Optimization not initialized")
                state = 0
            
        if state:    
            counter += 1
            # Optimize
            opt.get_gradients() # Compute J and residuals
            opt.optimize() # Solve linear system and update variables
            
            # Get Line congruence
            vk, sph_c, nd, rf = opt.uncurry_X("v", "c", "nd", "r")
            sph_c = sph_c.reshape(-1, 3)
            vk = vk.reshape(-1,3)
            nd = nd.reshape(-1,3)

            v0, v1, v2, v3 = vk[ffF[:, 0]], vk[ffF[:, 1]], vk[ffF[:, 2]], vk[ffF[:, 3]]

            dif0 = np.linalg.norm(v0 - sph_c, axis=1)**2 - rf**2
            dif1 = np.linalg.norm(v1 - sph_c, axis=1)**2 - rf**2
            dif2 = np.linalg.norm(v2 - sph_c, axis=1)**2 - rf**2
            dif3 = np.linalg.norm(v3 - sph_c, axis=1)**2 - rf**2


            planarity1 = np.zeros(len(dual_top))
            planarity  = np.zeros(len(dual_top))
            for i in range(len(dual_top)):
                if len(dual_top[i]) != 4:
                    print(dual_top[i])
                if len(dual_top[i]) == 4:
                    c0, c1, c2, c3 = sph_c[dual_top[i]]
                    planarity1[i] =  compute_volume_of_tetrahedron(c0, c1, c2, c3)
                    planarity[i]  =  compute_planarity(c0, c1, c2, c3)


            sphericity =  dif0**2 + dif1**2 + dif2**2 + dif3**2

            idx_max = np.argmax(sphericity)

            max_sph = ps.register_point_cloud("Max_Sphericity", np.array([sph_c[idx_max]]), enabled=True, transparency=0.2, color=(0.1, 0.1, 0.1))
            max_sph.set_radius(rf[idx_max], relative=False)

        
            mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
            mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
            mesh.add_scalar_quantity("Sphericity", np.array(sphericity), defined_on="faces", enabled=True )
            c_surf = ps.register_surface_mesh("Opt_C", sph_c, dual_top)
            c_surf.add_scalar_quantity("Planarity", planarity, defined_on='faces', enabled=True)
            c_surf.add_scalar_quantity("Face_Vol" , planarity1, defined_on='faces', enabled=True)
            c_surf.add_vector_quantity("Normals", nd, defined_on='faces', enabled=True)
            ps.register_curve_network("Edges", sph_c, dual_edges)
            ps.register_point_cloud("cc", sph_c)

    
    psim.Separator()


    if psim.Button("Report"):
        opt.get_energy_per_constraint()

    psim.Separator()

    if psim.Button("Draw Support Structure"):

        # Get results
        vk, sph_c, _, rf = opt.uncurry_X("v", "c", "nd", "r")
        sph_c = sph_c.reshape(-1, 3)
        vk = vk.reshape(-1,3)

        # Get vertices per edge
        vi, vj = vk[e_v_v[0]], vk[e_v_v[1]]

        # Get sphere indices per edge
        c_i, c_j = sph_c[e_f_f[0]], sph_c[e_f_f[1]]

        dir = c_j - c_i
        d = np.linalg.norm(dir, axis=1)
        dir /= d[:, None]

        ri, rj = rf[e_f_f[0]], rf[e_f_f[1]]

        mc = c_i + ((ri**2 - rj**2 + d**2)/(2*d))[:, None]*dir
        # # Dir vector between spheres
        # dij  = c_j - c_i
        # dij /= np.linalg.norm(dij, axis=1)[:, None]

        # # Point of bisecting plane between spheres
        # mc =  c_i +  np.einsum('ij,ij->i', (vi - c_i), dij)[:, None]*dij

        v1mc = vi - mc
        v2mc = vj - mc

        supp_obj = open(os.path.join(remeshing_dir, name+'_supp_opt.obj'), 'w')

        for i in range(len(mc)):
            verts = np.array([mc[i] + 0.998*v1mc[i], mc[i] + 0.998*v2mc[i], mc[i] + 1.001*v2mc[i], mc[i] + 1.001*v1mc[i]])
            #draw_polygon(verts, (0.1, 0, 0.2), name="plane"+str(i))

            
            supp_obj.write(f"v {verts[0][0]} {verts[0][1]} {verts[0][2]}\n")
            supp_obj.write(f"v {verts[1][0]} {verts[1][1]} {verts[1][2]}\n")
            supp_obj.write(f"v {verts[2][0]} {verts[2][1]} {verts[2][2]}\n")
            supp_obj.write(f"v {verts[3][0]} {verts[3][1]} {verts[3][2]}\n")

            supp_obj.write(f"f {4*i + 1} {4*i + 2} {4*i + 3} {4*i + 4} \n")

        supp_obj.close()
        
        mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
        mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
        ps.register_surface_mesh("Opt_C", sph_c, dual_top)

    psim.Separator()
    psim.TextUnformatted("Save Results")
    if psim.Button("Save"):

        # Get results
        vk, sph_c, rf  = opt.uncurry_X("v", 'c', 'r')
        vk = vk.reshape(-1,3)
        sph_c = sph_c.reshape(-1, 3)

        # save mesh
        write_obj(os.path.join(remeshing_dir, name+'_mesh_opt.obj'), vk, ffF)

        # Save spheres ======================================================================
        # Open file
        exp_file = open(os.path.join(remeshing_dir, name+'_sphs_opt.dat'), 'w')

        # Write inn_f in the first line
        for i in inn_f:
            exp_file.write(f"{i} ")
        exp_file.write("\n")

        for i in range(len(ffF)):

            exp_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} {vc[i][0]} {vc[i][1]} {vc[i][2]} ")

            for j in f_f_adj[i]:
                exp_file.write(f"{j} ")

            exp_file.write("\n")

        exp_file.close()


#print("Mesh", ffV[:10])
         
ps.init()

mesh = ps.register_surface_mesh("mesh", f_pts, ffF)
ps.register_surface_mesh("S_uv", ref_V, ref_F)
ps.register_surface_mesh("Remeshed", ffV, ffF)
ps.register_curve_network("Edges", cc, dual_edges)
ps.register_point_cloud("cc", cc)
#ps.register_point_cloud("Boundary", f_pts[bd_v], enabled=True, color=(0.1, 0.1, 0.1))
#ps.register_surface_mesh("C_uv", ref_C, ref_F)
ps.register_surface_mesh("Mid_mesh", VR, ffF)


ps.set_user_callback(optimization)
ps.show()



