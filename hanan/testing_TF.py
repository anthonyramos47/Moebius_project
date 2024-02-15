import os 
import sys
#from tqdm import tqdm

# Add hananLab to path
# #path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
# path = os.path.dirname(os.getcwd())
# sys.path.append(path)
from pathlib import Path
import pickle

# Obtain the path HananLab
path = os.path.dirname(Path(__file__).resolve().parent)
sys.path.append(path)
print(path)

# Directory path
dir_path = os.path.join(path, 'hanan')

import igl
import polyscope as ps
import numpy as np


from hanan.geometry.mesh import Mesh
from hanan.geometry.utils import *
from hanan.optimization.Sphericity import Sphericity
from hanan.optimization.Optimizer import Optimizer
from hanan.optimization.LineCong import LineCong
from hanan.optimization.LineCong_Fairness import LineCong_Fair
from hanan.optimization.Torsal import Torsal
from hanan.optimization.Torsal_fairness import Torsal_Fair
from hanan.optimization.Torsal_angle import Torsal_angle
from hanan.optimization.Sphere_angle import Sphere_angle


np.random.seed(0)
import json
import argparse

# Crear el analizador de argumentos
parser = argparse.ArgumentParser(description='Get a json file with parameters for the optimization.')
# Añadir un argumento posicional
parser.add_argument('json', type=str, help='Json with parameters for the optimization file path')

# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

with open(args.json, 'r') as f:
    data = json.load(f)

weights = data["weights_Opt"]
IT = data["iterations"]

Show = data["show"]
Show_Analytical_Torsal = data["analytical_torsal"]
Show_spheres = data["show_spheres"]

dir = -np.array([0,0,1])

data_path = dir_path+"/approximation/data/" # data path

# Load test mesh
v, f = igl.read_triangle_mesh(os.path.join(data_path, data["name_mesh"]))

v = normalize_vertices(v, 3)

igl.write_triangle_mesh("original_mesh.obj", v, f)

# v = np.array([
#     [ 1, 0, 0],
#     [-1, 0.01, 0],
#     [ 0, 1, 0],
#     [0.01, -0.8, 0.01]
# ])

# v = v + np.array([0,0, 0.1*np.random.rand()])

# f = np.array([
#     [0, 1, 2],
#     [0, 3, 1],
# ])

# Compute normals
n = igl.per_vertex_normals(v, f)





# # Compute the mean curvature
# h = 0.5*abs(k1 + k2)
# mean_r = 1/h

# Create mesh
mesh = Mesh()
mesh.make_mesh(v, f)

# Compute mesh properties
dual_top          = mesh.vertex_ring_faces_list()
inner_vertices    = mesh.inner_vertices()
vertex_adj        = mesh.vertex_adjacency_list()
boundary_faces    = mesh.boundary_faces()
boundary_vertices = mesh.boundary_vertices()
inner_edges       = mesh.inner_edges()
f_f_adj = mesh.face_face_adjacency_list()
d_f = mesh.dual_top()

auxf1, auxf2      = mesh.edge_faces()
ev1  , ev2        = mesh.edge_vertices()

# Faces of the edges
f1 = auxf1[inner_edges]
f2 = auxf2[inner_edges]

# Vertices of the edges
ev1 = ev1[inner_edges]
ev2 = ev2[inner_edges]

# Normals at the edges
ne = n[ev1] + n[ev2]

d_f = mesh.dual_top()

sphere_connectivity = open("connectivity.dat", "w")

for i in range(len(f_f_adj)):
    for j in range(len(f_f_adj[i])-1):
        sphere_connectivity.write(f"{f_f_adj[i][j]}\t")
    sphere_connectivity.write(f"{f_f_adj[i][-1]}\n")

sphere_connectivity.close() 

# Get vertex indices of each edge
ed_i, ed_j = mesh.edge_vertices()

# Get inner edge data only
ed_i = ed_i[inner_edges]
ed_j = ed_j[inner_edges]

# Get oposite vertex indices of each edge
ed_k, ed_l = mesh.edge_oposite_vertices()

# Get inner edge data only
ed_k = ed_k[inner_edges]
ed_l = ed_l[inner_edges]


signs = np.sign(np.sum(n * (dir), axis=1))
n = n * signs[:, None]


_, _, k1, k2 = igl.principal_curvature(v, f)

# # compute line congruence
offset = data["offset"]
# # Offset
#e = -(4/(k1+k2))[:, None] * n

e = offset * n

# Compute second envelope
vv = v + e

# Compute indices of vertices of each face
i, j, k = f[:,0], f[:,1], f[:,2]

# Compute baricenter of each face
vc  = np.sum(v[f], axis=1)/3
vvc = np.sum(vv[f], axis=1)/3

# Compute guess for sphere
ct, cr, nt = circle_3pts(v[i], v[j], v[k])
ct2, _, nt2 = circle_3pts(vv[i], vv[j], vv[k])

for bf in boundary_faces:
    for bv in f[bf]:
        if bv in boundary_vertices:
            n[bv] = nt[bf]
            np.delete(boundary_vertices, np.where(boundary_vertices == bv))

# Compute sphere center
signs = np.sign(np.sum(nt * (dir), axis=1))
nt = nt * signs[:, None]

lamb = np.sign(offset)*np.sqrt((offset/2)**2 - cr**2)

sph_c = ct + lamb[:,None]*nt

# Compute sphere radius
sph_r = offset/2*np.ones(len(f))


energy = 0
# Measure sphericity
for sph_i in range(len(sph_c)):
    # Get vertices of the face
    v_f = v[f[sph_i]]
    # Get vertices of the second envelope face
    vv_f = vv[f[sph_i]]

    # Compute the distance of each vertex to the sphere
    dist = np.linalg.norm(sph_c[sph_i] - v_f, axis=1)
    dist2 = np.linalg.norm(sph_c[sph_i] - vv_f, axis=1)

    # Compute the energy
    energy += np.sum((dist - sph_r[sph_i])**2) + np.sum((dist2 - sph_r[sph_i])**2)

print(f"Initial energy: {energy}")


# Compute number of variables
nV = len(v)
nF = len(f)
nIE = len(inner_edges)


#  Optimizer
optimizer = Optimizer()
# Define the variables
optimizer.add_variable("e", 3*nV)
optimizer.add_variable("sph_c", 3*nF)
optimizer.add_variable("sph_r", nF)
optimizer.add_variable("th", nF)
optimizer.add_variable("phi", nF)
optimizer.add_variable("nt1", 3*nF)
optimizer.add_variable("nt2", 3*nF)
optimizer.add_variable("u", 3*nIE)
optimizer.add_variable("alpha", nF)
optimizer.add_variable("le", nV)


# Init X 
X = np.zeros(optimizer.var)


X[optimizer.var_idx["e"]]      = e.flatten()
X[optimizer.var_idx["sph_c"]]  = sph_c.flatten()
X[optimizer.var_idx["sph_r"]]  = sph_r
X[optimizer.var_idx["le"]]     = offset
X[optimizer.var_idx["alpha"]]  = 0.5

# Initialize the variables
optimizer.initialize_optimizer(X, "LM", 0.5, 1)

# Define the constraints
# Init Sphericity
sphericity = Sphericity()
optimizer.add_constraint(sphericity,  args=(f, v), w=2 )


#optimizer.initialize_optimizer(X, var_idx, "LM", 0.5, 1)

print(optimizer.uncurry_X("sph_c")[:5])

print("Test-Optimization started")
for _ in range(100):
    optimizer.get_gradients()
    optimizer.optimize()

print("Test Optimization finished")

print(optimizer.uncurry_X("sph_c")[:5])


# t1, t2, a1, a2, b, validity = solve_torsal(v[i], v[j], v[k] , e[i], e[j], e[k])

# tt1 = unit(a1[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))
# tt2 = unit(a2[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))

ec = np.sum(e[f], axis=1)/3




#  pre_Optimizer
pre_Opt = Optimizer()
pre_Opt.initialize_optimizer(X, var_idx, "LM", 0.8, 1)


print("Pre-Optimization started")
for _ in range(100):
    pre_Opt.get_gradients()
    pre_Opt.optimize()

print("Pre Optimization finished")
X = pre_Opt.bestX

sph_c, sph_r = pre_Opt.uncurry_X("sph_c", "sph_r")

sph_c = sph_c.reshape((-1,3))
sph_r = np.abs(sph_r)
# Add one column to nc 
sph = np.hstack((sph_c, sph_r[:,None]))
# Export sph_c and sph_r in a file
np.savetxt("Init_sph_c.dat", sph, delimiter="\t", fmt="%1.7f")

print("Second Optimization")
# Init Sphericity
sphericity = Sphericity()
sphericity.initialize_constraint(X, var_idx, f, v)
sphericity.set_weigth(weights["sphericity"])

# Init Line Congruence Fairnes
line_fair = LineCong_Fair()
line_fair.initialize_constraint(X, var_idx, vertex_adj, inner_vertices, n) 
line_fair.set_weigth(weights["line_fair"])

# Init Line Cong
linecong = LineCong()
linecong.initialize_constraint(X, var_idx, len(v),  dual_top, inner_vertices, n)
linecong.set_weigth(weights["linecong"])

# Init Torsal 
torsal = Torsal()
torsal.initialize_constraint(X, var_idx, v, f)
torsal.set_weigth(weights["torsal"])

# Init Sphere angle
smooth = Sphere_angle() 
smooth.initialize_constraint(X, var_idx, ne, v[ev1], v[ev2], inner_edges, f1, f2)
smooth.set_weigth(weights["smoothness"])

# Init Torsal angle
tang = Torsal_angle()
tang.initialize_constraint(X, var_idx, v, f)
tang.set_weigth(weights["torsal_angle"])

# Init Torsal 
torsal_fair = Torsal_Fair()
torsal_fair.initialize_constraint(X, var_idx, v, inner_edges, ed_i, ed_j, ed_k, ed_l)
torsal_fair.set_weigth(weights["torsal_fair"])



for _ in range(IT):
    optimizer.unitize_variable("nt1", 3)
    optimizer.unitize_variable("nt2", 3)
    
    optimizer.get_gradients(sphericity)
    optimizer.get_gradients(torsal)
    optimizer.get_gradients(linecong)
    optimizer.get_gradients(line_fair)
    optimizer.get_gradients(smooth)
    optimizer.get_gradients(tang)
    optimizer.get_gradients(torsal_fair)

    optimizer.optimize()

optimizer.get_energy_per_constraint()
optimizer.report_energy()

## Extract variables
ne, nc, nr, th, phi, nt1, nt2, le = optimizer.uncurry_X("e", "sph_c", "sph_r", "th", "phi", "nt1", "nt2", "le")

ne = ne.reshape((-1,3))
nc = nc.reshape((-1,3))
nt1 = nt1.reshape((-1,3))
nt2 = nt2.reshape((-1,3))

ec = np.sum(ne[f], axis=1)/3

vv = v + ne
i, j, k = f[:,0], f[:,1], f[:,2]
vi, vj, vk = v[i], v[j], v[k]

vij = vj - vi 
vki = vk - vi


# Compute planarity
#if Show_Analytical_Torsal:
    # Compute initial torsal directions
at1, at2, a1, a2, b, validity = solve_torsal(vi, vj, vk, ne[i], ne[j], ne[k])

att1 = unit(a1[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))
att2 = unit(a2[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))

#else:
t1 = unit( np.cos(th)[:,None]*vij +  np.sin(th)[:,None]*vki)
t2 = unit(np.cos(th + phi)[:,None]*vij + np.sin(th + phi)[:,None]*vki)

tt1 = unit( np.cos(th)[:,None]*(vv[j] - vv[i]) +  np.sin(th)[:,None]*(vv[k] - vv[i]))
tt2 = unit(np.cos(th + phi)[:,None]*(vv[j] - vv[i]) + np.sin(th + phi)[:,None]*(vv[k] - vv[i]))

# Compute planarity
planarity1 = planarity_check(t1, tt1, ec)

planarity2 = planarity_check(t2, tt2, ec)

aplanarity1 = planarity_check(at1, att1, ec)
aplanarity2 = planarity_check(at2, att2, ec)

# Compute average planarity
aavg_planarity = (aplanarity1 + aplanarity2)/2


avg_planarity = (planarity1 + planarity2)/2

torsal_angles = np.arccos(abs(vec_dot(unit(nt1), unit(nt2))))*180/np.pi

nr = np.abs(nr)
# Add one column to nc 
sph = np.hstack((nc, nr[:,None]))

# Export sph_c and sph_r in a file
np.savetxt("sph_c.dat", sph, delimiter="\t", fmt="%1.7f")


## Visualize
ps.init()
ps.remove_all_structures()

# Show boundary spheres
# for id in range(len(boundary_faces)):
#     i = boundary_faces[id]
#     c = nc[i] 
#     sphere = ps.register_point_cloud(f"sphere_c{i}", np.array([c]), enabled=True, color=(0,0,0), transparency=0.5)
#     sphere.set_radius(nr[i], relative=False)

if Show_spheres:
    for _ in range(10):
        id = np.random.randint(0, len(nc))
        c = nc[id] 
        sphere = ps.register_point_cloud(f"sphere_c{id}", np.array([c]), enabled=True, color=(0,0,0), transparency=0.2)
        sphere.set_radius(nr[id], relative=False)

# edges_i, edges_j = mesh.edge_vertices()

# l = (ne[edges_i] + ne[edges_j])/2

# m_v = (v[edges_i] + v[edges_j])/2

# L = (ne[ed_i] + ne[ed_j])/2

# compute_barycenter_coord_proj(v, vv, inner_edges, L, ed_i, ed_j, ed_k, ed_l)
   

mesh = ps.register_surface_mesh("mesh", v, f, enabled=True, edge_color = (0,0,0), edge_width=1.7)
mesh2 = ps.register_surface_mesh("mesh 2", vv, f, enabled=True, edge_color = (0,0,0), edge_width=1.7)
mesh.add_vector_quantity("line", ne, length=1, vectortype="ambient", enabled=True, radius=0.0005, color=(0,0,0))
ps.register_surface_mesh("Center Mesh", nc, d_f)
mesh.add_scalar_quantity("H", 0.5*(k1 + k2), defined_on="vertices", enabled=True, cmap="jet")
mesh.add_scalar_quantity("HRad", 1/(0.5*(k1 + k2)), defined_on="vertices", enabled=True, cmap="jet")
#mesh.add_vector_quantity("line-e", -ne, length=1, enabled=True, vectortype="ambient", radius=0.0005, color=(0,0,0))

#mid_edges = ps.register_point_cloud("mid_points", m_v, enabled=True, radius=0.001, color=(0,0,0))

#mid_edges.add_vector_quantity("l", - l, length=0.01, enabled=True, vectortype="ambient", radius=0.0005, color=(0,0,0))

# Show sphere circumcircle axis
#mesh.add_vector_quantity(  "n",   nt, defined_on="faces", length=0.5, radius=0.0005, enabled=True,  color=(0.5,0,0.8))
#mesh2.add_vector_quantity("n2", -nt2, defined_on="faces", length=0.5, radius=0.0005, enabled=True,  color=(0,0.5,0))

# Visualize planarity as scalar quantity
# mesh.add_scalar_quantity("Validity", validity, defined_on="faces", enabled=True, cmap="jet")
mesh.add_scalar_quantity("planarity1", planarity1, defined_on="faces", enabled=True, cmap="coolwarm")
mesh.add_scalar_quantity("planarity2", planarity2, defined_on="faces", enabled=True, cmap="coolwarm")
mesh.add_scalar_quantity("Avg planarity", avg_planarity, defined_on="faces", enabled=True, cmap="coolwarm")
mesh.add_scalar_quantity("A Avg planarity", aavg_planarity, defined_on="faces", enabled=True, cmap="coolwarm")
mesh.add_scalar_quantity("Torsal angles", torsal_angles, defined_on="faces", enabled=True, cmap="coolwarm")
#mesh.add_scalar_quantity("LC Lenght", abs(le), enabled=True, cmap="coolwarm")

# Visualize sphere radius as scalar quantity
mesh.add_scalar_quantity("radius sphere", abs(nr), defined_on="faces", enabled=True, cmap="coolwarm")


# Torsal directions
mesh.add_vector_quantity("t1",   t1, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(0,0,0))
mesh.add_vector_quantity("t2",   t2, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(0,0,0))
mesh.add_vector_quantity("-t1", -t1, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(0,0,0))
mesh.add_vector_quantity("-t2", -t2, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(0,0,0))

# Torsal Normals
# mesh.add_vector_quantity("nt1",   nt1, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(1,0,0))
# mesh.add_vector_quantity("nt2",   nt2, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(1,0,0))

if Show_Analytical_Torsal:
    mesh.add_vector_quantity("at1",   at1, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(1,1,1))
    mesh.add_vector_quantity("at2",   at2, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(1,1,1))
    mesh.add_vector_quantity("a-t1", -at1, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(1,1,1))
    mesh.add_vector_quantity("a-t2", -at2, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(1,1,1))

# Principa Culvature Directions
# mesh.add_vector_quantity("v1",   v1, defined_on="vertices", length=0.01, radius=0.002, enabled=True,  color=(1,0,0))
# mesh.add_vector_quantity("v2",   v2, defined_on="vertices", length=0.01, radius=0.002, enabled=True,  color=(1,0,0))
# mesh.add_vector_quantity("-v1", -v1, defined_on="vertices", length=0.01, radius=0.002, enabled=True,  color=(1,0,0))
# mesh.add_vector_quantity("-v2", -v2, defined_on="vertices", length=0.01, radius=0.002, enabled=True,  color=(1,0,0))

# mesh2.add_vector_quantity("tt1", tt1, defined_on="faces", length=0.02, radius=0.001, enabled=True,  color=(0,0,0))
# mesh2.add_vector_quantity("tt2", tt2, defined_on="faces", length=0.02, radius=0.001, enabled=True,  color=(0,0,0))
#mesh.add_vector_quantity("Circum normal", -nt, defined_on="faces", length=4, enabled=True, vectortype="ambient", color=(0,0,0))
#mesh.add_vector_quantity("Circum normal1", nt, defined_on="faces", length=4, enabled=True, vectortype="ambient", color=(0,0,0))

if Show:
    ps.show()

# Optimization_Output = [v, f, t1, t2, tt1, tt2, ne, nc, nr, nt1, nt2 ]

# output_path = data["exp_dir"]

# output_path += data["out_name"]

# optimizer.report_energy(output_path)

# # Open a file for writing. The 'wb' parameter denotes 'write binary'
# with open(output_path+'_Optimization.pkl', 'wb') as file:
#     pickle.dump(Optimization_Output, file)

# # Save weights and iterations used in file 
# save_param = {
#     "name_mesh": data["name_mesh"],
#     "weights": weights,
#     "iterations": IT,
#     "offset": offset 
# }
# with open(output_path+'_weights.json', 'w') as file:
#     json.dump(save_param, file, indent=4)
    