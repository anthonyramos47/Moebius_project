import os 
import sys

# Add hananLab to path
#path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
path = os.path.dirname(os.getcwd())
sys.path.append(path)

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


np.random.seed(0)
IT = 100
weights = {"sphericity" : 1, 
           "linecong" : 1, 
           "torsal" : 1, 
           "line_fair" : 0.0001, 
           "torsal_fair" : 0,
           "torsal_angle": 1
           }
Show = True
Show_Analytical_Torsal = True
Type_Init = 0 # 0: Random, 1: Offset

# Define paths
dir_path = os.getcwd()
data_path = dir_path+"/approximation/data/" # data path

# Load test mesh
#v, f = igl.read_triangle_mesh(os.path.join(data_path, "New_Tri_mesh.obj"))
v, f = igl.read_triangle_mesh(os.path.join(data_path, "Coarse_tri_mesh.obj"))

# fix seed
# 
# v = np.random.rand(4,3)
# f = np.array([[0,1,2], [0,2,3]])

n = igl.per_vertex_normals(v, f)

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

# Get vertex indices of each edge
ed_i, ed_j = mesh.edge_vertices()

ed_i = ed_i[inner_edges]
ed_j = ed_j[inner_edges]

# Get oposite vertex indices of each edge
ed_k, ed_l = mesh.edge_oposite_vertices()

ed_k = ed_k[inner_edges]
ed_l = ed_l[inner_edges]


# Fix direction
signs = np.sign(np.sum(n * ([0,0,-1]), axis=1))
n = n * signs[:, None]

# compute line congruence


# Offset
if Type_Init == 1:
    e = 20 * n
else:
    # Random 
    # a = 15
    # b = 40
    # r = np.random.rand(len(n),1)
    # r = a + (b-a)*r

    # print(r[:5])
    e = (6*np.random.rand(len(n),1)+20)*n

# Compute second envelope
vv = v + e

# Compute indices of vertices of each face
i, j, k = f[:,0], f[:,1], f[:,2]


# Compute guess for sphere
ct, _, nt = circle_3pts(v[i], v[j], v[k])
ct2, _, nt2 = circle_3pts(vv[i], vv[j], vv[k])


for bf in boundary_faces:
    for bv in f[bf]:
        if bv in boundary_vertices:
            n[bv] = nt[bf]
            np.delete(boundary_vertices, np.where(boundary_vertices == bv))


vc  = np.sum(v[f], axis=1)/3
vvc = np.sum(vv[f], axis=1)/3


# Compute sphere center
signs = np.sign(np.sum(nt * ([0,0,-1]), axis=1))
nt = nt * signs[:, None]

sph_c = ct + 0.5*np.linalg.norm(ct2 - ct,axis=1)[:,None]*nt


# Compute sphere radius
sph_r = np.linalg.norm(sph_c - v[i], axis=1)

# Compute number of variables
nV = len(v)
nF = len(f)
nIE = len(inner_edges)


# Define variable indices
var_idx = {     "e"     : np.arange( 0            , 3*nV),  # Line congruence
                "sph_c" : np.arange( 3*nV         , 3*nV +  3*nF),  # Sphere centers
                "sph_r" : np.arange( 3*nV +  3*nF , 3*nV +  4*nF),  # Sphere radius
                "th"    : np.arange( 3*nV +  4*nF , 3*nV +  5*nF),  # theta  angle <t1, vji
                "phi"   : np.arange( 3*nV +  5*nF , 3*nV +  6*nF),  # phi  angel < t1, t2
                "nt1"   : np.arange( 3*nV +  6*nF , 3*nV +  9*nF),  # Normal torsal plane t1 
                "nt2"   : np.arange( 3*nV +  9*nF , 3*nV +  12*nF),  # Normal torsal plane t2
                "u"     : np.arange( 3*nV +  12*nF , 3*nV + 12*nF + 3*nIE),  # Normal torsal plane t2
                "alpha" : np.arange( 3*nV +  12*nF + 3*nIE , 3*nV +  13*nF + 3*nIE )  # Angle between nt1 and nt2
        }


# Init X 
X = np.zeros(sum(len(arr) for arr in var_idx.values()))


X[var_idx["e"]]      = e.flatten()
X[var_idx["sph_c"]]  = sph_c.flatten()
X[var_idx["sph_r"]]  = sph_r
X[var_idx["alpha"]]  = 0.1

t1, t2, a1, a2, b, validity = solve_torsal(v[i], v[j], v[k] , e[i], e[j], e[k])

tt1 = unit(a1[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))
tt2 = unit(a2[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))

ec = np.sum(e[f], axis=1)/3

# Init Sphericity
sphericity = Sphericity()
sphericity.initialize_constraint(X, var_idx, f, v)
sphericity.set_weigth(weights["sphericity"])

# Init Line Congruence Fairnes
line_fair = LineCong_Fair()
line_fair.initialize_constraint(X, var_idx, vertex_adj, inner_vertices) 
line_fair.set_weigth(weights["line_fair"])

# Init Line Cong
linecong = LineCong()
linecong.initialize_constraint(X, var_idx, len(v),  dual_top, inner_vertices)
linecong.set_weigth(weights["linecong"])

# Init Torsal 
torsal = Torsal()
torsal.initialize_constraint(X, var_idx, v, f)
torsal.set_weigth(weights["torsal"])

# Init Torsal angle

tang = Torsal_angle()
tang.initialize_constraint(X, var_idx, v, f)
tang.set_weigth(weights["torsal_angle"])

# Init Torsal 

torsal_fair = Torsal_Fair()
torsal_fair.initialize_constraint(X, var_idx, v, e, inner_edges, ed_i, ed_j, ed_k, ed_l)
torsal_fair.set_weigth(weights["torsal_fair"])

#  Optimizer
optimizer = Optimizer()
optimizer.initialize_optimizer(X, var_idx, "LM", 0.5, 1)


for _ in range(IT):
    # optimizer.unitize_variable("nt1", 3)
    # optimizer.unitize_variable("nt2", 3)
    
    optimizer.get_gradients(sphericity)
    optimizer.get_gradients(linecong)
    optimizer.get_gradients(torsal)
    optimizer.get_gradients(line_fair)
    optimizer.get_gradients(torsal_fair)
    optimizer.get_gradients(tang)
    optimizer.optimize()


## Extract variables
ne, nc, nr, th, phi, nt1, nt2 = optimizer.uncurry_X("e", "sph_c", "sph_r", "th", "phi", "nt1", "nt2")

ne = ne.reshape((-1,3))
nc = nc.reshape((-1,3))
nt1 = nt1.reshape((-1,3))
nt2 = nt2.reshape((-1,3))

vv = v + ne
i, j, k = f[:,0], f[:,1], f[:,2]
vi, vj, vk = v[i], v[j], v[k]

vij = vj - vi 
vki = vk - vi

if Show_Analytical_Torsal:
    # Compute initial torsal directions
    t1, t2, a1, a2, b, validity = solve_torsal(vi, vj, vk, ne[i], ne[j], ne[k])

    tt1 = unit(a1[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))
    tt2 = unit(a2[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))

else:
    t1 = unit( np.cos(th)[:,None]*vij +  np.sin(th)[:,None]*vki)
    t2 = unit(np.cos(th + phi)[:,None]*vij + np.sin(th + phi)[:,None]*vki)

    tt1 = unit( np.cos(th)[:,None]*(vv[j] - vv[i]) +  np.sin(th)[:,None]*(vv[k] - vv[i]))
    tt2 = unit(np.cos(th + phi)[:,None]*(vv[j] - vv[i]) + np.sin(th + phi)[:,None]*(vv[k] - vv[i]))

# Compute planarity
planarity1 = planarity_check(t1, tt1, ec)

planarity2 = planarity_check(t2, tt2, ec)

avg_planarity = (planarity1 + planarity2)/2

torsal_angles = np.arccos(abs(vec_dot(unit(nt1), unit(nt2))))*180/np.pi

## Visualize
ps.init()
ps.remove_all_structures()

# Show boundary spheres
for id in range(len(boundary_faces)):
    i = boundary_faces[id]
    c = nc[i] 
    sphere = ps.register_point_cloud(f"sphere_c{i}", np.array([c]), enabled=True, color=(0,0,0), transparency=0.5)
    sphere.set_radius(nr[i], relative=False)



ps.register_point_cloud("cr", ct, enabled=True, radius=0.001, color=(0,0,0))
mesh = ps.register_surface_mesh("mesh", v, f)
mesh2 = ps.register_surface_mesh("mesh 2", vv, f)
mesh.add_vector_quantity("n", ne, length=1, enabled=True, vectortype="ambient", radius=0.0005, color=(0,0,0))


# Show sphere circumcircle axis
#mesh.add_vector_quantity(  "n",   nt, defined_on="faces", length=0.5, radius=0.0005, enabled=True,  color=(0.5,0,0.8))
#mesh2.add_vector_quantity("n2", -nt2, defined_on="faces", length=0.5, radius=0.0005, enabled=True,  color=(0,0.5,0))


# Visualize planarity as scalar quantity
mesh.add_scalar_quantity("Validity", validity, defined_on="faces", enabled=True, cmap="jet")
mesh.add_scalar_quantity("planarity1", planarity1, defined_on="faces", enabled=True, cmap="coolwarm")
mesh.add_scalar_quantity("planarity2", planarity2, defined_on="faces", enabled=True, cmap="coolwarm")
mesh.add_scalar_quantity("Avg planarity", avg_planarity, defined_on="faces", enabled=True, cmap="coolwarm")
mesh.add_scalar_quantity("Torsal angles", torsal_angles, defined_on="faces", enabled=True, cmap="coolwarm")

# Visualize sphere radius as scalar quantity
mesh.add_scalar_quantity("radius sphere", nr, defined_on="faces", enabled=True, cmap="coolwarm")


mesh.add_vector_quantity("t1", t1, defined_on="faces", length=0.01, enabled=True,  color=(1,1,1))
mesh.add_vector_quantity("t2", t2, defined_on="faces", length=0.01, enabled=True,  color=(0,0,0))
#mesh.add_vector_quantity("Circum normal", -nt, defined_on="faces", length=4, enabled=True, vectortype="ambient", color=(0,0,0))
#mesh.add_vector_quantity("Circum normal1", nt, defined_on="faces", length=4, enabled=True, vectortype="ambient", color=(0,0,0))

if Show:
    ps.show()
