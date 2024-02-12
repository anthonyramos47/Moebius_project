import numpy as np
import polyscope as ps
import pickle as pkl
import argparse
import os
import sys
import json

from pathlib import Path
# Obtain the path HananLab
path = os.path.dirname(Path(__file__).resolve().parent)
sys.path.append(path)
print(path)


from hanan.optimization.Optimizer import Optimizer
from hanan.geometry.utils import unit, vec_dot, solve_torsal, planarity_check, draw_polygon, orth_proj




# Crear el analizador de argumentos
parser = argparse.ArgumentParser(description='Get direction of the pickle file.')
# Añadir un argumento posicional
parser.add_argument('json', type=str, help='parameters json file')
parser.add_argument('pickle_file', type=str, help='Pickle file dir')

# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

with open(args.pickle_file, 'rb') as file:
        v, inner_edges, f, t1, t2, tt1, tt2, ne, nc, nr, nt1, nt2, u, auxf1, auxf2, auxev1, auxev2, ed_i, ed_j, ed_k, ed_l = pkl.load(file)

with open(args.json, 'r') as file_f:
    data = json.load(file_f)


ne = ne.reshape((-1,3))
nc = nc.reshape((-1,3))
nt1 = nt1.reshape((-1,3))
nt2 = nt2.reshape((-1,3))
u = u.reshape((-1,3))


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



# Compute planarity
planarity1 = planarity_check(t1, tt1, ec)

planarity2 = planarity_check(t2, tt2, ec)

aplanarity1 = planarity_check(at1, att1, ec)
aplanarity2 = planarity_check(at2, att2, ec)

# Compute average planarity
aavg_planarity = (aplanarity1 + aplanarity2)/2


avg_planarity = (planarity1 + planarity2)/2

torsal_angles = np.arccos(abs(vec_dot(unit(nt1), unit(nt2))))*180/np.pi

              
## Visualize
ps.init()
ps.remove_all_structures()


if data["show_plane"]:

    # Draw planes 
    m_edge = data["edge"]
    m_pt = (v[auxev1[m_edge]]  + v[auxev2[m_edge]])/2
    e_m  = (ne[auxev1[m_edge]] + ne[auxev2[m_edge]])/2

    # Faces 
    f1 = auxf1[m_edge]
    f2 = auxf2[m_edge]

    # Get torsal directions
    t1f1 = at1[f1]
    t2f1 = at2[f1]

    t1f2 = at1[f2]
    t2f2 = at2[f2]

    index = np.where(inner_edges == m_edge)

    uf1 = u[index][0]

    print("uf1", uf1)

    vi, vj = v[ed_i[index]], v[ed_j[index]]
    vk, vl = v[ed_k[index]], v[ed_l[index]]

    # Project 
    pvi = orth_proj(vi, e_m)
    pvj = orth_proj(vj, e_m)
    pvk = orth_proj(vk, e_m)
    pvl = orth_proj(vl, e_m)

    # Second envelope
    vvi, vvj = vv[ed_i[index]], vv[ed_j[index]]
    vvk, vvl = vv[ed_k[index]], vv[ed_l[index]]

    # Project
    pvvi = orth_proj(vvi, e_m)
    pvvj = orth_proj(vvj, e_m)
    pvvk = orth_proj(vvk, e_m)
    pvvl = orth_proj(vvl, e_m)

    # barycetner of faces
    bf1 = (vi + vj + vk)/3
    bf2 = (vi + vj + vl)/3

    # check directions torsal
    t1f1 *= np.sign(vec_dot(t1f1,  bf1 -  m_pt)) 
    t2f1 *= np.sign(vec_dot(t2f1,  bf1 -  m_pt))

    t1f2 *= np.sign(vec_dot(t1f2,  bf2 -  m_pt))
    t2f2 *= np.sign(vec_dot(t2f2,  bf2 -  m_pt))

    # Check barycentric coordinates
    print("Envelope 1:", pvl - uf1[0]*pvi - uf1[1]*pvj - uf1[2]*pvk)
    print("Envelope 2:", pvvl - uf1[0]*pvvi - uf1[1]*pvvj - uf1[2]*pvvk)

    e_m = -unit(e_m)
    # Draw planes
    draw_polygon(np.array([m_pt, m_pt + t1f1, m_pt + t1f1 + e_m, m_pt + e_m]), (0,0,0), "plane1_f1")
    draw_polygon(np.array([m_pt, m_pt + t2f1, m_pt + t2f1 + e_m, m_pt + e_m]), (0,0,0), "plane2_f1")

    draw_polygon(np.array([m_pt, m_pt + t1f2, m_pt + t1f2 + e_m, m_pt + e_m]), (1,1,1), "plane1_f2")
    draw_polygon(np.array([m_pt, m_pt + t2f2, m_pt + t2f2 + e_m, m_pt + e_m]), (1,1,1), "plane2_f2")


# Show boundary spheres
# for id in range(len(boundary_faces)):
#     i = boundary_faces[id]
#     c = nc[i] 
#     sphere = ps.register_point_cloud(f"sphere_c{i}", np.array([c]), enabled=True, color=(0,0,0), transparency=0.5)
#     sphere.set_radius(nr[i], relative=False)

if data["show_spheres"]:
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
mesh.add_vector_quantity("Line COng", ne, length=1, vectortype="ambient", enabled=True, radius=0.0005, color=(0,0,0))
#mesh.add_vector_quantity("line-e", -ne, length=1, enabled=True, vectortype="ambient", radius=0.0005, color=(0,0,0))

#mid_edges = ps.register_point_cloud("mid_points", m_v, enabled=True, radius=0.001, color=(0,0,0))

#mid_edges.add_vector_quantity("l", - l, length=0.01, enabled=True, vectortype="ambient", radius=0.0005, color=(0,0,0))

# Show sphere circumcircle axis
# mesh.add_vector_quantity(  "n",   nt, defined_on="faces", length=0.005, radius=0.0005, enabled=True,  color=(0.5,0,0.8))
# mesh2.add_vector_quantity("n2", -nt2, defined_on="faces", length=0.005, radius=0.0005, enabled=True,  color=(0,0.5,0))

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
mesh.add_vector_quantity("t1",   t1, defined_on="faces", length=0.005, radius=0.001, enabled=True,  color=(0,0,0))
mesh.add_vector_quantity("t2",   t2, defined_on="faces", length=0.005, radius=0.001, enabled=True,  color=(0,0,0))
mesh.add_vector_quantity("-t1", -t1, defined_on="faces", length=0.005, radius=0.001, enabled=True,  color=(0,0,0))
mesh.add_vector_quantity("-t2", -t2, defined_on="faces", length=0.005, radius=0.001, enabled=True,  color=(0,0,0))

# Torsal Normals
# mesh.add_vector_quantity("nt1",   nt1, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(1,0,0))
# mesh.add_vector_quantity("nt2",   nt2, defined_on="faces", length=0.01, radius=0.001, enabled=True,  color=(1,0,0))

if data["analytical_torsal"]:
    mesh.add_vector_quantity("at1",   at1, defined_on="faces", length=0.005, radius=0.001, enabled=True,  color=(1,1,1))
    mesh.add_vector_quantity("at2",   at2, defined_on="faces", length=0.005, radius=0.001, enabled=True,  color=(1,1,1))
    mesh.add_vector_quantity("a-t1", -at1, defined_on="faces", length=0.005, radius=0.001, enabled=True,  color=(1,1,1))
    mesh.add_vector_quantity("a-t2", -at2, defined_on="faces", length=0.005, radius=0.001, enabled=True,  color=(1,1,1))

# Principa Culvature Directions
# mesh.add_vector_quantity("v1",   v1, defined_on="vertices", length=0.01, radius=0.002, enabled=True,  color=(1,0,0))
# mesh.add_vector_quantity("v2",   v2, defined_on="vertices", length=0.01, radius=0.002, enabled=True,  color=(1,0,0))
# mesh.add_vector_quantity("-v1", -v1, defined_on="vertices", length=0.01, radius=0.002, enabled=True,  color=(1,0,0))
# mesh.add_vector_quantity("-v2", -v2, defined_on="vertices", length=0.01, radius=0.002, enabled=True,  color=(1,0,0))

# mesh2.add_vector_quantity("tt1", tt1, defined_on="faces", length=0.02, radius=0.001, enabled=True,  color=(0,0,0))
# mesh2.add_vector_quantity("tt2", tt2, defined_on="faces", length=0.02, radius=0.001, enabled=True,  color=(0,0,0))
#mesh.add_vector_quantity("Circum normal", -nt, defined_on="faces", length=4, enabled=True, vectortype="ambient", color=(0,0,0))
#mesh.add_vector_quantity("Circum normal1", nt, defined_on="faces", length=4, enabled=True, vectortype="ambient", color=(0,0,0))

ps.show()