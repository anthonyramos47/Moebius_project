import numpy as np
import polyscope as ps
import pickle as pkl
import argparse
import os
import sys
from pathlib import Path
# Obtain the path HananLab
path = os.path.dirname(Path(__file__).resolve().parent)
sys.path.append(path)
print(path)


from hanan.optimization.Optimizer import Optimizer
from hanan.geometry.utils import unit, vec_dot, solve_torsal, planarity_check


Show_Analytical_Torsal = False
Show_spheres = False

# Crear el analizador de argumentos
parser = argparse.ArgumentParser(description='Get direction of the pickle file.')
# Añadir un argumento posicional
parser.add_argument('pickle_file', type=str, help='Pickle file dir')

# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

with open(args.pickle_file, 'rb') as file:
        v, f, t1, t2, tt1, tt2, ne, nc, nr, nt1, nt2 = pkl.load(file)



ec = np.sum(ne[f], axis=1)/3

vv = v + ne
i, j, k = f[:,0], f[:,1], f[:,2]
vi, vj, vk = v[i], v[j], v[k]

vij = vj - vi 
vki = vk - vi

# sph_c = nc
# sph_r = nr
# energy_init = 0
# for id_v in range(3):
#     energy_init += (np.linalg.norm(sph_c - v[f[:,id_v]], axis=1) - sph_r)@(np.linalg.norm(sph_c - v[f[:,id_v]], axis=1) - sph_r) 
#     energy_init += (np.linalg.norm(sph_c - vv[f[:,id_v]], axis=1) - sph_r)@(np.linalg.norm(sph_c - vv[f[:,id_v]], axis=1) - sph_r) 

# print(f"Final energy: {energy_init}")

# Compute planarity
#if Show_Analytical_Torsal:
    # Compute initial torsal directions
at1, at2, a1, a2, b, validity = solve_torsal(vi, vj, vk, ne[i], ne[j], ne[k])

att1 = unit(a1[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))
att2 = unit(a2[:,None]*(vv[j] - vv[i]) + b[:,None]*(vv[k] - vv[i]))

# Compute planarity
planarity1 = planarity_check(t1, tt1, ec)

planarity2 = planarity_check(t2, tt2, ec)



avg_planarity = (planarity1 + planarity2)/2

torsal_angles = np.arccos(abs(vec_dot(unit(nt1), unit(nt2))))*180/np.pi


# Add one column to nc 
sph = np.hstack((nc, nr[:,None]))

# Export sph_c and sph_r in a file
#np.savetxt("sph_c.dat", sph, delimiter="\t", fmt="%1.7f")


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



mesh = ps.register_surface_mesh("mesh", v, f, enabled=True, edge_color = (0,0,0), edge_width=1.7)
mesh2 = ps.register_surface_mesh("mesh 2", vv, f, enabled=True, edge_color = (0,0,0), edge_width=1.7)
mesh.add_vector_quantity("line", ne, length=1, vectortype="ambient", enabled=True, radius=0.0005, color=(0,0,0))
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
mesh.add_scalar_quantity("Torsal angles", torsal_angles, defined_on="faces", enabled=True, cmap="coolwarm")

# Visualize sphere radius as scalar quantity
mesh.add_scalar_quantity("radius sphere", nr, defined_on="faces", enabled=True, cmap="coolwarm")
#mesh.add_scalar_quantity("H", h, defined_on="vertices", enabled=True, cmap="coolwarm")

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

ps.show()