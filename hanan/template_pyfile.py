# This file is a template for creating a new python file in the hananLab/hanan directory

# Import the necessary libraries
import os
import sys
from pathlib import Path

# Obtain the path HananLab; this is the parent directory of the hananLab/hanan directory
# <Here you can manually add the path direction to the hananLab/hanan directory>
path = os.path.dirname(Path(__file__).resolve().parent)
sys.path.append(path)

# Verification of the path
print(path)

# Import the necessary libraries for visualization and computation
import igl
import polyscope as ps
import numpy as np

# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from hanan.geometry.mesh import Mesh
from hanan.geometry.utils import *

# Optimization classes
from hananLab.hanan.optimization.Template import Template
from hanan.optimization.Optimizer import Optimizer


# Here you can add the directory where you want to save the results
dir_path = os.path.join(path, 'hanan')


# Your code here
# Example:
# If your optimiation is simple and only need vertices and faces you can use igl to read the mesh
# v, f = igl.read_triangle_mesh(os.path.join(path, "data", "bunny.obj"))
# then work just with v and f.

# If you need to use the Mesh class to handle more complex operations related to the topology 
# of the mesh or read a non-triangle mesh you can use the Mesh class.
# mesh = Mesh() # Create an empty mesh
# mesh.read(os.path.join(path, "data", "bunny.obj")) # Read the mesh
# v = mesh.v # Vertices
# f = mesh.f # Faces
# nV = mesh.nV # Number of vertices
# nF = mesh.nF # Number of faces
# face_face_adjacency = mesh.face_face_adjacency() # Face-face adjacency array
# vertex_face_adjacency = mesh.vertex_face_adjacency() # Vertex-face adjacency array
# ...

# Example of optimization
# Load mesh data
v, f = create_hex_faces(2, 1, 1)

# Get any other data that you need ...
nV = v.shape[0]

# Create the optimizer
opt = Optimizer()

# Add variables to the optimizer
# Example:
opt.add_variable("sph_c", (nV, 3)) # Sphere center
opt.add_variable("v", (nV, 3)) # Sphere radius

# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", 0.8, 0)

# Add the constraint to the optimizer
# Example:
temp = Template() 
# add_constraint(name, args, weight)
opt.add_constraint(temp, args=(nV), w=1.0)

# Optimize loop
for i in range(10):
    # Get gradients
    opt.get_gradients() # Compute J and residuals
    opt.optimize() # Solve linear system and update variables

# Print in command line the energy per constraint and best energy
opt.get_energy_per_constraint()

# Save the results in a directory 
save_path = os.path.join(dir_path, "results")
opt.report_energy(save_path)

