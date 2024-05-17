# This file is a template for creating a new python file in the hananLab/hanan directory

# Import the necessary libraries
import os
import sys
from pathlib import Path

# Obtain the path HananLab; this is the parent directory of the hananLab/hanan directory
# <Here you can manually add the path direction to the hananLab/hanan directory>
path = "/Users/cisneras/hanan/hananLab/hanan"
sys.path.append(path)

# Verification of the path
print(path)

# Import the necessary libraries for visualization and computation
import igl
import polyscope as ps
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import json
# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *

# Optimization classes
from energies.BS_approx import BS_approx
from optimization.Optimizer import Optimizer


# Here you can add the directory where you want to save the results
dir_path = os.getcwd()
print(dir_path)


# Open and read the JSON file
with open('/Users/cisneras/hanan/hananLab/QS_project/data/Bsplines_Surfaces/Surfjson.json', 'r') as file:
    data = json.load(file)

# degree_u = data["degreeU"] + 1
# degree_v = data["degreeV"] + 1

# knots_u = data["knotsU"]
# knots_v = data["knotsV"]

# # Fix knots
# knots_u = [knots_u[0]] + knots_u + [knots_u[-1]]
# knots_v = [knots_v[0]] + knots_v + [knots_v[-1]]

# # normalized knots
# knots_u = np.array(knots_u) / knots_u[-1]
# knots_v = np.array(knots_v) / knots_v[-1]
# control_points = np.array(data["controlPoints"]).reshape(degree_u*degree_v,4)
# control_points = control_points[:,:3]


#Small test
degree_u = 3
degree_v = 4

knots_u = [0, 0, 0, 1, 1, 1]
knots_v = [0, 0, 0, 0, 1, 1, 1, 1]

# Generate set up 12 control points
control_points = [  [0, 0, 0], [0, 4, 0], [0, 8, -3],
                    [2, 0, 6], [2, 4, 0], [2, 8, 0],
                    [4, 0, 0], [4, 4, 0], [4, 8, 3],
                    [6, 0, 0], [6, 4, -3], [6, 8, 0]
                ]


basis_u = sp.BSplineBasis(degree_u, knots_u)   # quadratic basis: 3 functions in the u-direction
basis_v = sp.BSplineBasis(degree_v, knots_v) # 4 quadratic functions in the v-direction

control_points = np.array(control_points)

control_points2 = 2*np.array(control_points) 

surface2 = sp.Surface(basis_u, basis_v, control_points2)

surface = sp.Surface(basis_u, basis_v, control_points)

sample = 60

u = np.linspace(basis_u.start(), basis_u.end(), sample) # 31 uniformly spaced evaluation points in u (domain (0,1))
v = np.linspace(basis_v.start(), basis_v.end(), sample) # 41 uniformly spaced evaluation points in u (domain (0,2))
x = surface(u, v)


x2 = surface2(u, v)

opt = Optimizer()
# optimization
opt.add_variable("cp", len(control_points.flatten())) # Sphere center


# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", 0.5, 1)


sf1 = {
    "basis_u": basis_u,
    "basis_v": basis_v,
    "control_points": control_points
}

sf2 = {
    "basis_u": basis_u,
    "basis_v": basis_v,
    "control_points": control_points2
}


# Add the constraint to the optimizer
# Example:
bs_approx = BS_approx()
# add_constraint(name, args, weight)
opt.add_constraint(bs_approx, args=(sf1, sf2, 10, 10), w=1.0)

# Optimize loop
for i in range(100):
    # Get gradients
    opt.get_gradients() # Compute J and residuals
    opt.optimize() # Solve linear system and update variables

# Print in command line the energy per constraint and best energy
opt.get_energy_per_constraint()

cp2 = opt.X.reshape(len(control_points2),3)
surface3 = sp.Surface(basis_u, basis_v, cp2)
x3 = surface3(u, v)



# first we set up our 3D plotting environment
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot the (x,y,z)-coordinates of the surface (computed above)
ax.plot_surface(x[:,:,0], x[:,:,1], x[:,:,2])
ax.plot_surface(x2[:,:,0], x2[:,:,1], x2[:,:,2])
#ax.plot_surface(x3[:,:,0], x3[:,:,1], x3[:,:,2])

ax.scatter(control_points[:,0], control_points[:,1], control_points[:,2], color='r') # plot the control net
ax.scatter(control_points2[:,0], control_points2[:,1], control_points2[:,2], color='g') # plot the control net
#ax.scatter(cp2[:,0], cp2[:,1], cp2[:,2], color='b') # plot the control net
# aspect ratio is 1:1:1

min_x = np.min(control_points, axis=0)
max_x = np.max(control_points, axis=0)
scale = max(max_x - min_x)
ax.set_box_aspect((max_x-min_x)/scale)

# show the plot
plt.show()


