import splipy as sp
import numpy as np
import matplotlib.pyplot as plt
import json

# Open and read the JSON file
with open('/Users/cisneras/hanan/hananLab/QS_project/Surfjson.json', 'r') as file:
    data = json.load(file)

degree_u = data["degreeU"] + 1
degree_v = data["degreeV"] + 1

knots_u = data["knotsU"]
knots_v = data["knotsV"]

# Fix knots
knots_u = [knots_u[0]] + knots_u + [knots_u[-1]]
knots_v = [knots_v[0]] + knots_v + [knots_v[-1]]

# normalized knots
knots_u = np.array(knots_u) / knots_u[-1]
knots_v = np.array(knots_v) / knots_v[-1]


control_points = np.array(data["controlPoints"]).reshape(degree_u*degree_v,4)

basis_u = sp.BSplineBasis(degree_u, knots_u)   # quadratic basis: 3 functions in the u-direction
basis_v = sp.BSplineBasis(degree_v, knots_v) # 4 quadratic functions in the v-direction

control_points = control_points[:,:3]


surface = sp.Surface(basis_u, basis_v, control_points)


u = np.linspace(basis_u.start(), basis_u.end(),50) # 31 uniformly spaced evaluation points in u (domain (0,1))
v = np.linspace(basis_v.start(), basis_v.end(),50) # 41 uniformly spaced evaluation points in u (domain (0,2))
x = surface(u, v)

normals_sf = surface.normal([0, 0.1], [0, 0.1])


print("Normals: ", normals_sf)

print("Lenght", np.linalg.norm(normals_sf, axis=2))



# optimization

# first we set up our 3D plotting environment
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot the (x,y,z)-coordinates of the surface (computed above)
ax.plot_surface(x[:,:,0], x[:,:,1], x[:,:,2])
ax.scatter(control_points[:,0], control_points[:,1], control_points[:,2], color='r') # plot the control net
# aspect ratio is 1:1:1

min_x = np.min(control_points, axis=0)
max_x = np.max(control_points, axis=0)
scale = max(max_x - min_x)
ax.set_box_aspect((max_x-min_x)/scale)

# show the plot
plt.show()

# show a top-down meshgrid of the surface
# fig = plt.figure()                                # new figure
# ax = fig.add_subplot(111, projection='3d')        # 3d plot
# ax.plot_wireframe(x[:,:,0], x[:,:,1], x[:,:,2])   # plot as wireframe
# ax.view_init(90, -90)                             # view from above (top view)
# plt.show()


