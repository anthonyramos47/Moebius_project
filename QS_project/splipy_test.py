import splipy as sp
import numpy as np
import matplotlib.pyplot as plt



basis_u = sp.BSplineBasis(3, [0,0,0,1,1,1])   # quadratic basis: 3 functions in the u-direction
basis_v = sp.BSplineBasis(3, [0,0,0,1,2,2,2]) # 4 quadratic functions in the v-direction

control_net = [[0,0,0], [1,0,0], [2,0,0],
               [0,2,0], [1,2,1], [2,1,0],
               [1,3,0], [2,2,1], [3,1,0],
               [3,3,0], [3,2,0], [3,1,0]]

    
matrix = np.zeros_like(control_net)

auxmatrix = matrix.copy()
auxmatrix[0,0] = 1

print(auxmatrix)


surface = sp.Surface(basis_u, basis_v, control_net)

dev_a_surfaced = sp.Surface(basis_u, basis_v, auxmatrix)

print("Surface:", surface(0.5, 0.5)) # (x,y,z)-coordinate at the center of the surface
print("Dcp1_x Surface:", dev_a_surfaced(0.5, 0.5)) # (x,y,z)-coordinate at the center of the surface
auxmatrix = matrix.copy()
auxmatrix[0,0] = 1

dev_a_surfaced = sp.Surface(basis_u, basis_v, auxmatrix)

N_v_i = basis_u.evaluate([0.1, 0.5])
M_v_j = basis_v.evaluate([0.1, 0.5])

print("Ni:\n", N_v_i )
print("Mj:\n", M_v_j)
print("N0 x M0", N_v_i[0,0]*M_v_j[0,0])
print("N0 x M1", N_v_i[1,0]*M_v_j[1,0])



u = np.linspace(0,1,31) # 31 uniformly spaced evaluation points in u (domain (0,1))
v = np.linspace(0,2,41) # 41 uniformly spaced evaluation points in u (domain (0,2))
x = surface(u,v)




# first we set up our 3D plotting environment
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the (x,y,z)-coordinates of the surface (computed above)
ax.plot_surface(x[:,:,0], x[:,:,1], x[:,:,2])

control_net = np.array(control_net)
ax.scatter(control_net[:,0], control_net[:,1], control_net[:,2], color='r') # plot the control net
#plt.show()

# show a top-down meshgrid of the surface
fig = plt.figure()                                # new figure
ax = fig.add_subplot(111, projection='3d')        # 3d plot
ax.plot_wireframe(x[:,:,0], x[:,:,1], x[:,:,2])   # plot as wireframe
ax.view_init(90, -90)                             # view from above (top view)
#plt.show()


