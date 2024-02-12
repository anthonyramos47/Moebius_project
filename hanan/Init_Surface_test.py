import numpy as np
import splipy as sp
from geomdl import BSpline
import matplotlib.pyplot as plt
from geomdl.fitting import approximate_surface, interpolate_surface


# Load a file
data = np.loadtxt("/Users/cisneras/sph_dt.dat", dtype=float)



surface = approximate_surface(data, 22, 22, 6, 6)

data = data.reshape(22, 22, 3)

ord_u = surface.degree_u + 1
ord_v = surface.degree_v + 1
ct_pts = surface.ctrlpts
knots_u = surface.knotvector_u
knots_v = surface.knotvector_v

# Create the B-splines basis
basis_u = sp.BSplineBasis(ord_u, knots_u) 
basis_v = sp.BSplineBasis(ord_v, knots_v) 

# Create the B-spline surface
bsp1 = sp.Surface(basis_u, basis_v, ct_pts)

# Sampe size
sample = (50, 50)

u_pts = np.linspace(0+0.1, 1-0.1, sample[0])
v_pts = np.linspace(0+0.1, 1-0.1, sample[1])

# Generate points
s_pts = bsp1(u_pts, v_pts)

# Init figure
fig     = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d') 
print("Surface shape: ", s_pts.shape)

# Plot the B-spline surface
#ax.scatter(data[:, :, 0], data[:, :, 1], data[:, :, 2], color='r')
ax.plot_surface(s_pts[:, :, 0], s_pts[:, :, 1], s_pts[:, :, 2], alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

ax.set_title('B-spline surface ')
plt.show()