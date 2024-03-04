#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Examples for the NURBS-Python Package
    Released under MIT License
    Developed by Onur Rauf Bingol (c) 2018

    Surface fitting by global interpolation
"""

from geomdl import BSpline
import numpy as np

#from geomdl.visualization import VisMPL as vis


# Data set
# points = ((-5, -5, 0), (-2.5, -5, 0), (0, -5, 0), (2.5, -5, 0), (5, -5, 0), (7.5, -5, 0), (10, -5, 0),
#           (-5, 0, 3), (-2.5, 0, 3), (0, 0, 3), (2.5, 0, 3), (5, 0, 3), (7.5, 0, 3), (10, 0, 3),
#           (-5, 5, 0), (-2.5, 5, 0), (0, 5, 0), (2.5, 5, 0), (5, 5, 0), (7.5, 5, 0), (10, 5, 0),
#           (-5, 7.5, -3), (-2.5, 7.5, -3), (0, 7.5, -3), (2.5, 7.5, -3), (5, 7.5, -3), (7.5, 7.5, -3), (10, 7.5, -3),
#           (-5, 10, 0), (-2.5, 10, 0), (0, 10, 0), (2.5, 10, 0), (5, 10, 0), (7.5, 10, 0), (10, 10, 0))
#points = [ f(u, v) for u in np.linspace(-5, 5, 10) for v in np.linspace(-5, 5, 10)]

# Create a BSpline surface instance (Bezier surface)
surf = BSpline.Surface()

# Set degrees
surf.degree_u = 3
surf.degree_v = 2

# Set control points
control_points = [[0, 0, 0], [0, 4, 0], [0, 8, -3],
                  [2, 0, 6], [2, 4, 0], [2, 8, 0],
                  [4, 0, 0], [4, 4, 0], [4, 8, 3],
                  [6, 0, 0], [6, 4, -3], [6, 8, 0]]
surf.set_ctrlpts(control_points, 4, 3)

# Set knot vectors
surf.knotvector_u = [0, 0, 0, 0, 3, 3, 3, 3]
surf.knotvector_v = [0, 0, 0, 1, 1, 1]

# Set evaluation delta (control the number of surface points)
surf.delta = 0.05

# Get surface points (the surface will be automatically evaluated)
surf.evaluate()



u_points = [ i/10 for i in range(10)]
v_points = [ i/10 for i in range(10)]



parops = [ (u_points[i], v_points[j]) for i in range(10) for j in range(10) ]

vals = np.array([ surf.derivatives(i/10, j/10, 2) for i in range(2) for j in range(2) ])

print(np.array([ (i/10,j/10) for i in range(2) for j in range(2) ]) )

surface_points = surf.evalpts

print("Before control change:", surface_points[:5])

new_ctrpts = [[0,0,0] for i in range(len(control_points))]
new_ctrpts[0][0] = 1
new_ctrpts = list(new_ctrpts)

print("ctrpts bef: ", control_points)
print("ctrpts af: ", new_ctrpts)

surf.set_ctrlpts(new_ctrpts, 4, 3)

print("after change:",surf.evalpts[:5])

# dev = surf.derivatives(0, 0.1, 2)



# print(vals[:, 0, 1])
# print(dev[0][1])

# # Visualize data and evaluated points together
import numpy as np
import matplotlib.pyplot as plt
evalpts = np.array(surf.evalpts)
ctpts = np.array(surf.ctrlpts)
pts = np.array(control_points)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(evalpts[:, 0], evalpts[:, 1], evalpts[:, 2])
ax.scatter(ctpts[:, 0], ctpts[:, 1], ctpts[:, 2], color="red")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="blue", alpha=0.5)
#plt.show()