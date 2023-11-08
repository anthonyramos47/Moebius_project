import os
import sys

# Add hananLab to path
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

import igl
import vedo as vd
import numpy as np
from hanan.geometry.mesh import Mesh


# Define a function that toggles the transparency of a mesh
#  and changes the button state
def buttonfunc(obj, ename):
    mesh.alpha(1 - mesh.alpha())  # toggle mesh transparency
    bu.switch()                   # change to next status
    printc(bu.status(), box="_", dim=True)