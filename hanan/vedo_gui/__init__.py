#------------------------------------------------------------------------------
import os
import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(path)

print(path)

from hanan.geometry.mesh import Mesh