import sys

sys.path.append('C:/Users\david\Google_Drive\GitHub\geolab')

import geolab

try:
    from geopt.sparse_mkl.scipy_aliases import spsolve
except:
    try:
        from pypardiso import spsolve
    except:
        from scipy.sparse.linalg import spsolve
        print('Efficiency Warning: mkl spsolve not available')

try:
    from geopt.sparse_mkl.sparse_dot import dot_product_mkl as spdot
except:
    try:
        from sparse_dot_mkl import dot_product_mkl as spdot
    except:
        from geopt.sparse_mkl.routines import spdot
        print('Efficiency Warning: mkl dot not available')

from geopt.optimization.guidedprojection import GuidedProjection

from geopt.geometry.mesh import Mesh

from geopt.gui.meshoptimizer_gui import MeshOptimizerGUI
