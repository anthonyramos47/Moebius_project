import polyscope as ps
import numpy as np

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *

# Local files
from utils.bsplines_functions import *
from utils.visualization import *

def visualization_opt_lc( opt, bsp, r_uv, n, u_pts, v_pts):
    
     # Get Line congruence
    l, cp = opt.uncurry_X("l", "rij" )

    r_uv[2] = cp

    # Evaluate r(u,v) at grid points
    r_uv_surf = bisplev(u_pts, v_pts, r_uv)

    # # Reshape Line congruence
    l = l.reshape(len(u_pts), len(v_pts), 3)
    l /= np.linalg.norm(l, axis=2)[:,:,None]

    # Angle with normal
    ang_normal = np.arccos( np.sum( l*n, axis=2) )*180/np.pi
    
    surf =ps.get_surface_mesh("Mesh")
    # OPTIMIZED LC
    surf.add_vector_quantity("l", l.reshape(-1, 3), defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.1, 0.0, 0.0))
    surf.add_scalar_quantity("r_uv", r_uv_surf.flatten(), defined_on="vertices", enabled=True)

    # ANGLES WITH NORMAL SCALAR FIELD
    surf.add_scalar_quantity("Angles", ang_normal.flatten(), defined_on="vertices", enabled=True)

    # Transform the b-spline surface to a mesh
    V, F = Bspline_to_mesh(bsp, u_pts, v_pts)

    # Visualization Mid Surface
    V_R = V + r_uv_surf.flatten()[:,None]*n.reshape(-1,3)
    ps.register_surface_mesh("C_uv", V_R, F)


def visualization_opt_torsal( opt, bsp, r_uv, n, u_pts, v_pts):
    
    # Get RESULTS
    l, cp, tu1, tu2, tv1, tv2, nt1, nt2 = opt.uncurry_X("l", "rij", "u1", "u2", "v1", "v2", "nt1", "nt2")

    # Reshape Torsal normals
    nt1 = nt1.reshape(-1,3)
    nt2 = nt2.reshape(-1,3)

    # Compute Torsal angles
    torsal_angles = np.arccos(np.abs(np.sum(nt1*nt2, axis=1)))*180/np.pi

    # Update control points of r(u,v) spline surface
    r_uv[2] = cp  
    r_uv_surf = bisplev(u_pts, v_pts, r_uv)

    # Reshape Line congruence
    l = l.reshape(len(u_pts), len(v_pts), 3)
    l /= np.linalg.norm(l, axis=2)[:,:,None]
    
    # Transform the b-spline surface to a mesh
    V, F = Bspline_to_mesh(bsp, u_pts, v_pts)

    # Angle with normal
    ang_normal = np.arccos( np.sum( l*n, axis=2) )*180/np.pi

    # Get vertices
    v0, v1, v2, v3 = V[F[:,0]], V[F[:,1]], V[F[:,2]], V[F[:,3]]

    # Compute tangents
    du = v2 - v0
    dv = v1 - v3

    l_uv = l.reshape(-1, 3)

    l0, l1, l2, l3 = l_uv[F[:,0]], l_uv[F[:,1]], l_uv[F[:,2]], l_uv[F[:,3]]

    lu = l2 - l0
    lv = l1 - l3


    mean_diagonals = np.mean((np.linalg.norm(du, axis=1) + np.linalg.norm(dv, axis=1))/2)

    size_torsal = mean_diagonals/4

    # Compute barycenters
    barycenters = (v0 + v1 + v2 + v3)/4

    # Get torsal directions
    t1 = unit(tu1[:,None]*du + tv1[:,None]*dv)
    t2 = unit(tu2[:,None]*du + tv2[:,None]*dv)

    lt1 = unit(tu1[:,None]*lu + tv1[:,None]*lv)
    lt2 = unit(tu2[:,None]*lu + tv2[:,None]*lv)

    lc = (l0 + l1 + l2 + l3)/4

    planarity_opt = 0.5*(planarity_check(t1, lt1, lc) + planarity_check(t2, lt2, lc))


    surf = ps.get_surface_mesh("Mesh")

    # OPTIMIZED LC
    surf.add_vector_quantity("l", l.reshape(-1, 3), defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.1, 0.0, 0.0))

    surf.add_scalar_quantity("r_uv", r_uv_surf.flatten(), defined_on="vertices", enabled=True)

    # ANGLES WITH NORMAL SCALAR FIELD
    surf.add_scalar_quantity("Angles", ang_normal.flatten(), defined_on="vertices", enabled=True)

    
    surf.add_scalar_quantity("Torsal_Angles", torsal_angles, defined_on="faces", enabled=True)

    surf.add_scalar_quantity("Planarity", planarity_opt, defined_on="faces", enabled=False)

    torsal_dir_show(barycenters, t1, t2, size=size_torsal, rad=0.0014)
    #torsal_dir_show(barycenters, nt1, nt2, size=size_torsal, rad=0.0014)

    V_R = V + r_uv_surf.flatten()[:,None]*n.reshape(-1,3)

    ps.register_surface_mesh("C_uv", V_R, F)

def visualization_init(bsp, u_pts, v_pts):

    # Transform the b-spline surface to a mesh
    V, F = Bspline_to_mesh(bsp, u_pts, v_pts)

    # Compute the curvatures
    K, H, _ = curvatures_par(bsp, u_pts, v_pts)

    # Flatten values
    H = H.flatten()
    K = K.flatten()

    # Sign of the mean curvature
    H_sign = np.sign(H)

    # Add the mesh to the visualization
    mesh = ps.register_surface_mesh("Mesh", V, F)

    # Visualize curvatures
    mesh.add_scalar_quantity("Mean Curvature", H)
    mesh.add_scalar_quantity("Gaussian Curvature", K)
    mesh.add_scalar_quantity("Sign of the Mean Curvature", H_sign, enabled=True)

