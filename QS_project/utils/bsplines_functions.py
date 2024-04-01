import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import polyscope as ps
import json
import os
from scipy.interpolate import bisplev, bisplrep
from geomdl.fitting import approximate_surface
import splipy as sp
from geometry.utils import *



def normals_bsp(bsp, u_pts, v_pts):
    """
    Function to compute the normals of the surface
    Input:
        bsp: B-spline surface
        u_pts: u points
        v_pts: v points
    """

    # Compute the first and second derivatives
    du  = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=1, dy=0).flatten()
    dv  = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=0, dy=1).flatten()

    print(du[:5])
    
    du  = np.hstack((np.ones_like(u_pts.flatten())[:,None], v_pts.flatten()[:, None], du[:, None]))
    dv  = np.hstack((u_pts.flatten()[:, None], np.ones_like(v_pts.flatten())[:, None], dv[:, None]))
    
    # Compute the normal
    n = np.cross(du, dv)
    print(np.where(np.linalg.norm(n, axis=1) == 0))
    n = n/np.linalg.norm(n, axis=1)[:, None]

    return n


def curvatures_par(bsp, u_pts, v_pts):
    """
    Function to compute the curvatures of the surface
    Input:
        bsp: B-spline surface
        u_pts: u points
        v_pts: v points
    Output:
        K: Gaussian curvature
        H: Mean curvature
        n: Normals
    """    
        
    # Compute normals
    n = bsp.normal(u_pts, v_pts)

    # Compute derivatives
    # First derivatives
    du  = bsp.derivative(u_pts, v_pts,  d=(1, 0))
    dv  = bsp.derivative(u_pts, v_pts,  d=(0, 1))
    # Second derivatives
    duv = bsp.derivative(u_pts, v_pts,  d=(1, 1))
    duu = bsp.derivative(u_pts, v_pts,  d=(2, 0))
    dvv = bsp.derivative(u_pts, v_pts,  d=(0, 2))

    # Compute the first and second fundamental forms
    E = np.sum(du*du, axis=2)
    F = np.sum(du*dv, axis=2)
    G = np.sum(dv*dv, axis=2)

    L = np.sum(duu*n, axis=2)
    M = np.sum(duv*n, axis=2)
    N = np.sum(dvv*n, axis=2)

    K = (L*N - M**2)/(E*G - F**2)
    H = (E*N + G*L - 2*F*M)/(2*(E*G - F**2))

    return K, H, n
    


def curvatures_graph(bsp, u_pts, v_pts):
    """
    Function to compute the curvatures of the surface
    """

    # Compute the first and second derivatives
    fu  = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=1, dy=0).flatten()
    fv  = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=0, dy=1).flatten()
    fuu = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=2, dy=0).flatten()
    fvv = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=0, dy=2).flatten()
    fuv = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=1, dy=1).flatten()

  
    # flatten each
    # Compute with flatten arrays
    # Reshape end result
    
    # Compute the normal
    norms_n = np.sqrt(fu**2 + fv**2 + 1)
 
    
    # Gauss curvature
    K = (fuu*fvv - fuv**2)/(norms_n**4)

    # Mean curvature
    H = (fuu - 2*fu*fuv*fv + fuu*fv*2 + fvv + fu**2*fvv)/(2*norms_n**3)

    K = K.reshape(u_pts.shape[0], v_pts.shape[0])
    H = H.reshape(u_pts.shape[0], v_pts.shape[0])

    return K, H  


def read_bspline_json(dir):
    """
    Function to read the bspline surface from a json file
    Input: 
        dir: Direction to the json file
    """

    # Open and read the JSON file
    with open(dir, 'r') as file:
        data = json.load(file)
    
    
    # The order of the surface in each direction is the degree + 1
    order_u = data["degreeU"] + 1
    order_v = data["degreeV"] + 1
    
    # Print the order of the surface in each direction
    # 
    print("Reading B-Spline from Json:")
    print("U dir order: ", order_u )
    print("V dir order: ", order_v )

    # Get the knots
    knots_u = data["knotsU"]
    knots_v = data["knotsV"]

    # Fix knots, add one copy of the first and last knot
    knots_u = [knots_u[0]] + knots_u + [knots_u[-1]]
    knots_v = [knots_v[0]] + knots_v + [knots_v[-1]]

    # Normalized knots to the interval [0, 1]
    knots_u = np.array(knots_u) / knots_u[-1]
    knots_v = np.array(knots_v) / knots_v[-1]

    # Get the control points
    control_points = np.array(data["controlPoints"]).reshape(-1,4)
    control_points = np.array(control_points[:,:3])

    # Return data
    return control_points, knots_u, knots_v, order_u, order_v


def sample_grid(u_sample, v_sample, delta=0.05):
    """
    Function to sample the grid points to be used for the Bspline surface
    """

    # Compute the grid points
    u_pts = np.linspace(0 + delta, 1 - delta, u_sample)
    v_pts = np.linspace(0 + delta, 1 - delta, v_sample)

    return u_pts, v_pts

def central_spheres(bsp1, u_vals, v_vals):
    """
    Function to compute the central spheres of the surface per each point
    Input:
        bsp1: Bspline surface
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
    """

    # Compute mean curvature
    _, H, n = curvatures_par(bsp1, u_vals, v_vals)

    mean_H = np.mean(H)

    # Search for H close to zero
    idx = np.where(H < 0.0001)

    print("idx len", len(idx[0]))

    H0 = H.copy()

    H[idx[0]] = mean_H

    # Compute the radius
    r_H = 1/H

    # Evaluate surface
    s_uv = bsp1(u_vals, v_vals)

    # Compute the center
    c = s_uv + r_H[:,:,None]*n

    return c, r_H, H0, n

def line_congruence_uv(bsp, r_uv, u_vals, v_vals):
    """
    Function to compute the line congruence of the surface per each point
    Input:
        bsp: Bspline surface S(u,v): R^2 -> R^3
        ruv: (t: knots, c: control points; k: degrees) Bspline surface of the radius r(u,v): R^2 -> R 
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
    """

    # Compute derivatives of the central spheres surface c(u,v) = s(u,v) + r(u,v)*n(u,v)
    cu, cv = sphere_congruence_derivatives(bsp, r_uv, u_vals, v_vals)

    # Take the cross product of the derivatives
    l = np.cross(cu, cv, axis=2)

    l = l/np.linalg.norm(l, axis=2)[:,:,None]

    return l
    
def offset_spheres(bsp, u_vals, v_vals, offset):
    """
    Function to compute the offset spheres of the surface per each point
    Input:
        bsp: Bspline surface S(u,v): R^2 -> R^3
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
        offset: offset value
    """

    n = bsp.normal(u_vals, v_vals)

    c = bsp(u_vals, v_vals) + (offset/2)*n

    r = offset/2*np.ones((u_vals.shape[0], v_vals.shape[0]))

    return c, r

def sphere_congruence_derivatives(bsp, r_uv, u_vals, v_vals):
    """
    Function to compute the derivatives of the central spheres with respect to u and v
    Input:
        bsp: Bspline surface S(u,v): R^2 -> R^3
        ruv: (t: knots, c: control points; k: degrees) Bspline surface of the radius r(u,v): R^2 -> R 
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
    """

    # Compute the derivatives of the surface s(u,v)
    su = bsp.derivative(u_vals, v_vals, d=(1, 0))
    sv = bsp.derivative(u_vals, v_vals, d=(0, 1))

    n = bsp.normal(u_vals, v_vals)

    # Compute the derivatives of the Gauss map
    nu, nv = normal_derivatives_uv(bsp, u_vals, v_vals)
    
    # Compute the radius function r(u,v) at grid points
    r = bisplev(u_vals, v_vals, r_uv)

    # Compute the derivatives of the radius
    ru = bisplev(u_vals, v_vals, r_uv, dx=1, dy=0)
    rv = bisplev(u_vals, v_vals, r_uv, dx=0, dy=1)

    # Compute the derivatives of the center
    cu = su + ru[:,:,None]*n + r[:,:,None]*nu
    cv = sv + rv[:,:,None]*n + r[:,:,None]*nv

    return cu, cv

def normal_derivatives_uv(bsp, u_vals, v_vals):
    """
    Function tha compute the derivatives of the normal vector in the u and v directions
    Input:
        bsp: Bspline surface S(u,v): R^2 -> R^3
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
    """

    # Compute the derivatives of the surface s(u,v)
    su = bsp.derivative(u_vals, v_vals, d=(1, 0))
    sv = bsp.derivative(u_vals, v_vals, d=(0, 1))

    suu = bsp.derivative(u_vals, v_vals, d=(2, 0))
    svv = bsp.derivative(u_vals, v_vals, d=(0, 2))
    suv = bsp.derivative(u_vals, v_vals, d=(1, 1))

    # Compute the normal
    n = np.cross(su, sv)
    n_norm = np.linalg.norm(n, axis=2)[:, :, None]

    # Compute the derivatives of the normal
    # nu = f1/n_norm + f2/n_norm - n*(f1 + f2)@n)/n_norm**3
    suu_sv = np.cross(suu, sv, axis=2)
    su_suv = np.cross(su, suv, axis=2)
    suv_sv = np.cross(suv, sv, axis=2)
    su_svv = np.cross(su, svv, axis=2)


    nu = suu_sv/n_norm + su_suv/n_norm - n*np.sum((suu_sv + su_suv)*n, axis=2)[:,:,None]/n_norm**3
    nv = suv_sv/n_norm + su_svv/n_norm - n*np.sum((suv_sv + su_svv)*n, axis=2)[:,:,None]/n_norm**3

    return nu, nv


def r_uv_fitting(u_vals, v_vals, r_H, u_degree=5, v_degree=5):
    """
    Function to fit the function r(u,v) of radius of the central spheres, to a Bspline surface.
    Input:
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
        r_H: radius of the central spheres
    Output:
        r_uv: Bspline surface of the radius
    """

    # Get grid points
    u_grid, v_grid = np.meshgrid(u_vals, v_vals, indexing='ij')

    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()
    r_H = r_H.flatten()

    # Fit the radius to a Bspline surface
    r_uv = bisplrep(u_grid, v_grid, r_H, kx= u_degree, ky= v_degree)

    return r_uv


# ====================== Plotting Functions =================

def plot_scalar_value(ax, Surface, scalar, name):
    """
    Function to plot the scalar value on a surface. Meant to be used to plot the curvature values.
    Input:
        Surface       : Is a matrix u_size * v_size * 3, with n the number of points in the u direction and m the number of points in the v direction
        scalar        :  Is a matrix u_size * v_size, with the scalar value to be plotted per each point in the surface
        name          :    The name of the plot str
    """

    scalar_flatten = scalar.flatten()

    # Create a colormap
    cmap = plt.cm.RdBu # You can choose any existing colormap as a starting point
    norm = plt.Normalize(vmin=min(scalar_flatten), vmax=max(scalar_flatten))
    
    ax.plot_surface(Surface[:, :, 0], Surface[:, :, 1], Surface[:, :, 2], cmap =plt.cm.plasma,  facecolors=cmap(norm(scalar)), 
    vmin=0, vmax=1, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # # Add colorbar for reference
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(scalar_flatten)
    colorbar = plt.colorbar(mappable, ax=ax, orientation='vertical', shrink=0.6)
    colorbar.set_label(name+' Values')
    colorbar.set_ticks(np.linspace(np.min(scalar_flatten), np.max(scalar_flatten), 5))


def add_control_points(ax, control_points):
    """
    Function to add the control points to the plot
    Input:
        ax             : The axis of the plot
        control_points : The control points of the surface
    """
    ax.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2], c='r', marker='o', label='Control Points')
    # Set the aspect ratio of the plot to be equal
    min_x = np.min(control_points, axis=0)
    max_x = np.max(control_points, axis=0)
    scale = np.max(max_x - min_x)
    ax.set_box_aspect((max_x-min_x)/scale)


def plot_surface(ax, Surface, name):
    """
    Function to plot the B-spline surface
    Input:
        Surface       : Is a matrix u_size * v_size * 3, with n the number of points in the u direction and m the number of points in the v direction
        control_points: Is a matrix num_ctrl pots * 3, with the control points of the surface
        name          :    The name of the plot str
    """

    print("Surface shape: ", Surface.shape)

    # Plot the B-spline surface
    ax.plot_surface(Surface[:, :, 0], Surface[:, :, 1], Surface[:, :, 2], alpha=0.8,
    label=name)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.legend()

    #ax.set_title('B-spline surface '+name)
    


def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter

    return x, y, z


def approx_surface_from_data(file, box_size=2):
    """
    Function to approximate the surface from the data. 
    Input:
        file: File with the data
    """
    print("Reading data from file: ", file)

    # Read the data
    data = np.loadtxt(file)

    # Number of u, v pts 
    u_pts = int(data[0,0])
    v_pts = int(data[0,1])

    # Degree of u,v 
    u_degree = int(data[1,0])
    v_degree = int(data[1,1])

    # Points fitting
    #pts = normalize_vertices(data[2:], 2)
    pts = data[2:]

    geom_bsp = approximate_surface(pts, u_pts+1, v_pts+1, u_degree, v_degree)


    # Get controp points
    ctrl_pts = geom_bsp.ctrlpts

    # Knots
    u_knots = geom_bsp.knotvector_u
    v_knots = geom_bsp.knotvector_v

    # Order 
    u_order = geom_bsp.order_u
    v_order = geom_bsp.order_v


    # Create Bspline basis 
    basis_u = sp.BSplineBasis(u_order, u_knots)
    basis_v = sp.BSplineBasis(v_order, v_knots)

    # Create the B-spline surface
    bsp = sp.Surface(basis_u, basis_v, ctrl_pts)

    return bsp


# ====================== Optimization GUI Functions =================


def get_spline_data(choice_data, surface_dir, bspline_surf_name):
    """ Function to get the B-spline surface data
    Input:
        choice_data: Choice of the data type
        surface_dir: Directory of the surface
        bspline_surf_name: B-spline surface name
    Output:
        bsp1: B-spline surface
    """

    if choice_data == 0:
        # Define the path to the B-spline surface
        bspline_surf_path = os.path.join(surface_dir, bspline_surf_name + ".json")
        print("bspline_surf_path:", bspline_surf_path)

        # Load the B-spline surface
        control_points, knots_u, knots_v, order_u, order_v = read_bspline_json(bspline_surf_path)

        # Scale control points
        ctrl_pts_shape = control_points.shape
        flat_ctrl_pts  = control_points.reshape(-1,3)
        norm_ctrl_pts  = normalize_vertices(flat_ctrl_pts, 2)
        control_points = norm_ctrl_pts.reshape(ctrl_pts_shape)

        # Create the B-splines basis
        basis_u = sp.BSplineBasis(order_u, knots_u) 
        basis_v = sp.BSplineBasis(order_v, knots_v) 

        # Create the B-spline surface
        bsp1 = sp.Surface(basis_u, basis_v, control_points)
    else:
        data_bspline = "data_hyp.dat"
        data_bspline = os.path.join(surface_dir, data_bspline)
        bsp1 = approx_surface_from_data(data_bspline)

    return bsp1

def init_sphere_congruence(mid_init, bsp1, u_pts, v_pts, sample):
    """
    Function to get a sphere congruence.
    Input:
        mid_init: type of initialization
        bsp1: B-spline surface
        u_pts: u points
        v_pts: v points
        sample: sample size
    Output:
        c: center of the spheres
        r_H: radius of the spheres
        H: mean curvature
        n: normals
    """

    if mid_init == 0:
        # Compute central spheres radius and normals
        _, r, _, n = central_spheres(bsp1, u_pts, v_pts) 
    else: 
        n = bsp1.normal(u_pts, v_pts)
        r = 5*np.ones((sample[0], sample[1])) 
    
    return r, n


def Bspline_to_mesh(bsp1, u_pts, v_pts, sample):
    """
    Function to convert the B-spline surface to a mesh.
    Input:
        bsp1: B-spline surface
        u_pts: u points
        v_pts: v points
        sample: sample size
    Output:
        V: Vertices of the mesh
        F: Faces of the mesh
    """

    # Mesh Visualization ========================== 
    S_flat = bsp1(u_pts, v_pts).reshape(-1,3)

    # Get Grid as quad mesh V and F
    V = S_flat
    # Faces F_i = [i, i+1, sample[1]*i + i, sample[1]*i + i]
    F = np.array([  (sample[1]*j) +np.array([ i, i+1, sample[1] + i + 1, sample[1] + i])   for j in range(sample[1] - 1)for i in range(sample[0] - 1)] )

    return V, F


def visualize_LC(surf, r_uv, l, n, u_pts, v_pts, V, F,  cp):
    """ 
    Function to visualize the line congruence and centers of the spheres
    Here we visualized the optimize line congruence, the angles with the normal and the optimized centers of the spheres.
    Input:
        surf: polyscope surface
        r_uv: Bspline surface of the radius
        l: line congruence
        n: normals
        u_pts: u points
        v_pts: v points
        V: Vertices of the mesh
        F: Faces of the mesh
        cp: number of control points
    """
    
    # Update control points of r(u,v) spline surface
    r_uv[2] = cp  

    # Evaluate r(u,v) at grid points
    r_uv_surf = bisplev(u_pts, v_pts, r_uv)

    # Reshape Line congruence
    l = l.reshape(len(u_pts), len(v_pts), 3)
    l /= np.linalg.norm(l, axis=2)[:,:,None]

    # Angle with normal
    ang_normal = np.arccos( np.sum( l*n, axis=2) )*180/np.pi

    # OPTIMIZED LC
    surf.add_vector_quantity("l", l.reshape(-1, 3), defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.1, 0.0, 0.0))
    surf.add_scalar_quantity("r_uv", r_uv_surf.flatten(), defined_on="vertices", enabled=True)

    # ANGLES WITH NORMAL SCALAR FIELD
    surf.add_scalar_quantity("Angles", ang_normal.flatten(), defined_on="vertices", enabled=True)

    # Visualization Mid Surface
    V_R = V + r_uv_surf.flatten()[:,None]*n.reshape(-1,3)
    ps.register_surface_mesh("C_uv", V_R, F)

def visualization_LC_Torsal(surf, opt, r_uv, u_pts, v_pts, n, V, F):
    """
    Function to visualize the line congruence and the torsal angles
    Input:
        surf: polyscope surface
        opt: optimization object
        r_uv: Bspline surface of the radius
        u_pts: u points
        v_pts: v points
        n: normals
        V: Vertices of the mesh
        F: Faces of the mesh
    """

    
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

    # OPTIMIZED LC
    surf.add_vector_quantity("l", l.reshape(-1, 3), defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.1, 0.0, 0.0))

    surf.add_scalar_quantity("r_uv", r_uv_surf.flatten(), defined_on="vertices", enabled=True)

    # ANGLES WITH NORMAL SCALAR FIELD
    surf.add_scalar_quantity("Angles", ang_normal.flatten(), defined_on="vertices", enabled=True)

    surf.add_scalar_quantity("Torsal_Angles", torsal_angles, defined_on="faces", enabled=True)

    surf.add_scalar_quantity("Planarity", planarity_opt, defined_on="faces", enabled=True)

    torsal_dir_show(barycenters, t1, t2, size=size_torsal, rad=0.004)

    V_R = V + r_uv_surf.flatten()[:,None]*n.reshape(-1,3)

    ps.register_surface_mesh("C_uv", V_R, F)


def flip(l, n ):
    l = np.sign(np.sum(l*n, axis=2))[:,:,None]*l
    return l