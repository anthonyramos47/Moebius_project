import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import json
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
    print("V dir order: ", order_v)

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


def approx_surface_from_data(file,box_size=2):
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
    pts = normalize_vertices(data[2:], 2)

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

    