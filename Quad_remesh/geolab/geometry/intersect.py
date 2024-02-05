import numpy as np

# -----------------------------------------------------------------------------

from geolab.utilities.arrayutilities import orthogonal_vector

# -----------------------------------------------------------------------------


def points_plane_projection(points, plane_point=None,
                            plane_normal=np.array([0, 0, 1]),
                            light_direction=np.array([1, 1, 1]),
                            offset=0):
    """
    see:
    https://www.scratchapixel.com/lessons/3d-basic-rendering/
    ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    """
    if plane_point is None:
        plane_point = np.array([0, 0, 0]) - plane_normal * offset
    try:
        e1 = orthogonal_vector(plane_normal[0])
    except:
        e1 = orthogonal_vector(plane_normal)
    e2 = np.cross(plane_normal, e1)
    A = np.tile(plane_point, (len(points), 1))
    O = points
    E_1 = np.tile(e1, (len(points), 1))
    E_2 = np.tile(e2, (len(points), 1))
    D = np.array(light_direction)
    T = O - A
    P = np.cross(D, E_2, axisb=1)
    Q = np.cross(T, E_1, axisb=1)
    f = (np.einsum('ij,ij -> i', P, E_1) + 1e-20) ** -1
    # t = f * np.einsum('ij,ij -> i', Q, E_2)
    u = f * np.einsum('ij,ij -> i', P, T)
    v = f * np.einsum('ij,j -> i', Q, D)
    i = (np.einsum('i,ij -> ij', u, E_1) + np.einsum('i,ij -> ij', v, E_2)) + A
    return i
