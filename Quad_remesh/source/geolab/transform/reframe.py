import numpy as np

from geolab.mesh.halfedges import are_halfedges


def rigid_transformation(points, frame):
    d2 = False
    if len(points.shape) == 2:
        points = np.repeat([points], frame.shape[1], axis=0)
        if frame.shape[1] == 1:
            d2 = True
    P = np.einsum('lij,jlk->lik', points, frame[1:, :, :])
    P[:, :, 0] += np.repeat([frame[0, :, 0]], len(P[0]), axis=0).T
    P[:, :, 1] += np.repeat([frame[0, :, 1]], len(P[0]), axis=0).T
    P[:, :, 2] += np.repeat([frame[0, :, 2]], len(P[0]), axis=0).T
    if d2:
        P = P[0]
    return P


def scale(points, x_factor=1, y_factor=1, z_factor=1,):
    pass


def mesh_rigid_transformation(vertices, frame, connectivity=None):
    V = rigid_transformation(vertices, frame)
    if len(V.shape) == 2:
        V = np.array([V])
    N = len(V)
    V = np.reshape(V, (N*len(vertices), 3), order='C')
    if connectivity is not None:
        if are_halfedges(vertices, connectivity):
            # TODO: implement halfedges
            raise NotImplementedError('Halfedges not implemented')
        else:
            F = np.repeat([connectivity], N, axis=0)
            s = np.array([[np.arange(N)]]).T
            s = np.repeat(s, connectivity.shape[1], axis=2) * len(vertices)
            s = np.repeat(s, connectivity.shape[0], axis=1)
            F = np.reshape(F+s, (-1, F.shape[-1]), order='C')
        return V, F
    return V

