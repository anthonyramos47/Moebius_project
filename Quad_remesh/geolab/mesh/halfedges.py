import copy

import numpy as np

from scipy.sparse import coo_matrix

__author__ = 'Davide Pellis'

"""
Makes halfedges
"""


def are_halfedges(vertices, connectivity):
    return np.max(connectivity) > len(vertices)


def halfedges(faces):
    """Makes halfedges array

    Parameters
    ----------
    faces : list or np.array
        The faces of a manifold and orientable mesh

    Returns
    -------
    h : np.array (H,6)
        The array of halfedges. The i-th halfedge is stored in the i-th row.
        The columns correspond to:
            - 0: index of origin vertex
            - 1: index of adjacent face (-1 is boundary)
            - 2: index of next halfedge
            - 3: index of previous halfedge
            - 4: index of opposite halfedge
            - 5: index of edge

    """
    try:
        H = _make_halfedges(faces)
    except:
        try:
            oriented_faces = orient_faces(faces)
            _make_halfedges(oriented_faces)
            print('*** Mesh reoriented ***')
        except:
            raise ValueError('Half-Edge mesh creation failed. '
                             'Probably the mesh is non manifold '
                             'or non orientable')
    return H.astype('i')


def _make_halfedges(faces):
    h = _make_vectorized_halfedges(faces)
    if h is None:
        orig = []
        face = []
        nexx = []
        prev = []
        twin_i = []
        twin_j = []
        H = 0
        for f in range(len(faces)):
            N = len(faces[f])
            orig.append(faces[f][0])
            face.append(f)
            nexx.append(H + 1)
            prev.append(H + N - 1)
            twin_i.append(faces[f][1])
            twin_j.append(faces[f][0])
            for v in range(1, N - 1):
                orig.append(faces[f][v])
                face.append(f)
                nexx.append(H + v + 1)
                prev.append(H + v - 1)
                twin_i.append(faces[f][v + 1])
                twin_j.append(faces[f][v])
            orig.append(faces[f][N - 1])
            face.append(f)
            nexx.append(H)
            prev.append(H + N - 2)
            twin_i.append(faces[f][0])
            twin_j.append(faces[f][N - 1])
            H += N
        h = np.zeros((H, 6), 'i')
        h[:, 0] = orig
        h[:, 1] = face
        h[:, 2] = nexx
        h[:, 3] = prev
        h[:, 4] = - 1
        V = np.max(orig) + 1
        twin = coo_matrix((np.arange(H) + 1, (twin_i, twin_j)), shape=(V, V))
        twin = twin.tocsc()
        h[:, 4] = twin[h[:, 0], h[h[:, 2], 0]] - 1
    H = len(h)
    V = np.max(h[:, 0]) + 1
    b = np.where(h[:, 4] == -1)[0]
    boundary = h[b, :]
    if len(boundary) > 0:
        boundary[:, 0] = h[h[b, 2], 0]
        boundary[:, 1] = -1
        boundary[:, 4] = b
        B = boundary.shape[0]
        Bh = np.arange(H, H + B)
        h[b, 4] = Bh
        p = np.zeros(V)
        p[h[b, 0]] = Bh
        boundary[:, 3] = p[boundary[:, 0]]
        p = np.zeros(V)
        p[boundary[boundary[:, 3] - H, 0]] = Bh
        boundary[:, 2] = p[boundary[:, 0]]
        h = np.vstack((h, boundary))
    K = h[:, (3, 4)]
    K[:, 0] = np.arange(h.shape[0])
    m = np.amin(K, axis=1)
    u = np.unique(m)
    imap = np.arange(np.max(u) + 1, dtype=np.int)
    imap[u] = np.arange(u.shape[0])
    h[:, 5] = imap[m]
    return h


def _make_vectorized_halfedges(faces):
    try:
        N = faces.shape[1]
    except IndexError:
        return None
    except AttributeError:
        return None
    face = np.repeat(np.arange(len(faces)), N)
    index = (np.arange(N * len(faces))).reshape(len(faces), N).astype(int)
    prev = np.roll(index, 1, axis=1).flatten()
    nexx = np.roll(index, -1, axis=1).flatten()
    orig = faces.flatten()
    next_orig = np.roll(faces, -1, axis=1).flatten()
    V = np.max(orig) + 1
    T = coo_matrix((index.flatten() + 1, (orig, next_orig)), shape=(V, V))
    T = T.tocsc()
    twin = np.array(T[next_orig, orig]).flatten() - 1
    edge = -np.ones(len(orig)).astype(int)
    h = np.column_stack((orig, face, nexx, prev, twin, edge))
    return h


def orient_faces(faces):
    """Orients consistently the faces of a mesh counterclockwise

    Parameters
    ----------
    faces : list or np.array
        The faces of a manifold and orientable mesh

    Returns
    -------
    list
        Faces consistently oriented counterclockwise

    """
    V = np.max(np.max(np.array(faces, dtype=object))) + 1
    F = len(faces)
    f_map = -np.ones((V, V), dtype='i')
    inconsistent = np.zeros((V, V), dtype='i')
    flipped = np.zeros(F, dtype=bool)
    oriented = np.zeros(F, dtype=bool)
    oriented_faces = copy.deepcopy(faces)
    for f in range(F):
        face = faces[f]
        for j in range(len(face)):
            v0 = face[j - 1]
            v1 = face[j]
            if f_map[v0, v1] == -1:
                f_map[v0, v1] = f
            else:
                f_map[v1, v0] = f
                inconsistent[v0, v1] = True
                inconsistent[v1, v0] = True
    ring = [0]
    oriented[0] = True
    i = 1
    while len(ring) > 0:
        next_ring = []
        for f in ring:
            face = faces[f]
            for j in range(len(face)):
                flip = False
                v0 = face[j - 1]
                v1 = face[j]
                if f_map[v0, v1] == f:
                    v2 = v1
                    v3 = v0
                else:
                    v2 = v0
                    v3 = v1
                if inconsistent[v2, v3] and not flipped[f]:
                    flip = True
                if not inconsistent[v2, v3] and flipped[f]:
                    flip = True
                fi = f_map[v2, v3]
                if fi != -1 and not oriented[fi]:
                    if fi not in next_ring:
                        next_ring.append(fi)
                    if flip:
                        oriented_faces[fi].reverse()
                        flipped[fi] = True
                    i += 1
                    oriented[fi] = True
                    if i == F:
                        return oriented_faces
        ring = next_ring
        if len(ring) == 0:
            try:
                ring = [np.where(oriented is False)[0][0]]
            except:
                return
