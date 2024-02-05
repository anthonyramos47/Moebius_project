import numpy as np


# -----------------------------------------------------------------------------
#                             Local connectivity
# -----------------------------------------------------------------------------


def edge_halfedge(halfedges, edge_index):
    H = halfedges
    h = np.where(H[:, 5] == edge_index)[0][0]
    return h


def vertex_halfedge(halfedges, vertex_index):
    H = halfedges
    v = np.where(H[:, 0] == vertex_index)[0][0]
    return v


def halfedge_ring(halfedges, halfedge_index):
    H = halfedges
    h0 = halfedge_index
    ring = [h0]
    h = H[H[h0, 3], 4]
    while h != h0:
        ring.append(h)
        h = H[H[h, 3], 4]
    return ring


def vertex_ring_vertices(halfedges, vertex_index):
    h = vertex_halfedge(halfedges, vertex_index)
    ring = halfedge_ring_vertices(halfedges, h)
    return ring


def vertex_ring_faces(halfedges, vertex_index):
    h = vertex_halfedge(halfedges, vertex_index)
    ring = halfedge_ring_faces(halfedges, h)
    return ring


def halfedge_ring_vertices(halfedges, halfedge_index):
    H = halfedges
    ring = halfedge_ring(halfedges, halfedge_index)
    vertices = H[H[ring, 2], 0]
    return vertices


def halfedge_ring_faces(halfedges, halfedge_index):
    H = halfedges
    ring = halfedge_ring(halfedges, halfedge_index)
    faces = H[H[ring, 2], 1]
    return faces

