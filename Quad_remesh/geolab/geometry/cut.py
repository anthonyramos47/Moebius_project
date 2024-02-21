import numpy as np

# -----------------------------------------------------------------------------

from geolab.utilities.arrayutilities import repeated_range

from geolab.mesh.globalconnectivity import edges, faces_vertices, faces_edges, faces_halfedges

from geolab.mesh.counting import number_of_vertices

from geolab.mesh.subdivision import delete_faces

# -----------------------------------------------------------------------------


def cut_mesh_with_plane(vertices, halfedges, plane_normal, plane_origin):
    V = number_of_vertices(halfedges)
    e = edges(halfedges)
    points = np.copy(vertices)
    for i in range(3):
        points[:, i] -= plane_origin[i]
    distance = np.dot(points, plane_normal)
    out_vertices = np.where(distance > 0)[0]
    f1, vj = faces_vertices(halfedges)
    f, ej = faces_edges(halfedges)
    cut_edges = np.where(np.sign(distance[e[:, 0]] * distance[e[:, 1]]) < 0)[0]
    indices = np.full(len(e), np.nan)
    indices[cut_edges] = np.arange(len(cut_edges)) + V
    divide = distance[e[:, 0]] - distance[e[:, 1]]
    divide[divide == 0] = 1e-20
    a = distance[e[:, 0]] / divide
    b = -distance[e[:, 1]] / divide
    int_points = np.einsum('i,ij->ij', a, vertices[e[:, 1]]) + np.einsum('i,ij->ij', b, vertices[e[:, 0]])
    new_faces = np.zeros(2*len(f))
    new_face_indices = np.zeros(2*len(f))
    new_faces[::2] = vj
    new_face_indices[::2] = f
    new_faces[1::2] = indices[ej]
    new_face_indices[1::2] = f
    delete_1 = np.where(np.isnan(new_faces))[0]
    vj1 = np.delete(new_faces, delete_1)
    f1 = np.delete(new_face_indices, delete_1)
    delete_2 = np.where(np.in1d(vj1, out_vertices))[0]
    vj2 = np.delete(vj1, delete_2)
    f2 = np.delete(f1, delete_2)
    f2 = np.array(f2, dtype='i')
    vj2 = np.array(vj2, dtype='i')
    new_vertices = np.vstack((vertices, int_points[cut_edges]))
    proper_faces = np.copy(f2)
    for i in range(2):
        _, unique = np.unique(proper_faces, return_index=True)
        proper_faces = np.delete(proper_faces, unique)
    mask = np.in1d(f2, proper_faces)
    f2 = f2[mask]
    vj2 = vj2[mask]
    split = np.nonzero(f2[1:] - f2[:-1])[0] + 1
    f_list = np.split(vj2, split)
    return new_vertices, f_list


if __name__ == '__main__':
    from geolab.mesh.meshprimitives import mesh_sphere
    from geolab.mesh.halfedges import halfedges
    from geolab.plot.viewer import plotter
    from geolab.mesh.subdivision import clean_mesh

    VS0, HS0 = mesh_sphere(radius=1, around_faces=800, vertical_faces=400)
    VS0 = VS0[:, [0, 2, 1]]
    VS0, FS0 = cut_mesh_with_plane(VS0, HS0, (0, 0, -1), (0, 0, 0.97))
    HS0 = halfedges(FS0)
    VS0, HS0 = clean_mesh(VS0, HS0)


    V0, H0 = mesh_sphere(200,200)
    #H0 = halfedges(F0)
    V, F = cut_mesh_with_plane(V0, H0, (1.1, 0.1, 0), (-0.5, 0.1, 0))
    H = halfedges(F)
    V, H = clean_mesh(V, H)
    V, F = cut_mesh_with_plane(V, H, (0.1, 1.04, 0), (0.1, 0.1, 0))
    H = halfedges(F)
    V, H = clean_mesh(V, H)
    V, F = cut_mesh_with_plane(V, H, (0.1, 1.04, 0.5), (0.1, 0.1, -0.6))
    H = halfedges(F)
    V, H = clean_mesh(V, H)



    plotter = plotter()

    #plotter.plot_points(VS0, radius=0.01)
    plotter.plot_faces(VS0, HS0, smooth=False)
    plotter.show()

