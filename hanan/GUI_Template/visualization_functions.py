import polyscope as ps
import numpy as np


def visualization_init():

    # Generate 7 random points
    pts = np.random.rand(7, 3)

    # Mesh the points
    f = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

    # Add the mesh to the visualization

    mesh = ps.register_surface_mesh("Mesh", pts, f)

    # Add a scalar quantity to the mesh
    scalar_quantity = np.random.rand(7)

    mesh.add_scalar_quantity("scalar", scalar_quantity)

    # Add a vector quantity to the mesh
    vector_quantity = np.random.rand(7, 3)

    mesh.add_vector_quantity("vector", vector_quantity)



def visualization():

    # Generate 7 random points
    pts = np.random.rand(7, 3)

    # Mesh the points
    f = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

    # Add the mesh to the visualization
    mesh = ps.register_surface_mesh("Mesh", pts, f)

    # Add a scalar quantity to the mesh
    scalar_quantity = np.random.rand(7)

    mesh.add_scalar_quantity("scalar", scalar_quantity)

    # Add a vector quantity to the mesh
    vector_quantity = np.random.rand(7, 3)

    mesh.add_vector_quantity("vector", vector_quantity)

    id = np.where(scalar_quantity > 0.5)[0]

    new = np.zeros(len(scalar_quantity))
    new[id] = 1

    mesh.add_scalar_quantity("new", new)
