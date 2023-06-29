import unittest

from geometry.mesh import Mesh
from random import random
import numpy as np

class TestMesh(unittest.TestCase):

    def test_mesh_dimensions(self):
        """ Test that the mesh dimensions are correct during the creations
        """
        tmesh = Mesh() 
        vertices = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        faces = [[0,1,2], [2,1,3]]
        tmesh.make_mesh(vertices, faces)
        self.assertEqual(tmesh.V, 4, "Problem with V")
        self.assertEqual(tmesh.F, 2, "Problem with F")
        self.assertEqual(tmesh.E, 5, "Problem with E")
        self.assertEqual(tmesh.vertices.shape, (4,3), "Vertices shape is wrong") 
        self.assertEqual(tmesh.H.shape, (10,6), "Problem with H shape")


    def test_mesh_half_edge(self):
        """ Test that the half edge structure is correct
        """
        tmesh = Mesh() 
        vertices = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        faces = [[0,1,2], [2,1,3]]
        tmesh.make_mesh(vertices, faces)
        self.assertCountEqual(tmesh.H[0], [0,6,0,1,2,0], 'Problem Half edge 0')
    
    def test_vertex_star(self):
        """
        Test that the vertex star is correct
        """
        tmesh2 = Mesh()
        vstar = [[0,0,0], [1,0,0], [1,1,0], [-1,1,0], [-1, 0.5,0], [-1,-1,0] ]
        fstar = [[0,1,2],[0,2,3],[0,3,4],[0,4,5],[0,5,1]]
        tmesh2.make_mesh(vstar, fstar)
        star = tmesh2.vertex_star(0)
        self.assertCountEqual(star, [1,2,3,4,5], "Problem with vertex star")
        star2 = tmesh2.vertex_star(2)
        self.assertCountEqual(star2, [0,1,3], "Problem with vertex star 2")

    def test_face_operations(self):
        tmesh = Mesh()
        vertices = [[-1,0,0], [1,0,0], [0,1,0], [1,1,0], [-1,-1,0], [0,-1,0] ]
        faces = [[0,1,2],[2,1,3],[0,2,4],[0,4,5]]
        tmesh.make_mesh(vertices, faces)
        # Face ring of a face
        ring = tmesh.face_ring(0)
        self.assertCountEqual(ring, [1,2], "Problem with face ring")
        self.assertCountEqual(tmesh.faces()[0], [0, 1, 2], "Problem with face 0")
        self.assertCountEqual(tmesh.faces()[1], [2, 1, 3], "Problem with face 1")
        self.assertCountEqual(tmesh.faces()[2], [0, 2, 4], "Problem with face 2")
        self.assertCountEqual(tmesh.faces()[3], [0, 4, 5], "Problem with face 3")
        # Face adjacency list
        print(tmesh.face_face_adjacency_list())
        self.assertCountEqual(tmesh.face_face_adjacency_list()[0], [1,2], "Problem with face adjacency list 0")
        self.assertCountEqual(tmesh.face_face_adjacency_list()[1], [0], "Problem with face adjacency list 1")
        self.assertCountEqual(tmesh.face_face_adjacency_list()[2], [0,3], "Problem with face adjacency list 2")
        self.assertCountEqual(tmesh.face_face_adjacency_list()[3], [2], "Problem with face adjacency list 3")

        
    def test_vertex_operations(self):
        tmesh = Mesh()
        vertices = [[-1,0,0], [1,0,0], [0,1,0], [1,1,0], [-1,-1,0], [0,-1,0] ]
        faces = [[0,1,2],[2,1,3],[0,2,4],[0,1,5]]
        tmesh.make_mesh(vertices, faces)
        
        # Vertex ring
        self.assertCountEqual(tmesh.vertex_adjacency_list()[0], [1,2,4,5], "Problem with vertex adjacency list")
        self.assertCountEqual(tmesh.vertex_adjacency_list()[1], [0,2,3,5], "Problem with vertex adjacency list")
        self.assertCountEqual(tmesh.vertex_adjacency_list()[2], [0,1,3,4], "Problem with vertex adjacency list")
        self.assertCountEqual(tmesh.vertex_adjacency_list()[3], [1,2], "Problem with vertex adjacency list")
        self.assertCountEqual(tmesh.vertex_adjacency_list()[4], [0,2], "Problem with vertex adjacency list")
        self.assertCountEqual(tmesh.vertex_adjacency_list()[5], [0,1], "Problem with vertex adjacency list")

        # Face ring of a vertex
        self.assertCountEqual(tmesh.vertex_face_adjacency_list()[0], [0,2,3], "Problem with vertex face ring")
        self.assertCountEqual(tmesh.vertex_face_adjacency_list()[1], [0,1,3], "Problem with vertex face ring")  
        self.assertCountEqual(tmesh.vertex_face_adjacency_list()[2], [0,1,2], "Problem with vertex face ring")
        self.assertCountEqual(tmesh.vertex_face_adjacency_list()[3], [1], "Problem with vertex face ring")
        self.assertCountEqual(tmesh.vertex_face_adjacency_list()[4], [2], "Problem with vertex face ring")
        self.assertCountEqual(tmesh.vertex_face_adjacency_list()[5], [3], "Problem with vertex face ring")

    def test_boundary_functions(self):
        tmesh = Mesh()
        vertices = [[-1,0,0], [1,0,0], [0,1,0], [1,1,0], [-1,-1,0], [0,-1,0] ]
        faces = [[0,1,2],[2,1,3],[0,2,4],[0,1,5]]
        tmesh.make_mesh(vertices, faces)

        # Boundary vertices
        self.assertCountEqual(tmesh.boundary_vertices(), [0,1,2,3,4,5], "Problem with boundary vertices Mesh 1")

        # Mesh 2
        tmesh = Mesh()
        vertices = [random() for i in range(14)]
        faces = [[0,1,2],[1,3,2],[3,4,2],[2,4,5],[4,6,7],[5,4,7],[5,7,9],[5,9,10],[5,10,8],[5,8,2],[10,11,8],[8,11,12],[8,12,0],[8,0,2]]
        tmesh.make_mesh(vertices, faces)
        
        # Boundary vertices
        self.assertCountEqual(tmesh.boundary_vertices(), [0,1,3,4,6,7,9,10,11,12], "Problem with boundary vertices Mesh 2")
        
        # Boundary Faces
        self.assertCountEqual(tmesh.boundary_faces(), [ 0, 1, 2, 4, 6, 7, 10, 11, 12], "Problem with boundary Faces Mesh 2")

        print(tmesh.inner_vertices())
        # Innter vertices
        self.assertCountEqual(tmesh.inner_vertices(), [2,5,8], "Problem with inner vertices Mesh 2")
  
        # Inner Faces
        self.assertCountEqual(tmesh.inner_faces(), [3,5,8,9,13], "Problem with inner Faces Mesh 2")

    

if __name__ == '__main__':
    unittest.main()