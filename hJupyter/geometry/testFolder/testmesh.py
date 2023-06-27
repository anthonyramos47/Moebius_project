import unittest

from geometry.mesh import Mesh

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
        print("   O  T  F  N  P  E")
        print(tmesh.H)
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
        ring = tmesh.face_ring(0)
        self.assertCountEqual(ring, [1,2], "Problem with face ring")
        self.assertCountEqual(tmesh.faces(), [[0, 1, 2], [2, 1, 3], [0, 2, 4], [0, 4, 5]], "Problem with faces")
        
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
        

if __name__ == '__main__':
    unittest.main()