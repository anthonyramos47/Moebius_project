import unittest

from geometry.mesh import Mesh

tmesh = Mesh()

vertices = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
faces = [[0,1,2], [2,1,3]]




class TestMesh(unittest.TestCase):

    def test_mesh_dimensions(self):
        tmesh.make_mesh(vertices, faces)
        print("   O  T  F  N  P  E")
        print(tmesh.H)
        self.assertEqual(tmesh.V, 4, "Problem with V")
        self.assertEqual(tmesh.F, 2, "Problem with F")
        self.assertEqual(tmesh.E, 5, "Problem with E")
        self.assertEqual(tmesh.vertices.shape, (4,3), "Vertices shape is wrong") 
        self.assertEqual(tmesh.H.shape, (10,6), "Problem with H shape")
        self.assertCountEqual(tmesh.H[0], [0,6,0,1,2,0], 'Problem Half edge 0')
        
if __name__ == '__main__':
    unittest.main()