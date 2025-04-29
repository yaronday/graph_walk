import unittest
import sympy as sp
import numpy as np
from graph_walk import GraphWalk
import os


class TestGraphWalk(unittest.TestCase):
    def setUp(self):
        self.graph1 = [(1, 2), (1, 3), (2, 3)]
        self.adj_mat1 = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        self.graph2 = [(1, 2), (1, 4), (2, 3), (3, 4)]
        self.adj_mat2 = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
        self.graph3 = [(1, 2), (1, 2), (2, 3)]
        self.adj_mat3 = [[0, 2, 0], [2, 0, 1], [0, 1, 0]]
        self.graph4 = []
        self.adj_mat4 = []
        self.ind_var = sp.symbols('z')  # the independent variable or expansion arg

    def test_ugraph2adj_mat(self):
        graph_walk = GraphWalk(graph=self.graph1)
        np.testing.assert_array_equal(graph_walk.adj_mat, np.array(self.adj_mat1))

        graph_walk = GraphWalk(graph=self.graph2)
        np.testing.assert_array_equal(graph_walk.adj_mat, np.array(self.adj_mat2))

        graph_walk = GraphWalk(graph=self.graph3)
        np.testing.assert_array_equal(graph_walk.adj_mat, np.array(self.adj_mat3))

    def test_adj_mat2ugraph(self):
        graph_walk = GraphWalk(adj_mat=self.adj_mat1)
        self.assertEqual(sorted(graph_walk.graph), sorted(self.graph1))

        graph_walk = GraphWalk(adj_mat=self.adj_mat2)
        self.assertEqual(sorted(graph_walk.graph), sorted(self.graph2))

        graph_walk = GraphWalk(adj_mat=self.adj_mat3)
        self.assertEqual(sorted(graph_walk.graph), sorted(self.graph3))

    def test_compute_n_step_walk(self):
        graph_walk = GraphWalk(adj_mat=self.adj_mat1)
        n_walk_2 = graph_walk.compute_n_step_walk(2)
        expected_2 = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        np.testing.assert_array_equal(n_walk_2, expected_2)

        n_walk_3 = graph_walk.compute_n_step_walk(3)
        expected_3 = np.array([[2, 3, 3], [3, 2, 3], [3, 3, 2]])
        np.testing.assert_array_equal(n_walk_3, expected_3)

    def test_compute_generating_function(self):
        graph_walk = GraphWalk(adj_mat=self.adj_mat1)
        graph_walk.compute_generating_function(1, 1)
        self.assertIsNotNone(graph_walk.poly_ratio)

        graph_walk.compute_generating_function(1, 2)
        self.assertIsNotNone(graph_walk.poly_ratio)

        graph_walk.compute_generating_function(1, 5)
        self.assertIsNone(graph_walk.poly_ratio)

    def test_minor_matrix(self):
        graph_walk = GraphWalk(adj_mat=self.adj_mat1)
        graph_walk.compute_char_poly()

        a_ij = graph_walk.minor_matrix(1, 2)
        a_ji = graph_walk.minor_matrix(2, 1)
        expected_minor = sp.Matrix([[-self.ind_var, -self.ind_var], [-self.ind_var, 1]])
        self.assertEqual(a_ij, expected_minor)
        self.assertEqual(a_ji, expected_minor)

    def test_minor_matrix_w_empty_ch_poly(self):
        graph_walk = GraphWalk(adj_mat=self.adj_mat1)
        minor_mat = graph_walk.minor_matrix(1, 1)
        self.assertIsNone(minor_mat)

    def test_compute_det_mat_ij(self):
        graph_walk = GraphWalk(adj_mat=self.adj_mat1)
        graph_walk.compute_char_poly()
        det_mat_ij = graph_walk.compute_det_mat_ij(1, 1)
        expected_det = 1 - self.ind_var**2
        self.assertEqual(det_mat_ij, expected_det)

    def test_compute_char_poly(self):
        graph_walk = GraphWalk(adj_mat=self.adj_mat1)
        graph_walk.compute_char_poly()
        print(f'ch poly = {graph_walk.ch_poly_matrix}')
        print(f'det = {graph_walk.det_ch_poly}')

        expected_det = -2 * self.ind_var**3 - 3 * self.ind_var**2 + 1
        self.assertEqual(graph_walk.det_ch_poly, expected_det)

    def test_sym_matrix_pow(self):
        graph_walk = GraphWalk(adj_mat=self.adj_mat1)
        pow_mat = graph_walk.sym_matrix_pow(2)
        expected_pow = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
        np.testing.assert_array_almost_equal(pow_mat, expected_pow)

    def test_taylor_s(self):
        graph_walk = GraphWalk(adj_mat=self.adj_mat1)
        graph_walk.compute_generating_function(1, 1)
        taylor = graph_walk.taylor_s(graph_walk.poly_ratio)
        self.assertIsNotNone(taylor)

    def test_plot_graph(self):
        graph_walk = GraphWalk(graph=self.graph1)
        filename = 'test_graph.png'
        graph_walk.plot_graph(filename=filename, show=False, save=True)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_generate_rnd_graph(self):
        graph_walk = GraphWalk()
        graph_walk.generate_rnd_graph(num_edges=5, num_nodes=4)
        self.assertEqual(len(graph_walk.graph), 5)
        self.assertTrue(len(set(sum(graph_walk.graph, ()))) <= 4)

        graph_walk.generate_rnd_graph(num_edges=6, num_nodes=4, double_edges=True)
        self.assertEqual(len(graph_walk.graph), 6)
        self.assertTrue(len(set(sum(graph_walk.graph, ()))) <= 4)

        graph_walk.generate_rnd_graph(num_edges=0, num_nodes=0)
        self.assertEqual(len(graph_walk.graph), 0)

        with self.assertRaises(AssertionError):
            graph_walk.generate_rnd_graph(num_edges=1, num_nodes=3)
        with self.assertRaises(AssertionError):
            graph_walk.generate_rnd_graph(num_edges=5, num_nodes=3, double_edges=False)

    def test_graph_sync(self):
        graph_walk = GraphWalk()
        graph_walk.adj_mat = self.adj_mat1
        zero_matrix = np.zeros((2, 2), dtype=int)
        graph_walk.adj_mat = zero_matrix
        np.testing.assert_array_equal(graph_walk.adj_mat, zero_matrix)
        self.assertEqual(graph_walk.graph, [])

    def test_adj_mat_sync(self):
        graph_walk = GraphWalk(self.graph1)
        graph_walk.graph = self.graph2
        np.testing.assert_array_equal(graph_walk.adj_mat, np.array(self.adj_mat2))


if __name__ == '__main__':
    unittest.main()
