import sympy as sp
import numpy as np
import networkx as nx
import random
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from typing import Any


class GraphWalk:
    """analysis of walks on undirected graph"""

    def __init__(
        self,
        graph: list[tuple[int, int]] | None = None,
        adj_mat: list[list[int]] | None = None,
    ):
        self.expansion_var = sp.symbols('z')
        self._adj_mat: NDArray[Any] | None = None
        self._graph = None

        if graph is not None:
            self.graph = graph
        elif adj_mat is not None:
            self.adj_mat = np.array(adj_mat)

        self.det_ch_poly: sp.Expr | None = None
        self.poly_ratio: sp.Expr | None = None
        self.ch_poly_matrix: sp.Matrix | None = None
        self.sp_mat: sp.Matrix | None = None
        if self._adj_mat is not None:
            self.sp_mat = sp.Matrix(self._adj_mat)

    @property
    def adj_mat(self) -> NDArray[Any] | None:
        return self._adj_mat

    @adj_mat.setter
    def adj_mat(self, value: NDArray[Any] | list[list[int]]) -> None:
        if value is not None:
            self._adj_mat = (
                np.array(value) if not isinstance(value, np.ndarray) else value
            )
            if not hasattr(self._adj_mat, 'shape'):
                self._adj_mat = np.array(self._adj_mat)
            self.adj_mat2ugraph()
            self.compute_char_poly()

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, value):
        if (
            isinstance(value, np.ndarray) and value.size == 0
        ):  # Check for empty NumPy array
            self._graph = []
        elif not value:  # Check for an empty list
            self._graph = []
        else:
            self._graph = value
            self.ugraph2adj_mat()
            self.compute_char_poly()

    def ugraph2adj_mat(self) -> None:
        """Converts an undirected graph represented as a
        list of edges (1-based) to an adjacency matrix (1-based)."""
        vertices = set()
        if self._graph is None or len(self._graph) == 0:
            self._adj_mat = np.array([])
            return

        for u, v in self._graph:
            vertices.add(u)
            vertices.add(v)
        num_vertices = max(vertices)
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for u, v in self._graph:
            adj_matrix[u - 1, v - 1] += 1
            adj_matrix[v - 1, u - 1] += 1
        print(f'The adjacency matrix:\n{adj_matrix}\n')
        self._adj_mat = adj_matrix

    def adj_mat2ugraph(self) -> None:
        """Converts an adjacency matrix (1-based) to an undirected graph
        represented as a list of edges."""
        if self._adj_mat is None:
            return

        num_vertices = self._adj_mat.shape[0]
        edges = [
            (i + 1, j + 1)
            for i in range(num_vertices)
            for j in range(i + 1, num_vertices)
            for _ in range(self._adj_mat[i, j])
            if self._adj_mat[i, j] > 0
        ]
        print(f'The edge list:\n{edges}\n')
        self._graph = edges

    def compute_n_step_walk(self, n: int) -> np.ndarray:
        """Compute the adjacency matrix giving the number of n step walks"""
        result = np.round(self.sym_matrix_pow(n), 0)
        print(f'The matrix giving the number of {n} step walks:\n{result}\n')
        return result

    def compute_generating_function(self, i: int, j: int) -> sp.Expr | None:
        """return the generating function for walks from vertex i to vertex j"""
        self.poly_ratio = None
        if self.adj_mat is None:
            return
        n: int = len(self.adj_mat)
        if max(i, j) > n or min(i, j) < 1:
            print(
                f'Index error!\n'
                f'The condition: max(i, j) <= {n} '
                f'and min(i, j) >= 1 is not satisfied\n'
            )
            return None

        self.compute_char_poly()

        if self.det_ch_poly is None:
            print('Characteristic polynomial error!')
            return None

        sign = -1 if (i + j) % 2 else 1
        nom = sign * self.compute_det_mat_ij(i, j)
        try:
            self.poly_ratio = nom / self.det_ch_poly
            self.poly_ratio = sp.simplify(self.poly_ratio, rational=True)

            print(f'The generating function for walks from vertex {i} to {j}:')
            self.poly_ratio_display()
            self.taylor_s(self.poly_ratio)
        except (ValueError, TypeError) as e:
            print(f'Error computing generating function: {e}')
            return None

    def poly_ratio_display(self) -> None:
        """display polynomial fraction"""
        x = self.expansion_var
        pr = (
            str(self.poly_ratio)
            .replace(f'{x}**', f'{x}^')
            .replace(f'*{x}', f'{x}')
            .replace(f'{x}*', f'{x}')
        )

        nom, den = pr.split('/')
        den = den.strip('()')
        l_den, l_nom = len(den), len(nom)
        num_spaces = (l_den - l_nom) // 2
        bs_padding = ' ' * num_spaces
        print(f'{bs_padding}{nom}{bs_padding}')
        print(f'{"─" * l_den}\n{den}')

    def minor_matrix(self, i: int, j: int) -> sp.Matrix | None:
        """Compute the minor matrix A_ij by removing the i-th row
        and j-th column from matrix A. Indexing is 1-based.
        """
        if self.ch_poly_matrix is None:
            return None

        rows = list(range(self.ch_poly_matrix.shape[0]))
        cols = list(range(self.ch_poly_matrix.shape[1]))

        if min(i, j) < 1:
            print(f'Error! min(i, j) = {min(i, j)} < 1 => skipping op!\n')
            return None

        i -= 1
        j -= 1
        rows.pop(i)
        cols.pop(j)
        minor_rows = []
        for r in rows:
            minor_row = []
            for c in cols:
                minor_row.append(self.ch_poly_matrix[r, c])
            minor_rows.append(minor_row)
        return sp.Matrix(minor_rows)

    def compute_det_mat_ij(self, i: int, j: int) -> sp.Expr:
        """Compute det(Iij - zAij), where Aij is the minor matrix of A
        obtained by removing the i-th row and j-th column. Indexing is 1-based.
        """
        minor_mat_ij = self.minor_matrix(i, j)
        return sp.det(minor_mat_ij)

    def compute_char_poly(self) -> None:
        """Compute characteristic polynomial of a matrix, det(I - zA)"""
        if self.adj_mat is not None:
            self.sp_mat = sp.Matrix(self.adj_mat)
            n = len(self.adj_mat)
        else:
            print('Adjacency matrix must be updated first')
            return

        eye = sp.eye(n)
        self.ch_poly_matrix = eye - self.expansion_var * self.sp_mat
        self.det_ch_poly = sp.det(self.ch_poly_matrix)

    def sym_matrix_pow(self, n: int) -> np.ndarray:
        """Computes a symmetrical matrix raised to the nth power based on P * D^n * P.T
        The adjacency matrix of every undirected graph is symmetrical, and for its
        eigenvector matrix P, the equality P.T = P^-1 always exists.
        Args:
            n: The power to which the matrix should be raised.
        Returns:
            The matrix raised to the nth power.
        """
        eigenvalues, p = np.linalg.eigh(self.adj_mat)
        d_n = np.diag(eigenvalues**n)
        return p @ d_n @ p.T

    def taylor_s(self, poly: sp.Expr, order: int = 9) -> sp.Expr | None:
        """Computes the Taylor series around z=0 for polynomial P(z)"""
        try:
            tseries = sp.series(poly, self.expansion_var, 0, order + 1, '-')
            self.taylor_s_display(tseries)
            return tseries
        except Exception as e:
            print(f'Error computing Taylor series: {e}')
            return None

    def taylor_s_display(self, taylor_series: sp.Expr) -> None:
        tseries_formatted = (
            str(taylor_series)
            .replace('**', '^')
            .replace(f'*{self.expansion_var}', f'{self.expansion_var}')
        )
        print(f'\nTaylor expansion:\n{tseries_formatted}\n')

    def plot_graph(
        self,
        filename: str = 'graph_examples/graph_networkx.png',
        show: bool = True,
        save: bool = True,
    ) -> None:
        """Plots the graph using NetworkX MultiGraph and Matplotlib,
        visualizing multiple edges with curvature."""

        p_graph = nx.MultiGraph()

        if self.graph is None:
            return

        for u, v in self.graph:
            p_graph.add_edge(u, v)

        pos = nx.spring_layout(p_graph)

        nx.draw_networkx_nodes(p_graph, pos, node_color='skyblue', node_size=800)
        nx.draw_networkx_labels(p_graph, pos)

        # Visualize multiple edges with curvature
        for u, v, k in p_graph.edges(keys=True):
            if p_graph.number_of_edges(u, v) > 1:
                curvature = 0.4 * (k - 0.5 * (p_graph.number_of_edges(u, v) - 1))
            else:
                curvature = 0

            nx.draw_networkx_edges(
                p_graph,
                pos,
                edgelist=[(u, v)],
                edge_color='gray',
                connectionstyle=f'arc3,rad={curvature}',
            )
        if save:
            plt.savefig(filename)
        if show:
            plt.show()

        plt.close()

    def generate_rnd_graph(
        self, num_edges: int, num_nodes: int, double_edges: bool = False
    ) -> None:
        """Generates a random connected undirected graph with performance optimizations.
        :param num_edges: Number of edges to generate.
        :param num_nodes: Number of nodes in the graph (1-based indexing).
        :param double_edges: If False, prevents duplicate edges.
        :return: A list of tuples representing the edges.
        """
        if num_nodes <= 0:
            self.graph = []
            return None

        assert num_edges >= num_nodes - 1, (
            f'Error! num_edges must be >= {num_nodes - 1} for connectivity'
        )

        if not double_edges:
            max_edges = num_nodes * (num_nodes - 1) // 2
            assert num_edges <= max_edges, f'Error! num_edges must be <= {max_edges}'
            edges = set()
        else:
            edges = []

        # ensure connectivity of all nodes
        spanning_tree_edges = self._generate_spanning_tree(num_nodes)
        if double_edges:
            edges.extend(spanning_tree_edges)
        else:
            edges.update(spanning_tree_edges)

        remaining_edges = num_edges - len(edges)
        if remaining_edges > 0:
            nodes_list = list(range(1, num_nodes + 1))
            if double_edges:
                for _ in range(remaining_edges):
                    u, v = random.choice(nodes_list), random.choice(nodes_list)
                    while u == v:
                        u, v = random.choice(nodes_list), random.choice(nodes_list)
                    edges.append(tuple(sorted((u, v))))
            else:
                while len(edges) < num_edges:
                    u, v = random.choice(nodes_list), random.choice(nodes_list)
                    if u != v:
                        edge = tuple(sorted((u, v)))
                        if edge not in edges:
                            edges.add(edge)

        self.graph = list(edges)
        print(f'Randomized connected graph:\n{sorted(self.graph)}\n')

    @staticmethod
    def _generate_spanning_tree(num_nodes: int) -> set:
        """Generates a spanning tree using a simple random approach."""
        nodes = list(range(1, num_nodes + 1))
        random.shuffle(nodes)
        spanning_tree = set()
        for i in range(num_nodes - 1):
            spanning_tree.add(tuple(sorted((nodes[i], nodes[i + 1]))))
        return spanning_tree
