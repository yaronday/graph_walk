import numpy as np
from graph_walk import GraphWalk


def run():
    # Example 1: usage with a graph, represented by list of edges
    # Good Will Hunting, question 1
    graph = [(1, 4), (1, 2), (2, 3), (2, 3), (2, 4)]
    gw = GraphWalk(graph)

    gw.plot_graph('ex1_graph_input.png', show=False)
    gw.compute_n_step_walk(3)
    gw.compute_generating_function(1, 3)

    # Example 2: usage with an adjacency matrix
    adj_mat = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    gw.adj_mat = np.array(adj_mat)
    gw.plot_graph('ex2_matrix_input.png', show=False)
    gw.compute_n_step_walk(4)
    gw.compute_generating_function(2, 3)

    # Example 3: randomizing an undirected graph
    gw.generate_rnd_graph(11, 6, double_edges=True)
    gw.plot_graph('ex3_randomized_graph.jpg', show=False)
    gw.compute_n_step_walk(5)
    gw.compute_generating_function(3, 6)


if __name__ == '__main__':
    run()
