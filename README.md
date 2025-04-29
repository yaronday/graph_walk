# Graph Walk Analysis

This project provides tools for analyzing walks on undirected graphs using SymPy, 
NumPy, NetworkX, and Matplotlib. It includes functionalities to:

- Convert an undirected graph (edge list) to an adjacency matrix and vice versa thanks to 1:1 matching property.
- Compute the number of n-step walks between vertices.
- Calculate the generating function for walks between vertices.
- Compute the characteristic polynomial of the adjacency matrix.
- Perform Taylor series expansion of the generating function.
- Plot the graph using NetworkX and Matplotlib, supporting and visualizing multiple edges.
- Generate a randomized list of edges (graph).

## Requirements

- Python 3.10+
- NumPy
- SymPy
- NetworkX
- Matplotlib

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Character Encoding

This project uses UTF-8 encoding for all text files, including Python source code and data files. 

## Usage

```python
import numpy as np
from src import GraphWalk

# Example 1: usage with a graph, represented by list of edges
# Good Will Hunting, question 1
graph = [(1, 4), (1, 2), (2, 3), (2, 3), (2, 4)]
gw = GraphWalk(graph)

gw.plot_graph("ex1_graph_input.png", show=False)
gw.compute_n_step_walk(3)
gw.compute_generating_function(1, 3)

# Example 2: usage with an adjacency matrix
adj_mat = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
gw.adj_mat = np.array(adj_mat)
gw.plot_graph("ex2_matrix_input.png", show=False)
gw.compute_n_step_walk(4)
gw.compute_generating_function(2, 3)

# Example 3: randomizing an undirected graph
gw.generate_rnd_graph(11, 6, double_edges=True)
gw.plot_graph("ex3_randomized_graph.jpg", show=False)
gw.compute_n_step_walk(5)
gw.compute_generating_function(3, 6)

```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
