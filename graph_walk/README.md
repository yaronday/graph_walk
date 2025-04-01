# GraphWalk Analysis

This project provides tools for analyzing walks on undirected graphs using SymPy, 
NumPy, NetworkX, and Matplotlib. It includes functionalities to:

- Convert an undirected graph (edge list) to an adjacency matrix and vice versa thanks to 1:1 matching property.
- Compute the number of n-step walks between vertices.
- Calculate the generating function for walks between vertices.
- Compute the characteristic polynomial of the adjacency matrix.
- Perform Taylor series expansion of the generating function.
- Plot the graph using NetworkX and Matplotlib, supporting and visualizing multiple edges.

## Character Encoding

This project uses UTF-8 encoding for all text files, including Python source code and data files. 
This ensures proper display of Unicode characters, such as mathematical symbols and box-drawing characters (e.g., "─").

**If you encounter issues displaying characters correctly:**

* Ensure your text editor or terminal is set to use UTF-8 encoding.
* If you're using a terminal, verify that your system's locale settings are configured for UTF-8.

Example of a character that relies on UTF-8:

## Requirements

- Python 3.6+
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

## Usage

```python
import numpy as np
from graph_walk import GraphWalk

# Example usage with an edge list:
graph_edges = [(1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (2, 3)] #undirected connected graph
gw_edge = GraphWalk(graph=graph_edges)
gw_edge.compute_n_step_walk(2)
gw_edge.compute_generating_function(1, 1)
gw_edge.taylor_s(graph_walk_edges.poly_ratio)
gw_edge.plot_graph() #networkx plot

# Example usage with an adjacency matrix:
adj_matrix = np.array([[0, 1, 1, 1], [1, 0, 2, 1], [1, 2, 0, 0], [1, 1, 0, 0]])
gw_mat = GraphWalk(adj_mat=adj_matrix)
gw_mat.compute_n_step_walk(2)
gw_mat.compute_generating_function(1, 1)
gw_mat.taylor_s(gw_mat.poly_ratio)
gw_mat.plot_graph() #networkx plot

## License
This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
