class Graph(object):
    """Base class for Graph.

    This class defines the structure or graph of the system."""

    def __init__(self):
        self.adj_list = None

    def set_adj_list(self, adj_list):
        self.adj_list = adj_list

