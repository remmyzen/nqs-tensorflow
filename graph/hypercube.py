from graph import Graph

class Hypercube(Graph):
    """This class is used to define a hypercube structure or ordered lattice in n-dimension"""

    def __init__(self, length, dimension, pbc=True):
        """Construct a new squared hypercube

        Args:
            length: The length of the hypercube
            dimension: The dimension of the system
            pbc: True for hypercube with periodic boundary conditions or False for hypercube with open boundary conditions

        TODO: non-squared hypercube
        """
        Graph.__init__(self)
        self.length = length
        self.dimension = dimension
        self.num_points = self.length ** self.dimension
        self.pbc = pbc

        self.adj_list = self._create_adj_list()
        self.num_bonds, self.bonds = self._find_bonds()

    def _create_adj_list(self):
        """Create adjacency list for each point in the hypercube
        """
        adj_list = [[] for i in range(self.num_points)]
        for p in range(self.num_points):
            p_coordinate = self._point_to_coordinate(p)
            for d in range(self.dimension):
                neighbor1 = list(p_coordinate)
                neighbor2 = list(p_coordinate)
                if self.pbc or (not self.pbc and p_coordinate[d] + 1 < self.length):
                    neighbor1[d] = (p_coordinate[d] + 1) % self.length
                    adj_list[p].append(self._coordinate_to_point(neighbor1))
                if self.pbc or (not self.pbc and p_coordinate[d] - 1 >= 0):
                    neighbor2[d] = (p_coordinate[d] - 1 + self.length) % self.length
                    adj_list[p].append(self._coordinate_to_point(neighbor2))

        return adj_list

    def _find_bonds(self):
        """Create bonds for each point. Similar to adjacency list
           but no repetition is calculated.
        """
        num_bonds = 0
        bonds = []
        for i in range(self.num_points):
            for j in self.adj_list[i]:
                if j > i:
                    num_bonds += 1
                    bonds.append((i, j))

        return num_bonds, bonds

    def generate_permutation_table(self):
        """Generate permutation table for a symmetric hypercube.
           Used to impose symmetry in the hypercube for the optimisation of the machine.
        """
        table = []
        for target_point in range(self.num_points):
            target_coordinate = self._point_to_coordinate(target_point)
            vector = [-1 for i in range(self.num_points)]
            for point in range(self.num_points):
                coordinate = self._point_to_coordinate(point)
                new_coordinate = [(coordinate[i] + target_coordinate[i]) % self.length for i in range(self.dimension)]
                vector[point] = self._coordinate_to_point(new_coordinate)
            table.append(vector)

        return table

    def _point_to_coordinate(self, point):
        """Convert a given point to a coordinate based on row-major order
        """
        assert point < self.num_points
        coordinate = []
        for i in reversed(range(self.dimension)):
            v = self.length ** i
            coordinate.append(point // v)
            point = point % v

        return list(reversed(coordinate))

    def _coordinate_to_point(self, coordinate):
        """Convert a given coordinate to a point based on row-major order
        """
        assert len(coordinate) == self.dimension
        point = 0
        for i in range(self.dimension):
            point += coordinate[i] * (self.length ** i)

        return point

    def to_xml(self):
        stri = ""
        stri += "<graph>\n"
        stri += "\t<type>hypercube</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<length>%d</length>\n" % self.length
        stri += "\t\t<dimension>%d</dimension>\n" % self.dimension
        stri += "\t\t<pbc>%s</pbc>\n" % str(self.pbc)
        stri += "\t</params>\n"
        stri += "</graph>\n"
        return stri
