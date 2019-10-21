class Hamiltonian(object):

    def __init__(self, graph):
        self.graph = graph

    # Calculates the Hamiltonian matrix from list of samples. Returns a tensor.
    def calculate_hamiltonian_matrix(self, samples, num_samples):
        # implemented in subclass
        return None

    def calculate_lvd(self, samples, machine, num_samples):
        # implemented in subclass
        return None



