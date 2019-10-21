import tensorflow as tf
from hamiltonian import Hamiltonian
import itertools
import numpy as np

class Heisenberg(Hamiltonian):

    def __init__(self, graph, jx=1.0, jy=1.0, jz=1.0):

        Hamiltonian.__init__(self, graph)
        self.jx = jx
        self.jy = jy
        self.jz = jz

    def calculate_hamiltonian_matrix(self, samples, num_samples):

        num_spins = self.graph.num_points

        diagonal_element = None
        off_diagonal_element = None
        spins = tf.split(samples, num_spins, axis=1)
        for (s, s_2) in self.graph.bonds:
            if diagonal_element is None:
                diagonal_element = self.jz * spins[s] * spins[s_2]
            else:
                diagonal_element = tf.concat((diagonal_element, self.jz * spins[s] * spins[s_2]), axis=1)

            if off_diagonal_element is None:
                off_diagonal_element = -(self.jx - self.jy * spins[s] * spins[s_2])
            else:
                off_diagonal_element = tf.concat((off_diagonal_element, -(self.jx - self.jy * spins[s] * spins[s_2])),
                                                 axis=1)

        diagonal_element = tf.reduce_sum(input_tensor=diagonal_element, axis=1, keepdims=True)

        hamiltonian = tf.concat((diagonal_element, off_diagonal_element), axis=1)

        return hamiltonian

    def flip(self, x, p1, p2, num_samples):
        num_spins = self.graph.num_points
        y = np.eye(num_spins, dtype=np.float32)
        y[p1][p1] = -1
        y[p2][p2] = -1
        return tf.matmul(x, tf.convert_to_tensor(value=y))

    def calculate_lvd(self, samples, machine, num_samples):

        lvd = machine.log_val_diff(samples, samples)
        for (s, s_2) in self.graph.bonds:
            new_config = self.flip(samples, s, s_2, num_samples)
            lvd = tf.concat((lvd, machine.log_val_diff(new_config, samples)), axis=1)
        return lvd

    def __str__(self):
        return "Heisenberg %dD, jx=%.2f, jy=%.2f, jz=%.2f" % (self.graph.dimension, self.jx, self.jy, self.jz)

    def to_xml(self):
        str = ""
        str += "<hamiltonian>\n"
        str += "\t<type>heisenberg</type>\n"
        str += "\t<params>\n"
        str += "\t\t<jx>%d</jx>\n" % self.jx
        str += "\t\t<jy>%d</jy>\n" % self.jy
        str += "\t\t<jz>%d</jz>\n" % self.jz
        str += "\t</params>\n"
        str += "</hamiltonian>\n"
        return str
