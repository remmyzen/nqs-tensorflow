import tensorflow as tf
from hamiltonian import Hamiltonian
import itertools
import numpy as np


class Ising (Hamiltonian):

    def __init__(self, graph, j=1.0, h=1.0):

        Hamiltonian.__init__(self, graph)
        self.j = j
        self.h = h

    def calculate_hamiltonian_matrix(self, samples, num_samples):

        num_spins = self.graph.num_points

        interact_energy = None
        spins = tf.split(samples, num_spins, axis=1)
        for (s, s_2) in self.graph.bonds:
            if interact_energy is None:
                interact_energy = -self.j * spins[s] * spins[s_2]
            else:
                interact_energy = tf.concat((interact_energy, -self.j * spins[s] * spins[s_2]), axis=1)

        interact_energy = tf.reduce_sum(input_tensor=interact_energy, axis=1, keepdims=True)
        external_energy = tf.fill((num_samples, num_spins), -self.h)
        hamiltonian = tf.concat((interact_energy, external_energy), axis=1)

        return hamiltonian
    
    def flip(self, x, p, num_samples):
        num_spins = self.graph.num_points

        y = np.eye(num_spins, dtype=np.float32)
        y[p][p] = -1
        return tf.matmul(x, tf.convert_to_tensor(value=y))

    def calculate_lvd(self, samples, machine, num_samples):
        lvd = machine.log_val_diff(samples, samples)
        for s in range(self.graph.num_points):
            new_config = self.flip(samples, s, num_samples)
            lvd = tf.concat((lvd, machine.log_val_diff(new_config, samples)), axis=1)
        return lvd

    def __str__(self):
        return "Ising %dD, h=%.2f, J=%.2f" % (self.graph.dimension, self.h, self.j)

    def to_xml(self):
        str = ""
        str += "<hamiltonian>\n"
        str += "\t<type>ising</type>\n"
        str += "\t<params>\n"
        str += "\t\t<j>%.2f</j>\n" % self.j
        str += "\t\t<h>%.2f</h>\n" % self.h
        str += "\t</params>\n"
        str += "</hamiltonian>\n"
        return str

