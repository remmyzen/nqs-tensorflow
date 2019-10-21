from sampler import Sampler
import tensorflow as tf
import numpy as np


class MetropolisLocal(Sampler):

    def __init__(self, num_samples, num_steps):
        Sampler.__init__(self, num_samples)
        self.num_steps = num_steps

    def get_initial_random_samples(self, sample_size, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples

        init_data = np.random.uniform(0, 1, (num_samples, sample_size))
        init_data[init_data < 0.5] = -1.
        init_data[init_data >= 0.5] = 1.

        return init_data.astype(np.float32)

    def sample(self, machine, initial_sample, num_samples):
        sample = initial_sample
        for i in range(self.num_steps):
            sample = self.sample_once(machine, sample, num_samples)

        return sample

    def get_all_samples(self, machine, initial_sample, num_samples):
        all_samples = []
        sample = initial_sample
        for i in range(self.num_steps):
            sample = self.sample_once(machine, sample, num_samples)
            all_samples.append(sample)

        return all_samples

    def sample_once(self, machine, starting_sample, num_samples):
        new_config = self.get_new_config(starting_sample, num_samples)
        ratio = tf.abs(tf.exp(machine.log_val_diff(new_config, starting_sample)))
      	random = tf.distributions.Uniform(0.0, 1.0).sample((num_samples, 1))
        accept = tf.squeeze(tf.greater(ratio, random))
        sample = tf.where(accept, new_config, starting_sample)
        return sample

    def get_new_config(self, sample, num_samples):
        num_points = int(sample.shape[1])
        position = np.random.randint(0, num_points, num_samples)
        row_indices = np.reshape(range(num_samples), (num_samples, 1))
        col_indices = np.reshape(position, (num_samples, 1))
        indices = tf.convert_to_tensor(value=np.concatenate((row_indices, col_indices), axis=1))
        elements = tf.gather_nd(sample, indices)
        old = tf.scatter_nd(indices, elements, (num_samples, num_points))
        new = tf.scatter_nd(indices, tf.negative(elements), (num_samples, num_points))
        return sample - old + new


    def to_xml(self):
        str = ""
        str += "<sampler>\n"
        str += "\t<type>metropolis_local</type>\n"
        str += "\t<params>\n"
        str += "\t\t<num_samples>%d</num_samples>\n" % self.num_samples
        str += "\t\t<num_steps>%d</num_steps>\n" % self.num_steps
        str += "\t</params>\n"
        str += "</sampler>\n"
        return str
