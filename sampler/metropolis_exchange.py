from sampler import Sampler
import tensorflow as tf
import numpy as np


# This sampler is exclusively used for Heisenberg model
class MetropolisExchange(Sampler):

    def __init__(self, num_samples, num_steps, total_sz=0):
        Sampler.__init__(self, num_samples)
        self.num_steps = num_steps
        self.total_sz = total_sz

    # set total Sz to be 0
    def get_initial_random_samples(self, sample_size, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples

        assert self.total_sz <= sample_size
        assert (self.total_sz + sample_size) % 2 == 0

        plus = np.ones((sample_size + self.total_sz)/2) * 1.0
        minus = np.ones((sample_size - self.total_sz)/2) * -1.0
        model = np.concatenate((plus, minus))

        init_data = []
        for i in range(num_samples):
            data = np.copy(model)
            np.random.shuffle(data)
            init_data.append(data)

        init_data = np.array(init_data, np.float32)
        init_data = np.reshape(init_data, (num_samples, sample_size))

        return init_data

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
        position1 = np.random.randint(0, num_points, num_samples)
        position2 = np.random.randint(0, num_points, num_samples)
        row_indices = np.reshape(range(num_samples), (num_samples, 1))
        col_indices1 = np.reshape(position1, (num_samples, 1))
        col_indices2 = np.reshape(position2, (num_samples, 1))
        indices1 = tf.convert_to_tensor(value=np.concatenate((row_indices, col_indices1), axis=1))
        indices2 = tf.convert_to_tensor(value=np.concatenate((row_indices, col_indices2), axis=1))
        elements1 = tf.gather_nd(sample, indices1)
        elements2 = tf.gather_nd(sample, indices2)
        old1 = tf.scatter_nd(indices1, elements1, (num_samples, num_points))
        old2 = tf.scatter_nd(indices2, elements2, (num_samples, num_points))
        new1 = tf.scatter_nd(indices1, elements2, (num_samples, num_points))
        new2 = tf.scatter_nd(indices2, elements1, (num_samples, num_points))
        return sample - old1 - old2 + new1 + new2

    def to_xml(self):
        str = ""
        str += "<sampler>\n"
        str += "\t<type>metropolis_exchange</type>\n"
        str += "\t<params>\n"
        str += "\t\t<num_samples>%d</num_samples>\n" % self.num_samples
        str += "\t\t<num_steps>%d</num_steps>\n" % self.num_steps
        str += "\t\t<total_sz>%d</total_sz>\n" % self.total_sz
        str += "\t</params>\n"
        str += "</sampler>\n"
        return str
