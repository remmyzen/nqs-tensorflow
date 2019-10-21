from sampler import Sampler
import numpy as np


class Gibbs(Sampler):

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
            sample = machine.get_new_visible(sample)

        return sample

    def get_all_samples(self, machine, initial_sample, num_samples):
        all_samples = []
        sample = initial_sample
        for i in range(self.num_steps):
            sample = machine.get_new_visible(sample)
            all_samples.append(sample)

        return all_samples

    def to_xml(self):
        str = ""
        str += "<sampler>\n"
        str += "\t<type>gibbs</type>\n"
        str += "\t<params>\n"
        str += "\t\t<num_samples>%d</num_samples>\n" % self.num_samples
        str += "\t\t<num_steps>%d</num_steps>\n" % self.num_steps
        str += "\t</params>\n"
        str += "</sampler>\n"
        return str
