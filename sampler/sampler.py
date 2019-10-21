class Sampler(object):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def get_initial_random_samples(self, sample_size, num_samples=None):
        pass

    def sample(self, machine, initial_sample, num_samples):
        pass

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps
