import numpy as np
from observable import Observable


class MagnetizationZ(Observable):
    def __init__(self, prob, confs, num_particles):
        Observable.__init__(self, prob, confs)
        self.num_particles = num_particles

    def get_value(self):
        num_particles = self.num_particles
        M = 0
        ## From exact with probability
        if self.confs is None:
            M = np.zeros(num_particles + 1)

            for part in range(num_particles):
                for i in range(len(self.prob)):
                    conf_bin = format(i, '#0%db' % (num_particles + 2))
                    ## configuration in binary -1 1
                    conf = np.array([1 if c == '1' else -1 for c in conf_bin[2:]])
                    M[part] += self.prob[i] * conf[part]
                    M[num_particles] += self.prob[i] * np.abs(np.sum(conf))
        ## From RBM
        else:
            M = np.zeros(num_particles + 1)

            for part in range(num_particles):
                for i, conf in enumerate(self.confs):
                    M[part] += self.prob[i] * conf[part]

            for i, conf in enumerate(self.confs):
                M[num_particles] += self.prob[i] * np.abs(np.sum(conf))

        return M

    def get_value_ferro(self, M=None):
        if M is None: M = self.get_value()
        return np.abs(np.sum(M)) / self.num_particles
                 

    def get_name(self):
        return 'Mz'

