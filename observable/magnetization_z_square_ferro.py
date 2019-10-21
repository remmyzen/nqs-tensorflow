import numpy as np
from observable import Observable


class MagnetizationZSquareFerro(Observable):
    def __init__(self, prob, confs, num_particles):
        Observable.__init__(self, prob, confs)
        self.num_particles = num_particles

    def get_value(self):
        num_particles = self.num_particles
        M = 0
        ## From exact with probability
        if self.confs is None:
            for i in range(len(self.prob)):
                conf_bin = format(i, '#0%db' % (num_particles + 2))
                ## configuration in binary -1 1
                conf = np.array([1 if c == '1' else -1 for c in conf_bin[2:]])
                M += self.prob[i] * (np.sum(conf) ** 2)
        ## From RBM
        else:
            ## Accumulation 
            for i, conf in enumerate(self.confs):
                M += self.prob[i] * (np.sum(conf) ** 2)

        return M
 
    def get_name(self):
        return 'Mz2Ferro'

