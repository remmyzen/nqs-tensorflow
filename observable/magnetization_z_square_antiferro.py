import numpy as np
from observable import Observable

class MagnetizationZSquareAntiFerro(Observable):

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
                temp = 0
                for j, c in enumerate(conf):
                    temp += ((-1) ** j) * c
                M += self.prob[i] * (temp ** 2)
        ## From RBM
        else:
            ## Accumulation 
            for i, conf in enumerate(self.confs):
                temp = 0
                for j, c in enumerate(conf):
                    temp += ((-1) ** j) * c
                M += self.prob[i] * (temp ** 2)

        return M


    def get_name(self):
        return 'Mz2AntiFerro'
