import numpy as np
from observable import Observable

class CorrelationZ2D(Observable):
    def __init__(self, prob, confs, num_particles):
        Observable.__init__(self, prob, confs)
        self.num_particles = num_particles

    def get_value(self):
        num_particles = self.num_particles

        ## From exact with probability
        if self.confs is None:
            numrows = int(np.sqrt(num_particles))
            ## mid pos for 2D
            pos =  numrows * (numrows / 2 - 1) + (numrows / 2)
            C_total = np.zeros((num_particles, num_particles))

            for part_1 in [pos]:
                for part_2 in range(0, num_particles):
                    for i in range(2 ** num_particles):
                        conf_bin = format(i, '#0%db' % (num_particles + 2))
                        ## configuration in binary -1 1
                        conf = np.array([1 if c == '1' else -1 for c in conf_bin[2:]])
                        C_total[part_1][part_2] += self.prob[i] * conf[part_1] * conf[part_2]  

            C_total = C_total.flatten()

        ## From RBM
        else:
            numrows = int(np.sqrt(num_particles))
            ## mid pos for 2D
            pos =  numrows * (numrows / 2 - 1) + (numrows / 2)
            C_total = np.zeros((num_particles, num_particles))

            for part_1 in [pos]:
                for part_2 in range(0, num_particles):
                    for i, conf in enumerate(self.confs):
                        C_total[part_1][part_2] += self.prob[i] * conf[part_1] * conf[part_2]  


            C_total = C_total.flatten()

        return C_total

    def get_value_antiferro(self, C_total=None):
        if C_total is None: C_total = self.get_value()

        cz_rbm_antiferro = 0.
        data_temp = C_total[1:self.num_particles+1]
        for i, da in enumerate(data_temp):
          cz_rbm_antiferro += ((-1) ** (i+1)) * da
        cz_rbm_antiferro /= (self.num_particles - 1)

        return cz_rbm_antiferro

    def get_value_ferro(self, C_total=None):
        if C_total is None: C_total = self.get_value()

        cz_rbm_ferro = 0.
        data_temp = C_total[1:self.num_particles+1]
        cz_rbm_ferro = np.sum(data_temp)
        cz_rbm_ferro /= (self.num_particles - 1)

        return cz_rbm_ferro
 
    def get_name(self):
        return 'Cz'
