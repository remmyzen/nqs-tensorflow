from machine.machine import Machine

class RBM(Machine):

    def __init__(self, num_visible, density=2):
        Machine.__init__(self)
        self.num_visible = num_visible
        self.density = density
        self.num_hidden = self.num_visible * self.density
        self.W = None
        self.bv = None
        self.bh = None
        self.connection = None

    def is_complex(self):
        return False
