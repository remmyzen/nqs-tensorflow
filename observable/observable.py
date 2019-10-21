
class Observable(object):
    def __init__(self, prob, confs):
        self.prob = prob
        self.confs = confs

    def get_value(self):
        return None

    def get_name(self):
        return None
