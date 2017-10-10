import numpy as np


class Model(object):
    def __init__(self):
        pass

    def convert_to_nparray(self, data):
        if not isinstance(data, np.ndarray):
            return np.array(data)

    def detect(self, *args):
        """
        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError
