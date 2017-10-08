import numpy as np


class Model(object):
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data

    def detect(self):
        """
        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError
