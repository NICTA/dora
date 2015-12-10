import numpy as np


class AppendableArray(np.ndarray):

    def __new__(cls, array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    # DOES NOT WORK...
    def append(self, element):
        self.data = np.concatenate((self, np.array([element])), axis = 0)
