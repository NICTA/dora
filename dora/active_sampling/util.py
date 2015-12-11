""" Common utilities used across active sampling modules in dora
"""
import numpy as np
import time


class ArrayBuffer():
    """
    ArrayBuffer Class

    Provides an efficient structure for numpy arrays growing on the first axis.
    Implemented with an automatically resising buffer, providing matrices as
    views to this data.

    Attributes:

    """
    def __init__(self, dims, dtype=float):
        """
        Initialise Buffer

        .. note:: Dimensionality of the second axis must be provided a priori.

        Parameters
        ----------
        dims : int
            Second dimension of growing array - shape will by (., d)

        dtype : type (optional)
            Data format of numpy array buffer

        """
        assert(isinstance(dims, int))
        self.__buffer = np.zeros((0, dims), dtype)
        self.__count = 0

    def append(self, value):
        """
        Adds a length d vector to the buffer, extending its first axis by
        adding a 1xdims row.

        Parameters
        ----------
        value : array_like
            A one dimensional array of length dims consistent with the buffer.
        """
        value = np.asarray(value)
        assert(value.ndim == 1)
        self.__count += 1
        if self.__count >= self.__buffer.shape[0]:
            growth_factor = 2.0
            newsize = np.floor(growth_factor*self.__buffer.shape[0] + 2.0)
            # make a new buffer - cant change old one in case it is referenced
            self.__buffer = np.resize(self.__buffer,
                                      (newsize, self.__buffer.shape[1]))

        self.__buffer[self.__count-1] = value
        return self.__buffer[:self.__count]


def demo():
    d = 20
    n_stack = 10000
    buf = ArrayBuffer(d)

    st = time.time()
    for i in range(n_stack):
        a = buf.append(np.random.random(d))  # NOQA we dont do anything with a

    ft = time.time()
    print('Efficient buffer took {0:.5f} seconds'.format(ft-st))

    st = time.time()
    b = np.zeros((0, d))
    for i in range(n_stack):
        b = np.vstack((b, np.random.random(d)))
    ft = time.time()
    print('Repeated Vstack took {0:.5f} seconds'.format(ft-st))

    st = time.time()
    b_buf = []
    for i in range(n_stack):
        b_buf.append(np.random.random(d))
        b = np.array(b_buf)
    ft = time.time()
    print('List buffering and casting took {0:.5f} seconds'.format(ft-st))

    return


if __name__ == '__main__':
    demo()
