"""
Common utilities used across active sampling modules in dora.
"""
import time

import numpy as np


class ArrayBuffer:
    """
    ArrayBuffer Class.

    Provides an efficient structure for numpy arrays growing on the first axis.
    Implemented with an automatically resizing buffer, providing matrices as
    views to this data by slicing along the first axis.

    Attributes
    ----------
    initial_size : int
        The initial size of all ArrayBuffer instances.
    shape : tuple
        The shape of the ArrayBuffer
    """

    initial_size = 10

    def __init__(self):
        """
        Initialise the ArrayBuffer.
        """
        self.__buffer = None
        self.__value = None
        self.__count = 0

    @property
    def shape(self):
        """
        The shape of the ArrayBuffer.

        See Also
        --------
        numpy.ndarray.shape : Analogous Class Property
        """
        return self.__value.shape

    def __len__(self):
        """
        Private Method.

        .. note :: DOCUMENTATION INCOMPLETE
        """
        return self.__count

    def __getitem__(self, index):
        """
        Private Method.

        .. note :: DOCUMENTATION INCOMPLETE
        """
        return self.__value.__getitem__(index)

    def __setitem__(self, index, val):
        """
        Private Method.

        .. note :: DOCUMENTATION INCOMPLETE
        """
        return self.__value.__setitem__(index, val)

    def __delitem__(self, index):
        """
        Private Method.

        .. note :: DOCUMENTATION INCOMPLETE
        """
        return self.__value.__delitem__(index)

    def __repr__(self):
        """
        Private Method.

        .. note :: DOCUMENTATION INCOMPLETE
        """
        return 'Buffer Contains:\n' + self.__value.__repr__()

    def __call__(self):
        """
        Private Method.

        .. note :: DOCUMENTATION INCOMPLETE
        """
        return self.__value

    def append(self, value):
        """
        Add an array_like to the buffer.

        This extending its first axis by adding a (1 x value.shape) row
        onto the end.

        Parameters
        ----------
        value : array_like
            A one dimensional array of length dims consistent with the buffer.
        """
        value = np.asarray(value)

        if self.__buffer is None:
            newsize = (ArrayBuffer.initial_size,) + value.shape  # tuples
            self.__buffer = np.zeros(newsize, value.dtype)

        assert(value.ndim == self.__buffer.ndim - 1)
        assert(value.shape == self.__buffer.shape[1:])

        self.__count += 1

        if self.__count >= self.__buffer.shape[0]:
            growth_factor = 2.0
            newsize = list(self.__buffer.shape)
            newsize[0] = np.floor(growth_factor * newsize[0] + 2.0)
            self.__buffer = np.resize(self.__buffer, newsize)

        self.__buffer[self.__count - 1] = value
        self.__value = self.__buffer[:self.__count]


def demo():

    d = 20
    n_stack = 10000

    buf = ArrayBuffer()

    st = time.time()
    for i in range(n_stack):
        # buf.append(np.random.random())
        buf.append(np.random.random(d))  # NOQA we dont do anything with a

    ft = time.time()
    print('Efficient buffer took {0:.5f} seconds'.format(ft - st))

    a = buf()

    import IPython
    IPython.embed()
    import sys
    sys.exit()
    print(a)
    exit()

    st = time.time()
    b = np.zeros((0, d))
    for i in range(n_stack):
        b = np.vstack((b, np.random.random(d)))
    ft = time.time()
    print('Repeated Vstack took {0:.5f} seconds'.format(ft - st))

    st = time.time()
    b_buf = []
    for i in range(n_stack):
        b_buf.append(np.random.random(d))
        b = np.array(b_buf)
    ft = time.time()
    print('List buffering and casting took {0:.5f} seconds'.format(ft - st))


if __name__ == '__main__':
    demo()
