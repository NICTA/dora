"""
Array Buffer Usage Demonstration.
"""
import time

import numpy as np

from dora.active_sampling.utils import ArrayBuffer


def main():

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
    main()
