import numpy as np
import logging
from dora.active_sampling.types import AppendableArray


def main():

    x = AppendableArray([5, 2, 7])
    x.append(6)
    print(x)

    X = AppendableArray([[5, 2, 7], [4, 2, 5]])
    X.append([6, 1, 3])
    print(X)

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    main()
