"""
Base Sampler Module.

Provides the Base Sampler Class which contains the basic strategies for
active sampling a spatial field
"""
import uuid

from dora.active_sampling.utils import ArrayBuffer

import numpy as np


class Sampler:
    """
    Provide a basic template and interface to specific Sampler subclasses.

    Attributes
    ----------
    lower : numpy.ndarray
        Lower bounds for each parameter in the parameter space
    upper : numpy.ndarray
        Upper bounds for each parameter in the parameter space
    dims : int
        Dimension of the parameter space (number of parameters)
    X : ArrayBuffer
        Contiguous Buffer of feature vectors representing observed locations
        in the parameter space
    y : ArrayBuffer
        Contiguous Buffer of target outputs or expected (virtual) target
        outputs corresponding to the feature vectors 'X'
    virtual_flag : ArrayBuffer
        A contiguous array of boolean flags indicating virtual elements of 'y'

            True: Corresponding target output is virtual

            False: Corresponding target output is observed
    pending_results : dict
        A dictionary that maps the job ID to the corresponding index in both
        the 'X' and 'y' buffers.
    """

    def __init__(self, lower, upper):
        """
        Initialise the Sampler class.

        .. note::

            Currently only supports rectangular type restrictions on the
            parameter space

        Parameters
        ----------
        lower : array_like
            Lower or minimum bounds for the parameter space
        upper : array_like
            Upper or maximum bounds for the parameter space
        """
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.dims = self.upper.shape[0]
        assert (self.lower.ndim == 1) and (self.upper.ndim == 1)
        assert self.lower.shape[0] == self.dims
        self.X = ArrayBuffer()
        self.y = ArrayBuffer()
        self.virtual_flag = ArrayBuffer()
        self.pending_results = {}
        self.n_tasks = None

    def pick(self):
        """
        Pick the next feature location for the next observation to be taken.

        .. note::

            Currently only supports rectangular type restrictions on the
            parameter space

        Returns
        -------
        numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        str
            A random hexadecimal ID to identify the corresponding job

        Raises
        ------
        AssertionError
            Under all circumstances. See note above.
        """
        assert False

    def update(self, uid, y_true):
        """
        Update a job with its observed value.

        .. note::

            Currently a dummy function whose functionality will be
            filled by subclasses of the Sampler class

        Parameters
        ----------
        uid : str
            A hexadecimal ID that identifies the job to be updated
        y_true : float
            The observed value corresponding to the job identified by 'uid'

        Returns
        -------
        int
            Index location in the data lists 'Sampler.X' and
            'Sampler.y' corresponding to the job being updated

        Raises
        ------
        AssertionError
            Under all circumstances. See note above.
        """
        assert False

    def _assign(self, xq, yq_exp):
        """
        Assign a pair (location in parameter space, virtual target) a job ID.

        Parameters
        ----------
        xq : numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        yq_exp : float
            The virtual target output at that parameter location

        Returns
        -------
        str
            A random hexadecimal ID to identify the corresponding job
        """
        # Place a virtual observation onto the collected data
        n = len(self.X)
        self.X.append(xq)
        self.virtual_flag.append(True)

        # If we get a None, insert zeros instead
        if yq_exp is None and self.n_tasks is not None:
            self.y.append(np.zeros(self.n_tasks))
        else:
            self.y.append(yq_exp)  # then add the real one

        # Create an uid for this observation
        # m = hashlib.md5()
        # m.update(np.array(np.random.random()))
        # uid = m.hexdigest()
        uid = uuid.uuid4().hex  # "%032x" % random.getrandbits(128)

        # Note the index of corresponding to this picked location
        self.pending_results[uid] = n

        return uid

    def _update(self, uid, y_true):
        """
        Update a job with its observed value.

        Parameters
        ----------
        uid : str
            A hexadecimal ID that identifies the job to be updated
        y_true : float
            The observed value corresponding to the job identified by 'uid'

        Returns
        -------
        int
            Index location in the data lists 'Sampler.X' and
            'Sampler.y' corresponding to the job being updated
        """
        # Make sure the job uid given is valid
        if uid not in self.pending_results:
            raise ValueError('Result was not pending!')
        assert uid in self.pending_results

        # Kill the job and update collected data with true observation
        ind = self.pending_results.pop(uid)

        # If the user has been pushing Nones until now, we will init properly
        if self.n_tasks is None:
            self.n_tasks = len(np.atleast_1d(y_true))
            pending_count = len(self.y)
            self.y = ArrayBuffer()
            for _ in range(pending_count):
                self.y.append(np.zeros(self.n_tasks))

        self.y()[ind] = y_true
        self.virtual_flag()[ind] = False

        return ind


def random_sample(lower, upper, n):
    """
    Used to randomly sample the search space.

    Provide search parameters and the number of samples desired.

    Parameters
    ----------
    lower : array_like
        Lower or minimum bounds for the parameter space
    upper : array_like
        Upper or maximum bounds for the parameter space
    n : int
        Number of samples

    Returns
    -------
    np.ndarray
        Sampled location in feature space
    """
    dims = len(lower)
    X = np.random.random((n, dims))
    volume_range = [upper[i] - lower[i] for i in range(dims)]
    X_scaled = X * volume_range
    X_shifted = X_scaled + lower
    return X_shifted


def grid_sample(lower, upper, n):
    """
    Used to seed an algorithm with a regular pattern of corners and centres.

    This can be used to provide search parameters and the indices.

    Parameters
    ----------
    lower : array_like
        Lower or minimum bounds for the parameter space
    upper : array_like
        Upper or maximum bounds for the parameter space
    n : int
        Index of location

    Returns
    -------
    np.ndarray
        Sampled location in feature space
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    dims = lower.shape[0]
    n_corners = 2 ** dims
    if n < n_corners:
        xq = lower + (upper - lower) * \
            (n & 2 ** np.arange(dims) > 0).astype(float)
    elif n == n_corners:
        xq = lower + 0.5 * (upper - lower)
    else:
        assert(False)
    return xq
