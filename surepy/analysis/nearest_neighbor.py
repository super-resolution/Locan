"""
This module provides methods for nearest-neighbor analysis.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from surepy.analysis.analysis_base import _init_meta, _update_meta, save_results


#### The algorithms

def pdf_nnDistances_csr_2D(x, density):
    """
    Probability density function for nearest-neighbor distances of points distributed in 2D with complete spatial
    randomness.

    Parameters
    ----------
    x : float
        distance
    density : float
        density of points

    Returns
    -------
    float
        Probability density function pdf(x).
    """
    return 2 * density * np.pi * x * np.exp(-density * np.pi * x**2)


def pdf_nnDistances_csr_3D(x, density):
    """
    Probability density function for nearest-neighbor distances of points distributed in 3D with complete spatial
    randomness.


    Parameters
    ----------
    x : float
        distance
    density : float
        density of points

    Returns
    -------
    float
        Probability density function pdf(x).
    """
    a = (3 / 4 / np.pi / density)**(1/3)
    return 3 / a * (x/a)**2 * np.exp(-(x/a)**3)


def _nearest_neighbor_distances(points, k=1, other_points=None):

    if other_points is None:
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(points)
        distances, indices = nn.kneighbors()
    else:
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(other_points)
        distances, indices = nn.kneighbors(points)

    return pd.DataFrame({'nn_distance': distances[...,k-1], 'nn_index': indices[...,k-1]})


# The base analysis class

class _Nearest_neighbor_distances():
    """
    The base class for specialized analysis classes to be used on LocData objects.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    kwargs :
        Parameter that are passed to the algorithm.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    locdata : LocData object
        Localization data.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.
    """
    count = 0

    def __init__(self, locdata, meta, **kwargs):
        self.__class__.count += 1

        self.locdata = locdata
        self.parameter = kwargs
        self.meta = _init_meta(self)
        self.meta = _update_meta(self, meta)
        self.results = None

    def __del__(self):
        """ Update the counter upon deletion of class instance. """
        self.__class__.count -= 1

    def __str__(self):
        """ Return results in a printable format."""
        return str(self.results)

    def save_results(self, path):
        return save_results(self, path)

    def hist(self, ax=None, show=True, bins='auto', normed=True, fit=False):
        return hist(self, ax, show, bins, normed, fit)


    def compute(self):
        """ Apply analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError

    def save(self, path):
        """ Save Analysis object."""
        raise NotImplementedError

    def load(self, path):
        """ Load Analysis object."""
        raise NotImplementedError

    def report(self, ax):
        """ Show a report about analysis results."""
        raise NotImplementedError


# The specific analysis classes

class Nearest_neighbor_distances(_Nearest_neighbor_distances):
    '''
    Compute the k-nearest-neighbor distances within data or between data and other_data.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    k : int
        Compute the kth nearest neighbor.
    other_locdata : LocData object
        Other localization data from which nearest neighbors are taken.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    locdata : LocData object
        Localization data.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.
    '''
    count = 0

    def __init__(self, locdata=None, meta=None, k=1, other_locdata=None):
        super().__init__(locdata=locdata, meta=meta, k=k, other_locdata=other_locdata)

    def compute(self):
        points = self.locdata.coordinates

        # turn other_locdata into other_points
        new_parameter = {key: self.parameter[key] for key in self.parameter if key is not 'other_locdata'}
        if self.parameter['other_locdata'] is not None:
            other_points = self.parameter['other_locdata'].coordinates
        else:
            other_points = None

        self.results = _nearest_neighbor_distances(points=points,
                                                   **new_parameter, other_points=other_points)
        return self


#### Interface functions


def hist(self, ax=None, show=True, bins='auto', normed=True, fit=False):
    """
    Provide histogram as matplotlib axes object showing hist(results).

    Parameters
    ----------
    bins : int, list or 'auto'
        Bin specification as used in matplotlib.hist
    normed : bool
        Flag for normalization as used in matplotlib.hist
    fit : bool
        Flag indicating to fit pdf of nearest-neighbor distances under complete spatial randomness.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    if self.parameter['other_locdata'] is None:
        localization_density = self.locdata.properties['Localization_density_bb']
    else:
        localization_density = self.parameter['other_locdata'].properties['Localization_density_bb']

    values, bin_values, patches = ax.hist(self.results['nn_distance'], bins=bins, normed=normed, label = 'data')
    x_data = (bin_values[:-1] + bin_values[1:]) / 2

    ax.plot(x_data, pdf_nnDistances_csr_2D(x_data, localization_density), 'r-', label='CSR')

    ax.set(title = 'k-Nearest Neigbor Distances'+' (k = ' + str(self.parameter['k'])+')',
           xlabel = 'distance (nm)',
           ylabel = 'pdf' if normed else 'counts'
           )
    ax.legend(loc = 'best')
    ax.text(0.3,0.8,'density: {0:.2g}'.format(localization_density), transform = ax.transAxes)

    # show figure
    if show:
        plt.show()

    return None

