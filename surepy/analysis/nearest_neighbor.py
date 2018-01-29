"""
This module provides methods for nearest-neighbor analysis.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from surepy.analysis.analysis import Analysis


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


class Nearest_neighbor_distances(Analysis):
    """
    Compute the k-nearest-neighbor distances within data or between data and other_data.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    k : int
        COmpute the kth nearest neighbor.
    other_locdata : LocData object
        Other localization data from which nearest neighbors are taken.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    results : Pandas DataFrame
        Dataframe with distance to and index of the nearest neighbor for each localization in locdata.
    meta : dict
        metadata
    """
    count=0

    def __init__(self, locdata, k=1, other_locdata=None):
        super().__init__(locdata, k=k, other_locdata=other_locdata)

        if other_locdata is None:
            self.localization_density = locdata.properties['Localization_density_bb']
        else:
            self.localization_density = other_locdata.properties['Localization_density_bb']


    def _compute_results(self, locdata, k=1, other_locdata=None):

        points = locdata.coordinates

        if other_locdata is None:
            nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(points)
            distances, indices = nn.kneighbors()
        else:
            other_points = other_locdata.coordinates
            nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(other_points)
            distances, indices = nn.kneighbors(points)

        return pd.DataFrame({'nn_distance': distances[...,k-1], 'nn_index': indices[...,k-1]})


    def hist(self, ax, bins='auto', normed=True, fit=False):
        """
        Provide matplotlib axes showing histogram of nearest-neighbor distances.

        Parameters
        ----------
        bins : int, list or 'auto'
            Bin specification as used in matplotlib.hist
        normed : bool
            Flag for normalization as used in matplotlib.hist
        fit : bool
            Flag indicating to fit pdf of nearest-neighbor distances under complete spatial randomness.
        """
        values, bin_values, patches = ax.hist(self.results['nn_distance'], bins=bins, normed=normed, label = 'data')
        x_data = (bin_values[:-1] + bin_values[1:]) / 2

        ax.plot(x_data, pdf_nnDistances_csr_2D(x_data, self.localization_density), 'r-', label='CSR')

        ax.set(title = 'k-Nearest Neigbor Distances'+' (k = ' + str(self.parameter['k'])+')',
               xlabel = 'distance (nm)',
               ylabel = 'pdf' if normed else 'counts'
               )
        ax.legend(loc = 'best')
        ax.text(0.5,0.7,'localization density: %.2f'%self.localization_density, transform = ax.transAxes)
