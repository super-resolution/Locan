"""
This module provides methods for nearest-neighbor analysis.
"""
import surepy.constants
import surepy.analysis
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


def pdf_nnDistances_csr_2D(x, density):
    """
    probability density function for nearest-neighbor distances of points distributed in 2D at complete spatial
    randomness.

    :param x (float): distance
    :param density (float): density of points
    :return probability density function
    :rtype float
    """
    return 2 * density * np.pi * x * np.exp(-density * np.pi * x**2)


def pdf_nnDistances_csr_3D(x, density):
    """
    probability density function for nearest-neighbor distances of points distributed in 3D at complete spatial
    randomness.

    :param x (float): distance
    :param density (float): density of points
    :return probability density function
    :rtype float
    """
    a = (3 / 4 / np.pi / density)**(1/3)
    return 3 / a * (x/a)**2 * np.exp(-(x/a)**3)


def k_nearest_neighbor(points, k = 1, other_points = None):
    """ k_nearest_neighbor(points, k = 1, other_points = None) computes nearest neighbor distances. """
    # todo: take out points and get rid of if clause
    from sklearn.neighbors import NearestNeighbors
    if other_points is None:
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(points)
        distances, indices = nn.kneighbors()
        return distances[...,k-1]
    else:
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(other_points)
        distances, indices = nn.kneighbors(points)
        return distances[...,k-1]


class Nearest_neighbor_distances(surepy.analysis.Analysis):
    """
    Compute the k-nearest-neighbor distances within data or between data and other_data.

    Parameters:
    :param data (Selection object): Selection with localizations for which a nearest neighbor is computed.
    :param k (int): the kth neirest neighbor is computed.
    :param other_data (Selection object): Selection with localizations from which a nearest neighbor is taken.

    Attributes:
    :cvar count (int): a counter for counting instantiations
    :var data (Selection object): reference to the Selection object specified as input data.
    :var parameter (dict): a dictionary with all settings for the current computation.
    :var results (np.array): an array with the nearest neighbor distance for each localization in data.
    """
    count=0

    def __init__(self, data, k = 1, other_data = None):
        Analysis.count += 1
        self.__class__.count += 1
        self.data = data
        self.other_data = other_data
        self.parameter = {'method': 'k_nearest_neighbor'}
        self.parameter.update(k=k)
        self.results = self.compute_results(data=data, k=k, other_data=other_data)


    def compute_results(self, data, k = 1, other_data = None):
        """
        Compute the k-nearest-neighbor Euclidean distances for data or between data and other_data.

        :param data (Selection object): Selection with localizations for which a nearest neighbor is computed.
        :param k (int): the kth neirest neighbor is computed.
        :param other_data (Selection object): Selection with localizations from which a nearest neighbor is taken.

        :return np.array: the nearest neighbor distances
        """
        return k_nearest_neighbor(points=data.coordinates,
                                             k=k,
                                             other_points=None if other_data is None else other_data.coordinates
                                             )

    def hist(self, ax, bins='auto', normed=True):
        """
        Provide matplotlib axes showing results.

        :param bins (int, list, 'auto'): bin specification as used in matplotlib
        :param normed (bool):
        """
        values, bin_values, patches = ax.hist(self.results, bins=bins, normed=normed, label = 'data')
        x_data = (bin_values[:-1] + bin_values[1:]) / 2

        # fit to csr (does not make sense)
        # x_data = (bin_values[:-1]+bin_values[1:])/2
        # self._fit_to_csr_distribution(x_data, values)
        # todo: add 3D

        # estimate for csr given localization density
        try:
            density = self.data.localization_density if self.other_data is None else self.other_data.localization_density
            ax.plot(x_data, af.pdf_nnDistances_csr_2D(x_data, density), 'r-', label='CSR')
        except:
            pass

        ax.set(title = 'k-Nearest Neigbor Distances'+' (k = ' + str(self.parameter['k'])+')',
               xlabel = 'distance (nm)',
               ylabel = 'pdf' if normed else 'counts'
               )
        ax.legend(loc = 'best')
        ax.text(0.5,0.7,'localization density: %.2f'%density, transform = ax.transAxes)

    def _fit_to_csr_distribution(self, x_histo, y_histo):
        """ Fit results to analytical function for complete spatial randomness."""
        popt, pcov = curve_fit(af.pdf_nnDistances_csr_2D, x_histo, y_histo, p0=1e-3)
        plt.plot(x_histo, af.pdf_nnDistances_csr_2D(x_histo, *popt), 'b-', marker='.', label='CSR fit')
        # todo: add 3D