"""
Utility functions for rendering in 2 and 3 dimensions
"""
import warnings
from math import isclose

import numpy as np


__all__ = []


# todo: add DataFrame input
def _coordinate_ranges(locdata, range=None):
    """
    Provide coordinate range for locdata that can be fed into a binning algorithm.

    Parameters
    ----------
    locdata : pandas DataFrame or LocData object
        Localization data.
    range : tuple with shape (dimension, 2) or None or str 'zero'
        ((min_x, max_x), (min_y, max_y), ...) range for each coordinate;
        for None (min, max) range are determined from data;
        for 'zero' (0, max) range with max determined from data.

    Returns
    -------
    numpy array of float with shape (dimension, 2)
        A range (min, max) for each available coordinate.
    """
    if range is None:
        ranges_ = locdata.bounding_box.hull.T
    elif isinstance(range, str):
        if range == 'zero':
            ranges_ = locdata.bounding_box.hull
            ranges_[0] = np.zeros(len(ranges_))
            ranges_ = ranges_.T
        else:
            raise ValueError(f'The parameter range={range} is not defined.')
    else:
        if np.ndim(range) != locdata.dimension:
            raise TypeError(f'The tuple {range} must have the same dimension as locdata which is {locdata.dimension}.')
        else:
            ranges_ = np.asarray(range)

    return ranges_


def _bin_edges(n_bins, range):
    """
    Compute ndarray with bin edges from bins and range.

    Parameters
    ----------
    n_bins : int or tuple, list, ndarray with length equal to that of range.
        Number of bins to be used in all or each dimension for which a range is provided.
    range : tuple, list, ndarray of float with shape (n_dimension, 2)
        Minimum and maximum edge of binned range for each dimension.

    Returns
    -------
    list of ndarray
        Array(s) of bin edges
    """

    def bin_edges_for_single_range(n_bins_, range_):
        """Compute bins for one range"""
        return np.linspace(*range_, n_bins_ + 1, endpoint=True, dtype=float)

    if np.ndim(range) == 1:
        if np.ndim(n_bins) == 0:
            bin_edges = bin_edges_for_single_range(n_bins, range)
        elif np.ndim(n_bins) == 1:
            bin_edges = [bin_edges_for_single_range(n_bins_=n, range_=range) for n in n_bins]
        else:
            raise TypeError('n_bins and range must have the same dimension.')

    elif np.ndim(range) == 2:
        if np.ndim(n_bins) == 0:
            bin_edges = [bin_edges_for_single_range(n_bins, range_=single_range) for single_range in range]
        elif len(n_bins) == len(range):
            bin_edges = [_bin_edges(n_bins=b, range=r) for b, r in zip(n_bins, range)]
        else:
            raise TypeError('n_bins and range must have the same length.')

    else:
        raise TypeError('range has two many dimensions.')

    return bin_edges


def _bin_edges_from_size(bin_size, range, extend_range=True):
    """
    Compute ndarray with bin edges from bin size and range.

    Parameters
    ----------
    bin_size : float or tuple, list, ndarray with with length equal to that of range.
        Number of bins to be used in all or each dimension for which a range is provided.
    range : tuple, list, ndarray of float with shape (n_dimension, 2)
        Minimum and maximum edge of binned range for each dimension.
    extend_range : bool or None
        If for equally-sized bins the final bin_edge is different from the maximum range,
        the last bin_edge will be smaller than the maximum range but all bins are equally-sized (None);
        the last bin_edge will be equal to the maximum range but bins are not equally-sized (False);
        the last bin_edge will be larger than the maximum range but all bins are equally-sized (True).

    Returns
    -------
    list of ndarray
        Array(s) of bin edges
    """

    def bin_edges_for_single_range(bin_size, range):
        """Compute bins for one range"""
        bin_edges = np.arange(*range, bin_size, dtype=float)
        last_edge = bin_edges[-1] + bin_size

        if isclose(last_edge, range[-1]):
            bin_edges = np.append(bin_edges, last_edge)
        else:
            if extend_range is None:
                pass
            elif extend_range is True:
                bin_edges = np.append(bin_edges, last_edge)
            elif extend_range is False:
                bin_edges = np.append(bin_edges, range[-1])
            else:
                raise ValueError('`extend_range` must be None, True or False.')
        return bin_edges

    if np.ndim(range) == 1:
        if np.ndim(bin_size) == 0:
            bin_edges = bin_edges_for_single_range(bin_size, range)
        elif np.ndim(bin_size) == 1:
            bin_edges = [bin_edges_for_single_range(bin_size=n, range=range) for n in bin_size]
        else:
            raise TypeError('n_bins and range must have the same dimension.')

    elif np.ndim(range) == 2:
        if np.ndim(bin_size) == 0:
            bin_edges = [bin_edges_for_single_range(bin_size, range=single_range) for single_range in range]
        elif len(bin_size) == len(range):
            bin_edges = [_bin_edges_from_size(bin_size=b, range=r) for b, r in zip(bin_size, range)]
        else:
            raise TypeError('n_bins and range must have the same length.')

    else:
        raise TypeError('range has two many dimensions.')

    return bin_edges


def _bin_edges_to_number(bin_edges):
    """
    Check if bins are equally sized and return the number of bins.

    Parameters
    ----------
    bin_edges : tuple, list, ndarray of float with shape (n_dimension, n_bin_edges)
        Array of bin edges for each dimension

    Returns
    -------
    n_bins : int or ndarray of int
        Number of bins
    """
    def bin_edges_to_number_single_dimension(bin_edges):
        bin_edges = np.asarray(bin_edges, dtype=np.float64)
        differences = np.diff(bin_edges)
        all_equal = np.all(np.isclose(differences, differences[0]))
        if all_equal:
            n_bins = len(bin_edges)-1
        else:
            warnings.warn('Bins are not equally sized.')
            n_bins = None
        return n_bins

    if np.ndim(bin_edges) == 1 and np.asarray(bin_edges).dtype != object:
        n_bins = bin_edges_to_number_single_dimension(bin_edges)
    elif np.ndim(bin_edges) == 2 or np.asarray(bin_edges).dtype == object:
        n_bins = np.array([bin_edges_to_number_single_dimension(edges) for edges in bin_edges])
    else:
        raise TypeError('The shape of bin_edges must be (n_dimension, n_bin_edges).')

    return n_bins


def _bin_edges_to_centers(bin_edges):
    """
    Compute bin centers

    Parameters
    ----------
    bin_edges : tuple, list, ndarray of float with shape (n_dimension, n_bin_edges)
        Array of bin edges for each dimension

    Returns
    -------
    list of ndarray of shape (dimension, n_bins)
        bin centers
    """
    bin_centers = [np.diff(bedges) / 2 + bedges[0:-1] for bedges in bin_edges]
    return bin_centers


def _indices_to_bin_centers(bin_edges, indices):
    """
    Compute bin centers for given indices

    Parameters
    ----------
    bin_edges : tuple, list, ndarray of float with shape (n_dimension, n_bin_edges)
        Array of bin edges for each dimension
    indices : tuple, list, ndarray of int with shape (n_indices, n_dimension)
        Array of multi-dimensional indices, e.g. reflecting a list of vertices.

    Returns
    -------
    ndarray of shape (n_indices, n_dimension)
        selected bin centers
    """
    # todo: fix input for tuple and list
    bin_centers = _bin_edges_to_centers(bin_edges)
    selected_bin_centers = np.array([bc[ver] for bc, ver in zip(bin_centers, indices.T)]).T
    return selected_bin_centers

