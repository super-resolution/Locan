"""

Aggregate localization data.

This module provides functions to bin LocData objects to form a histogram or image.

Specify bins through one of the parameters (`bins`, `bin_edges`, `n_bins`, `bin_size`, `bin_range`, `labels`)
as further outlined in the documentation for :class:`Bins`.
"""
import warnings
from math import isclose
from collections import namedtuple

import numpy as np
import fast_histogram
from skimage import exposure

from surepy.constants import _has_boost_histogram
if _has_boost_histogram: import boost_histogram as bh
from surepy.data.properties.locdata_statistics import ranges


__all__ = ['Bins', 'adjust_contrast', 'histogram']


def _is_scalar(element) -> bool:
    return np.size(element) == 1 and np.ndim(element) == 0  # not isinstance(element, (tuple, list))


def _is_1d_array_of_scalar(element) -> bool:
    return np.size(element) > 0 and np.asarray(element).dtype != object and np.ndim(element) == 1


def _is_1d_array_of_two_scalar(element):
    return np.size(element) == 2 and np.asarray(element).dtype != object and np.ndim(element) == 1


def _is_2d_homogeneous_array_of_scalar(element) -> bool:
    return np.size(element) > 1 and np.asarray(element).dtype != object and np.ndim(element) == 2


def _is_2d_inhomogeneous_array_of_scalar(element) -> bool:
    return np.size(element) > 1 and np.asarray(element).dtype == object and np.ndim(element) in (1, 2)


def _is_2d_array_of_1d_array_of_scalar(element) -> bool:
    return (_is_2d_homogeneous_array_of_scalar(element) or _is_2d_inhomogeneous_array_of_scalar(element)) and \
        all(_is_1d_array_of_scalar(d) for d in element)


def _n_bins_to_bin_edges_one_dimension(n_bins: int, bin_range) -> np.ndarray:
    """
    Compute bin edges from n_bins and bin_range.

    Parameters
    ----------
    n_bins : int
        Number of bins.
    bin_range : tuple, list, numpy.ndarray of float with shape (2,)
        Minimum and maximum edge.

    Returns
    -------
    tuple of numpy.ndarray
        Array with bin edges.
    """
    return np.linspace(*bin_range, n_bins + 1, endpoint=True, dtype=float)


def _bin_size_to_bin_edges_one_dimension(bin_size, bin_range, extend_range=None) -> np.ndarray:
    """
    Compute bin edges from bin_size and bin_range.

    Use bin_edges if you need to use variable bin_sizes for Bins construction.

    Parameters
    ----------
    bin_size : float or tuple, list, numpy.ndarray of lfoat with shape (n_bins,).
        One size or sequence of sizes for bins.
    bin_range : tuple, list, or numpy.ndarray of float with shape (2,).
        Minimum and maximum edge of binned bin_range.
    extend_range : bool or None
        If for equally-sized bins the final bin_edge is different from the maximum bin_range,
        the last bin_edge will be smaller than the maximum bin_range but all bins are equally-sized (None);
        the last bin_edge will be equal to the maximum bin_range but bins are not equally-sized (False);
        the last bin_edge will be larger than the maximum bin_range but all bins are equally-sized (True).

        If for variable-sized bins the final bin_edge is different from the maximum bin_range,
        the last bin_edge will be smaller than the maximum bin_range (None);
        the last bin_edge will be equal to the maximum bin_range (False);
        the last bin_edge will be larger than the maximum bin_range but taken from the input sequence (True).

    Returns
    -------
    numpy.ndarray
        Array of bin edges
    """
    if _is_scalar(bin_size):
        bin_edges = np.arange(*bin_range, bin_size, dtype=float)
        last_edge = bin_edges[-1] + bin_size
        if isclose(last_edge, bin_range[-1]):
            bin_edges = np.append(bin_edges, last_edge)
        else:
            if extend_range is None:
                pass
            elif extend_range is True:
                bin_edges = np.append(bin_edges, last_edge)
            elif extend_range is False:
                bin_edges = np.append(bin_edges, bin_range[-1])
            else:
                raise ValueError('`extend_range` must be None, True or False.')

    elif _is_1d_array_of_scalar(bin_size):
        bin_edges_ = np.concatenate((np.asarray([bin_range[0]]), np.cumsum(bin_size) + bin_range[0]))
        bin_edges = bin_edges_[bin_edges_ <= bin_range[-1]]
        if extend_range is None:
            pass
        elif extend_range is True:
            if len(bin_edges_) > len(bin_edges):
                last_edge = bin_edges_[len(bin_edges)]
                bin_edges = np.append(bin_edges, last_edge)
            else:
                pass
        elif extend_range is False:
            bin_edges = np.append(bin_edges, bin_range[-1])
        else:
            raise ValueError('`extend_range` must be None, True or False.')

    else:
        raise TypeError("`bin_size` must be 0- or 1-dimensional.")

    return bin_edges


def _bin_edges_to_n_bins_one_dimension(bin_edges):
    n_bins = len(bin_edges) - 1
    return n_bins


def _bin_edges_to_n_bins(bin_edges):
    """
    Check if bins are equally sized and return the number of bins.

    Parameters
    ----------
    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimensions, n_bin_edges)
        Array of bin edges for each dimension.

    Returns
    -------
    tuple of int
        Number of bins
    """
    if np.ndim(bin_edges) == 1 and np.asarray(bin_edges).dtype != object:
        n_bins = (_bin_edges_to_n_bins_one_dimension(bin_edges),)
    elif np.ndim(bin_edges) == 2 or np.asarray(bin_edges).dtype == object:
        n_bins = tuple(_bin_edges_to_n_bins_one_dimension(edges) for edges in bin_edges)
    else:
        raise TypeError('The shape of bin_edges must be (n_dimensions, n_bin_edges).')
    return n_bins


def _bin_edges_to_bin_size_one_dimension(bin_edges):
    differences = np.diff(bin_edges)
    if np.all(differences == differences[0]):
        bin_size = differences[0]
    else:
        bin_size = tuple(differences)
    return bin_size


def _bin_edges_to_bin_size(bin_edges):
    """
    Check if bins are equally sized and return the number of bins.

    Parameters
    ----------
    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimensions, n_bin_edges)
        Array of bin edges for each dimension.

    Returns
    -------
    tuple of float
        Number of bins
    """
    if np.ndim(bin_edges) == 1 and np.asarray(bin_edges).dtype != object:
        bin_size = (_bin_edges_to_bin_size_one_dimension(bin_edges),)
    elif np.ndim(bin_edges) == 2 or np.asarray(bin_edges).dtype == object:
        bin_size = tuple(_bin_edges_to_bin_size_one_dimension(edges) for edges in bin_edges)
    else:
        raise TypeError('The shape of bin_edges must be (n_dimensions, n_bin_edges).')
    return bin_size


def _bin_edges_to_bin_centers(bin_edges):
    """
    Compute bin centers.

    Parameters
    ----------
    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimensions, n_bin_edges)
        Array of bin edges for each dimension

    Returns
    -------
    tuple of numpy.ndarray of shape (n_dimensions, n_bins)
        bin centers
    """
    if np.ndim(bin_edges) == 1 and np.asarray(bin_edges).dtype != object:
        bin_centers = (np.diff(bin_edges) / 2 + bin_edges[0:-1],)
    elif np.ndim(bin_edges) == 2 or np.asarray(bin_edges).dtype == object:
        bin_centers = tuple(np.diff(edges) / 2 + edges[0:-1] for edges in bin_edges)
    else:
        raise TypeError('The shape of bin_edges must be (n_dimensions, n_bin_edges).')
    return bin_centers


def _indices_to_bin_centers(bin_edges, indices):
    """
    Compute bin centers for given indices.

    Parameters
    ----------
    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimensions, n_bin_edges)
        Array of bin edges for each dimension
    indices : int or tuple, list, numpy.ndarray of int with shape (n_indices, n_dimensions)
        Array of multi-dimensional indices, e.g. reflecting a list of vertices.

    Returns
    -------
    numpy.ndarray of shape (n_indices, n_dimensions)
        selected bin centers
    """
    bin_centers = _bin_edges_to_bin_centers(bin_edges)
    if len(bin_centers) == 1:
        bin_centers = bin_centers[0]
    indices = np.asarray(indices)

    if _is_1d_array_of_scalar(bin_centers):
        if _is_scalar(indices):
            selected_bin_centers = bin_centers[indices]
        elif _is_1d_array_of_scalar(indices):
            selected_bin_centers = bin_centers[indices]
        elif _is_2d_homogeneous_array_of_scalar(indices):
            selected_bin_centers = bin_centers[indices]
        else:
            raise TypeError('The given array shapes cannot be processed.')

    elif _is_2d_array_of_1d_array_of_scalar(bin_centers):
        if _is_scalar(indices):
            selected_bin_centers = [bc[indices] for bc in bin_centers]
        elif _is_1d_array_of_scalar(indices):
            selected_bin_centers = [bc[indices] for bc in bin_centers]
        elif _is_2d_homogeneous_array_of_scalar(indices):
            if len(bin_centers) != len(indices.T):
                raise TypeError("`bin_centers` and `indices` must be for the same n_dimensions.")
            else:
                selected_bin_centers = np.array([bc[idx] for bc, idx in zip(bin_centers, indices.T)]).T
        else:
            raise TypeError('The given array shapes cannot be processed.')
    else:
        raise TypeError('The given array shapes cannot be processed.')

    return selected_bin_centers


class _BinsFromBoostHistogramAxis:
    """
    Adapter class for dealing with boost-histogram.axis elements through the `bins` parameter in :class:`Bins`.

    Parameters
    ----------
    bins : :class:`boost_histogram.axis.Axis` or :class:`boost_histogram.axis.AxesTuple`
    """

    def __init__(self, bins):
        if _has_boost_histogram and isinstance(bins, bh.axis.Axis):
            self._bins = bins
            self.n_dimensions = 1
            self.n_bins = (self._bins.size,)
            self.bin_size = (tuple(self._bins.widths),)
            self.bin_edges = (self._bins.edges,)
            self.bin_range = ((self._bins.edges[0], self._bins.edges[-1]),)
            self.bin_centers = (self._bins.centers,)

        elif _has_boost_histogram and isinstance(bins, bh.axis.AxesTuple):
            self._bins = bins
            self.n_dimensions = len(self._bins)
            self.n_bins = self._bins.size
            self.bin_size = tuple(tuple(arr) for arr in self._bins.widths.flatten())
            self.bin_edges = self._bins.edges.flatten()
            self.bin_range = tuple((axis.edges[0], axis.edges[-1]) for axis in self._bins)
            self.bin_centers = self._bins.centers.flatten()

        else:
            raise TypeError

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self._bins, attr)


class _BinsFromEdges:
    """
    Builder for :class:`Bins`.

    Parameters
    ----------
    bin_edges : list, tuple or numpy.ndarray
        Array with bin edges for all or each dimension.
    """

    def __init__(self, bin_edges):

        if not isinstance(bin_edges, (tuple, list, np.ndarray)):
            raise TypeError
        elif _is_1d_array_of_scalar(bin_edges):
            self.bin_edges = (np.array(bin_edges),)
            self.n_dimensions = 1
            self.bin_range = ((bin_edges[0], bin_edges[-1]),)
        elif _is_2d_array_of_1d_array_of_scalar(bin_edges):
            self.bin_edges = tuple(np.array(edges) for edges in bin_edges)
            self.n_dimensions = len(self.bin_edges)
            self.bin_range = tuple((edges[0], edges[-1]) for edges in self.bin_edges)
        else:
            raise TypeError("`bin_edges` must have 1 or 2 dimensions.")

        self.n_bins = _bin_edges_to_n_bins(self.bin_edges)
        self.bin_size = _bin_edges_to_bin_size(self.bin_edges)


class _BinsFromNumber:
    """
    Builder for :class:`Bins`.

    Parameters
    ----------
    n_bins : int or list, tuple or numpy.ndarray of ints
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_range : tuple, list, numpy.ndarray of float with shape (2,) or (n_dimensions, 2)
        Minimum and maximum edge for all or each dimensions.
    """

    def __init__(self, n_bins, bin_range):

        if np.ndim(n_bins) > 1:
            raise TypeError("`n_bins` must be 0- or 1-dimensional.")

        elif _is_scalar(n_bins):
            if _is_1d_array_of_two_scalar(bin_range):
                self.n_dimensions = 1
                self.bin_edges = (_n_bins_to_bin_edges_one_dimension(n_bins, bin_range),)
                self.n_bins = (n_bins,)
                self.bin_range = (bin_range,)

            elif _is_2d_homogeneous_array_of_scalar(bin_range):
                self.n_dimensions = len(bin_range)
                self.bin_edges = tuple(_n_bins_to_bin_edges_one_dimension(n_bins, single_range)
                                       for single_range in bin_range)
                self.n_bins = tuple(n_bins for _ in bin_range)
                self.bin_range = tuple(bin_range)

            else:
                raise TypeError('n_bins and/or bin_range have incorrect shapes.')

        elif _is_1d_array_of_scalar(n_bins) and len(n_bins) == 1:
            if _is_1d_array_of_two_scalar(bin_range):
                self.n_dimensions = 1
                self.bin_edges = (_n_bins_to_bin_edges_one_dimension(n_bins[0], bin_range),)
                self.n_bins = tuple(n_bins)
                self.bin_range = (bin_range,)

            elif _is_2d_homogeneous_array_of_scalar(bin_range):
                self.n_dimensions = len(bin_range)
                self.bin_edges = tuple(_n_bins_to_bin_edges_one_dimension(n_bins[0], single_range)
                                       for single_range in bin_range)
                self.n_bins = tuple(n_bins for _ in bin_range)
                self.bin_range = tuple(bin_range)

            else:
                raise TypeError('n_bins and/or bin_range have incorrect shapes.')

        elif _is_1d_array_of_scalar(n_bins) and len(n_bins) > 1:
            if _is_1d_array_of_two_scalar(bin_range):
                self.n_dimensions = len(n_bins)
                self.bin_edges = tuple(_n_bins_to_bin_edges_one_dimension(n_bins=n, bin_range=bin_range)
                                       for n in n_bins)
                self.n_bins = tuple(n_bins)
                self.bin_range = tuple(bin_range for _ in n_bins)

            elif _is_2d_homogeneous_array_of_scalar(bin_range):
                if len(n_bins) != len(bin_range):
                    raise TypeError('n_bins and bin_range have incompatible shapes.')
                else:
                    self.n_dimensions = len(n_bins)
                    self.bin_edges = tuple(_n_bins_to_bin_edges_one_dimension(n_bins=b, bin_range=r)
                                      for b, r in zip(n_bins, bin_range))
                    self.n_bins = tuple(n_bins)
                    self.bin_range = tuple(bin_range)

            else:
                raise TypeError('n_bins and/or bin_range have incorrect shapes.')

        else:
            raise TypeError('n_bins and/or bin_range have incorrect shapes.')

        self.bin_size = _bin_edges_to_bin_size(self.bin_edges)


class _BinsFromSize:
    """
    Builder for :class:`Bins`.

    Parameters
    ----------
    bin_size : float, list, tuple or numpy.ndarray of float with shape (n_dimensions,) or (n_dimensions, n_bins)
        The size of bins for all or each bin and for all or each dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and (1, 3) for the other dimension.
    bin_range : tuple, list, numpy.ndarray of float with shape (2,) or (n_dimensions, 2)
        Minimum and maximum edge for all or each dimensions.
    extend_range : bool or None
        If for equally-sized bins the final bin_edge is different from the maximum bin_range,
        the last bin_edge will be smaller than the maximum bin_range but all bins are equally-sized (None);
        the last bin_edge will be equal to the maximum bin_range but bins are not equally-sized (False);
        the last bin_edge will be larger than the maximum bin_range but all bins are equally-sized (True).

        If for variable-sized bins the final bin_edge is different from the maximum bin_range,
        the last bin_edge will be smaller than the maximum bin_range (None);
        the last bin_edge will be equal to the maximum bin_range (False);
        the last bin_edge will be larger than the maximum bin_range but taken from the input sequence (True).
    """
    def __init__(self, bin_size, bin_range, extend_range=None):

        if np.ndim(bin_size) > 2:
            raise TypeError("`bin_size` must be 0-, 1- or 2-dimensional. \
                            Construct from bin_edges if you need variable bin_sizes in one dimension.")

        elif _is_scalar(bin_size):
            if _is_1d_array_of_two_scalar(bin_range):
                self.n_dimensions = 1
                self.bin_edges = (_bin_size_to_bin_edges_one_dimension(bin_size, bin_range, extend_range),)
                self.bin_range = (bin_range,)

            elif _is_2d_homogeneous_array_of_scalar(bin_range):
                self.n_dimensions = len(bin_range)
                self.bin_edges = tuple(_bin_size_to_bin_edges_one_dimension(bin_size, single_range, extend_range)
                                       for single_range in bin_range)
                self.bin_range = tuple(bin_range)

            else:
                raise TypeError('bin_size and/or bin_range have incorrect shapes.')

        elif _is_1d_array_of_scalar(bin_size) and len(bin_size) == 1:
            if _is_1d_array_of_two_scalar(bin_range):
                self.n_dimensions = 1
                self.bin_edges = (_bin_size_to_bin_edges_one_dimension(bin_size[0], bin_range, extend_range),)
                self.bin_range = (bin_range,)

            elif _is_2d_homogeneous_array_of_scalar(bin_range):
                self.n_dimensions = len(bin_range)
                self.bin_edges = tuple(_bin_size_to_bin_edges_one_dimension(bin_size[0], single_range, extend_range)
                                       for single_range in bin_range)
                self.bin_range = tuple(bin_range)

            else:
                raise TypeError('bin_size and/or bin_range have incorrect shapes.')

        elif _is_1d_array_of_scalar(bin_size) and len(bin_size) > 1:
            if _is_1d_array_of_two_scalar(bin_range):
                self.n_dimensions = len(bin_size)
                self.bin_edges = tuple(_bin_size_to_bin_edges_one_dimension(bin_size=bs, bin_range=bin_range,
                                                                            extend_range=extend_range)
                                       for bs in bin_size)
                self.bin_range = tuple(bin_range for _ in bin_size)

            elif _is_2d_homogeneous_array_of_scalar(bin_range):
                if len(bin_size) != len(bin_range):
                    raise TypeError('bin_size and bin_range have incompatible shapes.')
                else:
                    self.n_dimensions = len(bin_size)
                    self.bin_edges = tuple(_bin_size_to_bin_edges_one_dimension(bin_size=bs, bin_range=br,
                                                                                extend_range=extend_range)
                                           for bs, br in zip(bin_size, bin_range))
                    self.bin_range = tuple(bin_range)

            else:
                raise TypeError('bin_size and/or bin_range have incorrect shapes.')

        elif _is_2d_array_of_1d_array_of_scalar(bin_size):
            if _is_1d_array_of_two_scalar(bin_range):
                self.n_dimensions = len(bin_size)
                self.bin_edges = tuple(_bin_size_to_bin_edges_one_dimension(bin_size=bs, bin_range=bin_range,
                                                                            extend_range=extend_range)
                                       for bs in bin_size)
                self.bin_range = tuple(bin_range for _ in bin_size)

            elif _is_2d_homogeneous_array_of_scalar(bin_range):
                if len(bin_size) != len(bin_range):
                    raise TypeError('bin_size and bin_range have incompatible shapes.')
                else:
                    self.n_dimensions = len(bin_size)
                    self.bin_edges = tuple(_bin_size_to_bin_edges_one_dimension(bin_size=bs, bin_range=br,
                                                                                extend_range=extend_range)
                                           for bs, br in zip(bin_size, bin_range))
                    self.bin_range = tuple(bin_range)

        elif _is_2d_inhomogeneous_array_of_scalar(bin_size):
            if _is_1d_array_of_two_scalar(bin_range):
                self.n_dimensions = len(bin_size)
                self.bin_edges = tuple(_bin_size_to_bin_edges_one_dimension(bin_size=bs, bin_range=bin_range,
                                                                            extend_range=extend_range)
                                       for bs in bin_size)
                self.bin_range = tuple(bin_range for _ in bin_size)

            elif _is_2d_homogeneous_array_of_scalar(bin_range):
                if len(bin_size) != len(bin_range):
                    raise TypeError('bin_size and bin_range have incompatible shapes.')
                else:
                    self.n_dimensions = len(bin_size)
                    self.bin_edges = tuple(_bin_size_to_bin_edges_one_dimension(bin_size=bs, bin_range=br,
                                                                                extend_range=extend_range)
                                           for bs, br in zip(bin_size, bin_range))
                    self.bin_range = tuple(bin_range)

        else:
            raise TypeError('bin_size and/or bin_range have incorrect shapes.')

        self.n_bins = _bin_edges_to_n_bins(self.bin_edges)
        self.bin_size = _bin_edges_to_bin_size(self.bin_edges)


# todo: add option for the following bin specifications.
# Bin specifications from an appropriate class or as defined in :func:`numpy.histogramdd`:
# The number of bins for all dimensions n.
# The number of bins for each dimension (nx, ny)
# A sequence of arrays ((edge_x1, edge_x2), (edge_y1, edge_y2)) describing the monotonically
# increasing bin edges along each dimension.
class Bins:
    """
    Bin definitions to be used in histogram and render functions.
    `Bins`can be instantiated from specifications for `bins` or `bin_edges`
    or for one of `n_bins` or `bin_size` in combination with `bin_range`.
    One and only one of (`bins`, `bin_edges`, `n_bins`, `bin_size`) must be different from None in any instantiating
    function.
    To pass bin specifications to other functions use an instance of `Bins` or `bin_edges`.

    Parameters
    ----------
    bins : `Bins` or `boost_histogram.axis.Axis` or None
        Specific class specifying the bins.

    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimensions, n_bin_edges) or None
        Array of bin edges for all or each dimension.

    bin_range : tuple, list, numpy.ndarray of float with shape (2,) or (n_dimensions, 2)
        Minimum and maximum edge for all or each dimensions.

    n_bins : int, list, tuple or numpy.ndarray or None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.

    bin_size : float, list, tuple or numpy.ndarray of float with shape (n_dimensions,) or (n_dimensions, n_bins)
        The size of bins in units of locdata coordinate units for all or each bin and for all or each dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and (1, 3) for the other dimension.

    labels : list of str with shape (n_dimensions,) or None
        Names for each bin axis.

    extend_range : bool or None
        Only applicable in combination with bin_size.
        If for equally-sized bins the final bin_edge is different from the maximum bin_range,
        the last bin_edge will be smaller than the maximum bin_range but all bins are equally-sized (None);
        the last bin_edge will be equal to the maximum bin_range but bins are not equally-sized (False);
        the last bin_edge will be larger than the maximum bin_range but all bins are equally-sized (True).

        If for variable-sized bins the final bin_edge is different from the maximum bin_range,
        the last bin_edge will be smaller than the maximum bin_range (None);
        the last bin_edge will be equal to the maximum bin_range (False);
        the last bin_edge will be larger than the maximum bin_range but taken from the input sequence (True).

    Attributes
    ----------
    n_dimensions : int
        The number of dimensions for which bins are provided.

    bin_range : tuple of tuples of float with shape (n_dimensions, 2)
        Minimum and maximum edge for each dimension.

    bin_edges : tuple of numpy.ndarray with shape (n_dimensions,)
        Array(s) with bin edges for each dimension.

    n_bins : tuple of int with shape (n_dimensions,)
        Number of bins for each dimension.

    bin_size : tuple of float with shape (n_dimensions,) or (n_dimensions, n_bins)
        Size of bins for each dimension.

    bin_centers :  tuple of numpy.ndarray with shape (n_dimensions)
        Array(s) with bin centers for all or each dimension.

    labels : list of str or None
        Names for each bin axis.
    """

    def __init__(self, bins=None, n_bins=None, bin_size=None, bin_edges=None, bin_range=None,
                 labels=None, extend_range=None):

        # check for correct inputs
        excluding_parameter = (bins, n_bins, bin_size, bin_edges)
        excluding_parameter_strings = ('bins', 'n_bins', 'bin_size', 'bin_edges')
        n_inputs = sum(param is not None for param in excluding_parameter)
        if n_inputs != 1:
            raise ValueError(f"One and only one of {excluding_parameter_strings} must be different from None.")

        # inject builder class
        if bins is not None:
            if bin_range is not None:
                raise ValueError("The parameter `bin_range` is derived from bins class and must be None.")
            if isinstance(bins, Bins):
                self._bins = bins
            if _has_boost_histogram and isinstance(bins, (bh.axis.Axis, bh.axis.AxesTuple)):
                self._bins = _BinsFromBoostHistogramAxis(bins)
        elif n_bins is not None:
            self._bins = _BinsFromNumber(n_bins, bin_range)
        elif bin_size is not None:
            self._bins = _BinsFromSize(bin_size, bin_range, extend_range)
        elif bin_edges is not None:
            if bin_range is not None:
                raise ValueError("The parameter `bin_range` is derived from `bin_edges` and must be None.")
            self._bins = _BinsFromEdges(bin_edges)

        self._bin_centers = None
        self.labels = labels

    @property
    def n_dimensions(self):
        return self._bins.n_dimensions

    @property
    def bin_edges(self):
        return self._bins.bin_edges

    @property
    def n_bins(self):
        return self._bins.n_bins

    @property
    def bin_size(self):
        return self._bins.bin_size

    @property
    def bin_range(self):
        return self._bins.bin_range

    @property
    def bin_centers(self):
        if self._bin_centers is None:
            self._bin_centers = getattr(self._bins, 'bin_centers', None)
            if self._bin_centers is None:
                self._bin_centers = _bin_edges_to_bin_centers(self.bin_edges)
        return self._bin_centers

    @property
    def labels(self):
        if self._labels is None:
            self._labels = getattr(self._bins, 'labels', None)
        return self._labels

    @labels.setter
    def labels(self, value):
        if value is None:
            self._labels = None
        elif isinstance(value, str):
            self._labels = [value]
        elif isinstance(value, (tuple, list)):
            self._labels = list(value)
        else:
            raise TypeError("`labels` must be str or list of str or None.")

        if self.labels is not None and len(self.labels) != self.n_dimensions:
            self._labels = None
            raise ValueError("`labels` must have a length of `n_dimensions`.")

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self._bins, attr)

    @property
    def is_equally_sized(self) -> np.array:
        """True for each dimension if all bins are of the same size."""
        if _is_2d_array_of_1d_array_of_scalar(self.bin_size):
            return tuple(all(np.isclose(bs, bs[0])) for bs in self.bin_size)
        elif _is_1d_array_of_scalar(self.bin_size):
            return tuple(True for _ in self.bin_size)
        elif _is_2d_inhomogeneous_array_of_scalar(self.bin_size):
            result = []
            for bs in self.bin_size:
                if _is_scalar(bs):
                    result.append(True)
                else:
                    result.append(all(np.isclose(bs, bs[0])))
            return tuple(result)
        else:
            raise TypeError

    def equalize_bin_size(self):
        """
        Return a new instance of `Bins` with bin_size set equal to the first bin_size element in each dimension
        and extend_range=None.
        """
        new_bin_size = []
        for bs in self.bin_size:
            try:
                new_bin_size.append(bs[0])
            except IndexError:
                new_bin_size.append(bs)
        return Bins(bin_size=new_bin_size, bin_range=self.bin_range, extend_range=None)


def adjust_contrast(image, rescale=True, **kwargs):
    """
    Adjust contrast of image by equalization or rescaling all values.

    Parameters
    ----------
    image : array-like
        Values to be adjusted
    rescale : True, tuple, False or None, 'equal', or 'unity.
        Rescale intensity values to be within percentile of max and min intensities
        (tuple with upper and lower bounds provided in percent).
        For True intensity values are rescaled to the min and max possible values of the given representation.
        For 'equal' intensity values are rescaled by histogram equalization.
        For 'unity' intensity values are rescaled to (0, 1).
        For None or False no rescaling occurs.

    Other Parameters
    ----------------
    kwargs : dict
        For 'rescale' = True kwargs are passed to :func:`skimage.exposure.rescale_intensity`.
        For 'rescale' = 'equal' kwargs are passed to :func:`skimage.exposure.equalize_hist`.

    Returns
    -------
    numpy.ndarray
    """
    if rescale is None or rescale is False:
        pass
    elif rescale is True:
        image = exposure.rescale_intensity(image, **kwargs)  # scaling to min/max of image intensities
    elif rescale == 'equal':
        image = exposure.equalize_hist(image, **kwargs)
    elif rescale == 'unity':
        image = exposure.rescale_intensity(image * 1., **kwargs)
    elif isinstance(rescale, tuple):
        p_low, p_high = np.ptp(image) * np.asarray(rescale) / 100 + image.min()
        image = exposure.rescale_intensity(image, in_range=(p_low, p_high))
    else:
        raise TypeError('Set rescale to tuple, None or "equal".')

    return image


def _fast_histo_mean(x, y, values, bins, range):
    """
    Provide histogram with averaged values for all counts in each bin.

    Parameters
    ----------
    x : array-like
        first coordinate values
    y : array-like
        second coordinate values
    values : int or float
        property to be averaged
    bins : sequence or int or None
        The bin specification as defined in fast_histogram_histogram2d:
            A sequence of arrays describing the monotonically increasing bin edges along each dimension.
            The number of bins for each dimension (nx, ny, … =bins)
    range : tuple with shape (dimension, 2) or None
        bin_range as requested by fast_histogram_histogram2d

    Returns
    -------
    numpy.ndarray
    """
    hist_1 = fast_histogram.histogram2d(x, y, range=range, bins=bins)
    hist_w = fast_histogram.histogram2d(x, y, range=range, bins=bins, weights=values)

    with np.errstate(divide='ignore', invalid='ignore'):
        hist_mean = np.true_divide(hist_w, hist_1)
        hist_mean[hist_mean == np.inf] = 0
        hist_mean = np.nan_to_num(hist_mean)

    return hist_mean


# todo: implement use of boost_histogram to deal with 3d images and variable bin sizes.
def histogram(locdata, loc_properties=None, other_property=None,
              bins=None, n_bins=None, bin_size=None, bin_edges=None, bin_range=None,
              rescale=None,
              **kwargs):
    """
    Make histogram of loc_properties (columns in locdata.data) by binning all localizations
    or averaging other_property within each bin.

    Parameters
    ----------
    locdata : LocData
        Localization data.
    loc_properties : list of str or None
        Localization properties to be grouped into bins. If None The coordinate_values of locdata are used.
    other_property : str or None
        Localization property that is averaged in each pixel. If None localization counts are
        shown.
    bins : int or sequence or `Bins` or `boost_histogram.axis.Axis` or None
        The bin specification as defined in :class:`Bins`
    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimensions, n_bin_edges) or None
        Array of bin edges for all or each dimension.
    n_bins : int, list, tuple or numpy.ndarray or None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float, list, tuple or numpy.ndarray or None
        The size of bins in units of locdata coordinate units for all or each dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other dimension.
        To specify arbitrary sequence of `bin_sizes` use `bin_edges` instead.
    bin_range : tuple or tuple of tuples of float with shape (n_dimensions, 2) or None or 'zero'
        The data bin_range to be taken into consideration for all or each dimension.
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    rescale : True, tuple, False or None, 'equal', or 'unity.
        Rescale intensity values to be within percentile of max and min intensities
        (tuple with upper and lower bounds provided in percent).
        For True intensity values are rescaled to the min and max possible values of the given representation.
        For 'equal' intensity values are rescaled by histogram equalization.
        For 'unity' intensity values are rescaled to (0, 1).
        For None or False no rescaling occurs.

    Other Parameters
    ----------------
    kwargs : dict
        For 'rescale' = True kwargs are passed to :func:`skimage.exposure.rescale_intensity`.
        For 'rescale' = 'equal' kwargs are passed to :func:`skimage.exposure.equalize_hist`.

    Returns
    -------
    namedtuple('Histogram', "data bins labels"): (numpy.ndarray, `Bins`, list)
    """
    if loc_properties is None:  # use coordinate_labels
        labels_ = locdata.coordinate_labels.copy()
        data = locdata.coordinates.T
    elif isinstance(loc_properties, str):
        if loc_properties not in locdata.data.columns:
            raise ValueError(f'{loc_properties} is not a valid property in locdata.data.')
        labels_ = [loc_properties]
        data = locdata.data[loc_properties].values.T
    elif isinstance(loc_properties, (tuple, list)):
        labels_ = list(loc_properties)
        if all(loc_property not in locdata.data.columns for loc_property in loc_properties):
            raise ValueError(f'{loc_properties} is not a valid property in locdata.data.')
        data = locdata.data[labels_].values.T
    else:
        raise ValueError(f'{loc_properties} is not a valid property in locdata.data.')

    if (bin_range is None or isinstance(bin_range, str)) and bin_edges is None:
        bin_range_ = ranges(locdata, loc_properties=labels_, special=bin_range)
    else:
        bin_range_ = bin_range

    bins = Bins(bins, n_bins, bin_size, bin_edges, bin_range_, labels=labels_)
    if bins.n_dimensions != len(labels_):
        raise TypeError("Shape of `bin_range` and `loc_properties` is incompatible.")

    if not all(bins.is_equally_sized):
        raise ValueError("Only equally sized bins can be forwarded to fast_histogram.")

    if other_property is None:
        # histogram data by counting points
        if np.ndim(data) == 1:
            img = fast_histogram.histogram1d(data, range=bins.bin_range[0], bins=bins.n_bins[0])
        elif data.shape[0] == 2:
            img = fast_histogram.histogram2d(*data, range=bins.bin_range, bins=bins.n_bins)
            img = img.T  # to show image in the same format as scatter plot
        elif data.shape[0] == 3:
            raise NotImplementedError
        else:
            raise TypeError('loc_properties must contain a string or a list with 2 or 3 elements.')
        labels_.append('counts')

    elif other_property in locdata.data.columns:
        # histogram data by averaging values
        if np.ndim(data) == 1:
            raise NotImplementedError
        elif data.shape[0] == 2:
            values = locdata.data[other_property].values
            img = _fast_histo_mean(*data, values, range=bins.bin_range, bins=bins.n_bins)
            img = img.T  # to show image in the same format as scatter plot
        elif data.shape[0] == 3:
            raise NotImplementedError
        else:
            raise TypeError('No more than 3 elements in loc_properties are allowed.')
        labels_.append(other_property)
    else:
        raise TypeError(f'Parameter for `other_property` {other_property} is not a valid property name.')

    if rescale:
        img = adjust_contrast(img, rescale, **kwargs)

    Histogram = namedtuple('Histogram', "data bins labels")
    return Histogram(img, bins, labels_)
