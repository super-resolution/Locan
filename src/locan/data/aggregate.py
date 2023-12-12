"""

Aggregate localization data.

This module provides functions to bin LocData objects to form a histogram or
image.

Specify bins through one of the parameters (`bins`, `bin_edges`, `n_bins`,
`bin_size`, `bin_range`, `labels`) as further outlined in the documentation
for :class:`Bins`.
"""
from __future__ import annotations

import logging
import warnings
from collections import namedtuple
from collections.abc import Iterable, Sequence
from math import isclose
from typing import Any, Literal, cast

import boost_histogram as bh
import fast_histogram
import numpy as np
import numpy.typing as npt

from locan.data.locdata import LocData
from locan.data.properties.locdata_statistics import ranges
from locan.data.validation import _check_loc_properties

__all__: list[str] = ["Bins", "histogram"]

logger = logging.getLogger(__name__)


def is_array_like(anything: Any) -> bool:
    """
    Return true if `anything` can be turned into a numpy.ndarray without
    creating elements of type object.

    Catches numpy.VisibleDeprecationWarning or ValueError when setting an
    array element with a sequence.

    Parameters
    ----------
    anything
        Anything to be classified as being array-like or not.

    Returns
    -------
    bool
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.asarray(anything).dtype != object
        except np.VisibleDeprecationWarning as e:
            if "Creating an ndarray from ragged nested sequences" in str(e):
                return False
            else:
                raise e
        except ValueError as e:
            if "setting an array element with a sequence." in str(e):
                return False
            else:
                raise e
        except Exception as e:
            raise e


def _is_scalar(element: Any) -> bool:
    return isinstance(element, (int, float)) or (
        is_array_like(element) and np.size(element) == 1 and np.ndim(element) == 0
    )


def _is_single_element(element: Any) -> bool:
    return isinstance(element, (int, float)) or (
        is_array_like(element) and np.size(element) == 1 and np.ndim(element) in (0, 1)
    )


def _is_1d_array_of_scalar(element: Any) -> bool:
    return is_array_like(element) and np.size(element) >= 1 and np.ndim(element) == 1


def _is_1d_array_of_two_or_more_scalar(element: Any) -> bool:
    return is_array_like(element) and np.size(element) > 1 and np.ndim(element) == 1


def _is_1d_array_of_two_scalar(element: Any) -> bool:
    return is_array_like(element) and np.size(element) == 2 and np.ndim(element) == 1


def _is_2d_homogeneous_array(element: Any) -> bool:
    return is_array_like(element) and np.size(element) > 1 and np.ndim(element) == 2


def _is_2d_inhomogeneous_array_of_1d_array_of_scalar(element: Any) -> bool:
    return not is_array_like(element) and all(
        _is_1d_array_of_scalar(el) for el in element
    )


def _is_2d_inhomogeneous_array(element: Any) -> bool:
    return not is_array_like(element) and all(
        _is_scalar(el) or _is_1d_array_of_scalar(el) for el in element
    )


def _is_2d_array_of_1d_array_of_scalar(element: Any) -> bool:
    return _is_2d_homogeneous_array(
        element
    ) or _is_2d_inhomogeneous_array_of_1d_array_of_scalar(element)


def _n_bins_to_bin_edges_one_dimension(
    n_bins: int, bin_range: tuple[float, float] | Sequence[float]
) -> npt.NDArray[np.float_]:
    """
    Compute bin edges from n_bins and bin_range.

    Parameters
    ----------
    n_bins
        Number of bins.
    bin_range
        Minimum and maximum edge. Array with shape (2,).

    Returns
    -------
    npt.NDArray[np.float_]
        Array with bin edges.
    """
    return np.linspace(*bin_range, n_bins + 1, endpoint=True, dtype=float)  # type: ignore


def _bin_size_to_bin_edges_one_dimension(
    bin_size: float | Sequence[float],
    bin_range: tuple[float, float] | Sequence[float],
    extend_range: bool | None = None,
) -> npt.NDArray[np.float_]:
    """
    Compute bin edges from bin_size and bin_range.

    Use bin_edges if you need to use variable bin_sizes for Bins construction.

    Parameters
    ----------
    bin_size
        One size or sequence of sizes for bins with shape (n_bins,).
    bin_range
        Minimum and maximum edge. Array with shape (2,).
    extend_range
        If for equally-sized bins the final bin_edge is different from the
        maximum bin_range, the last bin_edge will be smaller than the maximum
        bin_range but all bins are equally-sized (None);
        the last bin_edge will be equal to the maximum bin_range but bins
        are not equally-sized (False);
        the last bin_edge will be larger than the maximum bin_range but all
        bins are equally-sized (True).

        If for variable-sized bins the final bin_edge is different from the
        maximum bin_range, the last bin_edge will be smaller than the maximum
        bin_range (None);
        the last bin_edge will be equal to the maximum bin_range (False);
        the last bin_edge will be larger than the maximum bin_range but taken
        from the input sequence (True).

    Returns
    -------
    npt.NDArray[np.float_]
        Array of bin edges
    """
    if _is_scalar(bin_size):
        bin_edges = np.arange(*bin_range, bin_size, dtype=float)  # type: ignore
        if (
            bin_edges.size == 1
        ):  # this is the case if bin_size is greater than the bin_range
            if extend_range is True:
                bin_edges = np.append(bin_edges, bin_edges + bin_size)
            elif extend_range is None or extend_range is False:
                bin_edges = np.array(bin_range)
            else:
                raise ValueError("`extend_range` must be None, True or False.")
        else:
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
                    raise ValueError("`extend_range` must be None, True or False.")

    elif _is_1d_array_of_scalar(bin_size):
        bin_size = cast(Sequence[float], bin_size)
        if bin_size[0] > np.diff(bin_range):
            if extend_range is True:
                bin_edges = np.array([bin_range[0], bin_range[0] + bin_size[0]])
            elif extend_range is None or extend_range is False:
                bin_edges = bin_range
            else:
                raise ValueError("`extend_range` must be None, True or False.")
        else:
            bin_edges_ = np.concatenate(
                (np.asarray([bin_range[0]]), np.cumsum(bin_size) + bin_range[0])
            )
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
                raise ValueError("`extend_range` must be None, True or False.")

    else:
        raise TypeError("`bin_size` must be 0- or 1-dimensional.")

    return np.array(bin_edges)


def _bin_edges_to_n_bins_one_dimension(bin_edges: Sequence[float]) -> int:
    """
    Return the number of bins.

    Parameters
    ----------
    bin_edges
        Array of bin edges
        with shape (n_bin_edges,).

    Returns
    -------
    int
        Number of bins
    """
    n_bins = len(bin_edges) - 1
    return n_bins


def _bin_edges_to_n_bins(
    bin_edges: Sequence[float] | Sequence[Sequence[float]],
) -> tuple[int, ...]:
    """
    Check if bins are equally sized and return the number of bins.

    Parameters
    ----------
    bin_edges
        Bin edges for each dimension with shape (dimension, n_bin_edges).

    Returns
    -------
    tuple[int, ...]
        Number of bins
    """
    n_bins: tuple[int, ...]
    if _is_1d_array_of_scalar(bin_edges):
        bin_edges = cast("Sequence[float]", bin_edges)
        n_bins = (_bin_edges_to_n_bins_one_dimension(bin_edges),)
    elif _is_2d_array_of_1d_array_of_scalar(bin_edges):
        bin_edges = cast("Sequence[Sequence[float]]", bin_edges)
        n_bins = tuple(_bin_edges_to_n_bins_one_dimension(edges) for edges in bin_edges)
    else:
        raise TypeError("The shape of bin_edges must be (dimension, n_bin_edges).")
    return n_bins


def _bin_edges_to_bin_size_one_dimension(
    bin_edges: Sequence[float],
) -> float | npt.NDArray[np.float_]:
    """
    Compute the sizes of bins.

    Parameters
    ----------
    bin_edges
        Array of bin edges
        with shape (n_bin_edges,).

    Returns
    -------
    float | npt.NDArray[np.float_]
        Bin size for all bins or for each bin.
    """
    differences = np.diff(bin_edges)
    if np.all(differences == differences[0]):
        bin_size: float | npt.NDArray[np.float_] = differences[0]
    elif np.all(np.isclose(differences, differences[0], atol=0)):
        bin_size = differences[0]
        logger.debug(
            "bin_sizes differ by floating point instability with less than rtol=1.e-5"
        )
    else:
        bin_size = differences
    return bin_size


def _bin_edges_to_bin_size(
    bin_edges: Sequence[float] | Sequence[Sequence[float]],
) -> tuple[float, ...] | tuple[npt.NDArray[np.float_], ...]:
    """
    Compute the sizes of bins.

    Parameters
    ----------
    bin_edges
        Bin edges for each dimension with shape (dimension, n_bin_edges).

    Returns
    -------
    tuple[float, ...] | tuple[npt.NDArray[np.float_], ...]
        Bin size for all bins or for each bin in each dimension.
    """
    if _is_1d_array_of_scalar(bin_edges):
        bin_edges = cast("Sequence[float]", bin_edges)
        bin_size = (_bin_edges_to_bin_size_one_dimension(bin_edges),)
    elif _is_2d_array_of_1d_array_of_scalar(bin_edges):
        bin_edges = cast("Sequence[Sequence[float]]", bin_edges)
        bin_size = tuple(  # type: ignore
            _bin_edges_to_bin_size_one_dimension(edges) for edges in bin_edges
        )
    else:
        raise TypeError("The shape of bin_edges must be (dimension, n_bin_edges).")
    return bin_size  # type: ignore


def _bin_edges_to_bin_centers(
    bin_edges: Sequence[float] | Sequence[Sequence[float]],
) -> tuple[npt.NDArray[np.float_], ...]:
    """
    Compute bin centers.

    Parameters
    ----------
    bin_edges
        Bin edges for each dimension with shape (dimension, n_bin_edges).

    Returns
    -------
    tuple[npt.NDArray[np.float_], ...]
        Array of bin centers for each dimension with shape (n_bins,)
    """
    if _is_1d_array_of_scalar(bin_edges):
        bin_edges = cast("Sequence[float]", bin_edges)
        bin_centers = (np.diff(bin_edges) / 2 + bin_edges[0:-1],)
    elif _is_2d_array_of_1d_array_of_scalar(bin_edges):
        bin_edges = cast("Sequence[Sequence[float]]", bin_edges)
        bin_centers = tuple(np.diff(edges) / 2 + edges[0:-1] for edges in bin_edges)  # type: ignore
    else:
        raise TypeError("The shape of bin_edges must be (dimension, n_bin_edges).")
    return bin_centers


def _indices_to_bin_centers(
    bin_edges: Sequence[float] | Sequence[Sequence[float]], indices: npt.ArrayLike
) -> npt.NDArray[np.float_]:
    """
    Compute bin centers for given indices.

    Parameters
    ----------
    bin_edges
        Bin edges for each dimension with shape (dimension, n_bin_edges).
    indices
        Array of multi-dimensional indices with shape (n_indices, dimension),
        e.g. reflecting a list of vertices.

    Returns
    -------
    npt.NDArray[np.float_]
        Selected bin centers with shape (n_indices, dimension)
    """
    bin_centers = _bin_edges_to_bin_centers(bin_edges)
    if len(bin_centers) == 1:
        bin_centers = bin_centers[0]  # type: ignore
    indices = np.asarray(indices)

    if _is_1d_array_of_scalar(bin_centers):
        if _is_scalar(indices):
            selected_bin_centers = bin_centers[indices]
        elif _is_1d_array_of_scalar(indices):
            selected_bin_centers = bin_centers[indices]
        elif _is_2d_homogeneous_array(indices):
            selected_bin_centers = bin_centers[indices]
        else:
            raise TypeError("The given array shapes cannot be processed.")

    elif _is_2d_array_of_1d_array_of_scalar(bin_centers):
        if _is_scalar(indices):
            selected_bin_centers = np.array([bc[indices] for bc in bin_centers])
        elif _is_1d_array_of_scalar(indices):
            selected_bin_centers = np.array([bc[indices] for bc in bin_centers])
        elif _is_2d_homogeneous_array(indices):
            if len(bin_centers) != len(indices.T):
                raise TypeError(
                    "`bin_centers` and `indices` must be for the same dimension."
                )
            else:
                selected_bin_centers = np.array(
                    [bc[idx] for bc, idx in zip(bin_centers, indices.T)]
                ).T
        else:
            raise TypeError("The given array shapes cannot be processed.")
    else:
        raise TypeError("The given array shapes cannot be processed.")

    return selected_bin_centers


class _BinsFromBoostHistogramAxis:
    """
    Adapter class for dealing with `boost-histogram.axis` elements through the
    `bins` parameter in :class:`Bins`.

    Parameters
    ----------
    bins : boost_histogram.axis.Axis | boost_histogram.axis.AxesTuple
    """

    def __init__(self, bins: bh.axis.Axis | bh.axis.AxesTuple) -> None:
        self.dimension: int
        self.bin_range: tuple[tuple[float, float], ...]
        self.bin_edges: tuple[npt.NDArray[np.float_], ...]
        self.n_bins: tuple[int, ...]
        self.bin_size: tuple[float, ...] | tuple[npt.NDArray[np.float_], ...]
        self.bin_centers: tuple[npt.NDArray[np.float_], ...]
        self._bins: bh.axis.Axis | bh.axis.AxesTuple

        if isinstance(bins, bh.axis.Axis):
            self._bins = bins
            self.dimension = 1
            self.n_bins = (self._bins.size,)
            self.bin_size = (np.asarray(self._bins.widths),)
            self.bin_edges = (self._bins.edges,)
            self.bin_range = ((self._bins.edges[0], self._bins.edges[-1]),)
            self.bin_centers = (self._bins.centers,)

        elif isinstance(bins, bh.axis.AxesTuple):
            self._bins = bins
            self.dimension = len(self._bins)
            self.n_bins = self._bins.size
            self.bin_size = tuple(
                np.asarray(arr) for arr in self._bins.widths.flatten()
            )
            self.bin_edges = self._bins.edges.flatten()
            self.bin_range = tuple(
                (axis.edges[0], axis.edges[-1]) for axis in self._bins
            )
            self.bin_centers = self._bins.centers.flatten()

        else:
            raise TypeError

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self._bins, attr)


class _BinsFromEdges:
    """
    Builder for :class:`Bins`.

    Parameters
    ----------
    bin_edges : Sequence[float] | Sequence[Sequence[float]]
        Bin edges for all or each dimension with shape (dimension, n_bin_edges).
    """

    def __init__(self, bin_edges: Sequence[float] | Sequence[Sequence[float]]) -> None:
        self.dimension: int
        self.bin_range: tuple[tuple[float, float], ...]
        self.bin_edges: tuple[npt.NDArray[np.float_], ...]
        self.n_bins: tuple[int, ...]
        self.bin_size: tuple[float, ...] | tuple[npt.NDArray[np.float_], ...]

        if _is_1d_array_of_scalar(bin_edges):
            bin_edges = cast(Sequence[float], bin_edges)
            self.bin_edges = (np.array(bin_edges),)
            self.dimension = 1
            self.bin_range = ((bin_edges[0], bin_edges[-1]),)
        elif _is_2d_array_of_1d_array_of_scalar(bin_edges):
            bin_edges = cast(Sequence[Sequence[float]], bin_edges)
            self.bin_edges = tuple(np.array(edges) for edges in bin_edges)
            self.dimension = len(self.bin_edges)
            self.bin_range = tuple((edges[0], edges[-1]) for edges in self.bin_edges)
        else:
            raise TypeError("`bin_edges` must have 1 or 2 dimensions.")

        self.n_bins = _bin_edges_to_n_bins(self.bin_edges)  # type: ignore
        self.bin_size = _bin_edges_to_bin_size(self.bin_edges)  # type: ignore


class _BinsFromNumber:
    """
    Builder for :class:`Bins`.

    Parameters
    ----------
    n_bins : int | Sequence[int]
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_range : tuple[float, float] | Sequence[float] | Sequence[Sequence[float]]
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
    """

    def __init__(
        self,
        n_bins: int | Sequence[int],
        bin_range: tuple[float, float] | Sequence[float] | Sequence[Sequence[float]],
    ) -> None:
        self.dimension: int
        self.bin_range: tuple[tuple[float, float], ...]
        self.bin_edges: tuple[npt.NDArray[np.float_], ...]
        self.n_bins: tuple[int, ...]
        self.bin_size: tuple[float, ...] | tuple[npt.NDArray[np.float_], ...]

        if not is_array_like(n_bins) or np.ndim(n_bins) > 1:
            raise TypeError("`n_bins` must be 0- or 1-dimensional.")

        elif _is_scalar(n_bins):
            n_bins = cast("int", n_bins)
            if _is_1d_array_of_two_scalar(bin_range):
                bin_range = cast("tuple[float, float]", bin_range)
                self.dimension = 1
                self.bin_edges = (
                    _n_bins_to_bin_edges_one_dimension(n_bins, bin_range),
                )
                self.n_bins = (n_bins,)
                self.bin_range = (bin_range,)

            elif _is_2d_homogeneous_array(bin_range):
                bin_range = cast("Sequence[tuple[float, float]]", bin_range)
                self.dimension = len(bin_range)
                self.bin_edges = tuple(
                    _n_bins_to_bin_edges_one_dimension(n_bins, single_range)
                    for single_range in bin_range
                )
                self.n_bins = tuple(n_bins for _ in bin_range)
                self.bin_range = tuple(bin_range)

            else:
                raise TypeError("n_bins and/or bin_range have incorrect shapes.")

        elif _is_1d_array_of_scalar(n_bins):
            n_bins = cast("Sequence[int]", n_bins)
            if _is_1d_array_of_two_or_more_scalar(bin_range):
                bin_range = cast("Sequence[float]", bin_range)
                self.dimension = len(n_bins)
                self.bin_edges = tuple(
                    _n_bins_to_bin_edges_one_dimension(n_bins=n, bin_range=bin_range)
                    for n in n_bins
                )
                self.n_bins = tuple(n_bins)
                self.bin_range = tuple(tuple(bin_range) for _ in n_bins)  # type: ignore

            elif _is_2d_homogeneous_array(bin_range):
                bin_range = cast("Sequence[tuple[float, float]]", bin_range)
                if len(n_bins) != len(bin_range):
                    raise TypeError("n_bins and bin_range have incompatible shapes.")
                else:
                    self.dimension = len(n_bins)
                    self.bin_edges = tuple(
                        _n_bins_to_bin_edges_one_dimension(n_bins=b, bin_range=r)
                        for b, r in zip(n_bins, bin_range)
                    )
                    self.n_bins = tuple(n_bins)
                    self.bin_range = tuple(bin_range)

            else:
                raise TypeError("n_bins and/or bin_range have incorrect shapes.")

        else:
            raise TypeError("n_bins and/or bin_range have incorrect shapes.")

        self.bin_size = _bin_edges_to_bin_size(self.bin_edges)  # type: ignore


class _BinsFromSize:
    """
    Builder for :class:`Bins`.

    Parameters
    ----------
    bin_size : float | Sequence[float] | Sequence[Sequence[float]]
        The size of bins for all or each bin and for all or each dimension
        with shape (dimension,) or (dimension, n_bins).
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
        (1, 3) for the other dimension.
    bin_range : tuple[float, float] | Sequence[float] | Sequence[Sequence[float]]
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
    extend_range : bool | None
        If for equally-sized bins the final bin_edge is different from the
        maximum bin_range, the last bin_edge will be smaller than the maximum
        bin_range but all bins are equally-sized (None);
        the last bin_edge will be equal to the maximum bin_range but bins
        are not equally-sized (False);
        the last bin_edge will be larger than the maximum bin_range but all
        bins are equally-sized (True).

        If for variable-sized bins the final bin_edge is different from the
        maximum bin_range, the last bin_edge will be smaller than the maximum
        bin_range (None);
        the last bin_edge will be equal to the maximum bin_range (False);
        the last bin_edge will be larger than the maximum bin_range but taken
        from the input sequence (True).
    """

    def __init__(
        self,
        bin_size: float | Sequence[float] | Sequence[Sequence[float]],
        bin_range: tuple[float, float] | Sequence[float] | Sequence[Sequence[float]],
        extend_range: bool | None = None,
    ) -> None:
        self.dimension: int
        self.bin_range: tuple[tuple[float, float], ...]
        self.bin_edges: tuple[npt.NDArray[np.float_], ...]
        self.n_bins: tuple[int, ...]
        self.bin_size: tuple[float, ...] | tuple[npt.NDArray[np.float_], ...]

        if _is_scalar(bin_size):
            bin_size = cast("int | float", bin_size)
            if _is_1d_array_of_two_scalar(bin_range):
                bin_range = cast("tuple[float, float] | Sequence[float]", bin_range)
                self.dimension = 1
                self.bin_edges = (
                    _bin_size_to_bin_edges_one_dimension(
                        bin_size, bin_range, extend_range
                    ),
                )
                self.bin_range = (tuple(bin_range),)  # type: ignore

            elif _is_2d_homogeneous_array(bin_range):
                bin_range = cast("Sequence[Sequence[float]]", bin_range)
                self.dimension = len(bin_range)
                self.bin_edges = tuple(
                    _bin_size_to_bin_edges_one_dimension(
                        bin_size, single_range, extend_range
                    )
                    for single_range in bin_range
                )
                self.bin_range = tuple(
                    (edges[0], edges[-1]) for edges in self.bin_edges
                )

            else:
                raise TypeError("bin_size and/or bin_range have incorrect shapes.")

        elif _is_single_element(bin_size):
            if _is_1d_array_of_two_scalar(bin_range):
                bin_range = cast("tuple[float, float] | Sequence[float]", bin_range)
                self.dimension = 1
                self.bin_edges = (
                    _bin_size_to_bin_edges_one_dimension(
                        bin_size[0], bin_range, extend_range  # type: ignore
                    ),
                )
                self.bin_range = (tuple(bin_range),)  # type: ignore

            elif _is_2d_homogeneous_array(bin_range):
                bin_range = cast("Sequence[Sequence[float]]", bin_range)
                self.dimension = len(bin_range)
                self.bin_edges = tuple(
                    _bin_size_to_bin_edges_one_dimension(
                        bin_size[0], single_range, extend_range  # type: ignore
                    )
                    for single_range in bin_range
                )
                self.bin_range = tuple(
                    (edges[0], edges[-1]) for edges in self.bin_edges
                )

            else:
                raise TypeError("bin_size and/or bin_range have incorrect shapes.")

        elif _is_1d_array_of_scalar(bin_size):
            bin_size = cast("Sequence[int | float]", bin_size)
            if _is_1d_array_of_two_scalar(bin_range):
                bin_range = cast("tuple[float, float] | Sequence[float]", bin_range)
                self.dimension = len(bin_size)
                self.bin_edges = tuple(
                    _bin_size_to_bin_edges_one_dimension(
                        bin_size=bs, bin_range=bin_range, extend_range=extend_range
                    )
                    for bs in bin_size
                )
                self.bin_range = tuple(
                    (edges[0], edges[-1]) for edges in self.bin_edges
                )

            elif _is_2d_homogeneous_array(bin_range):
                bin_range = cast("Sequence[Sequence[float]]", bin_range)
                if len(bin_size) != len(bin_range):
                    raise TypeError("bin_size and bin_range have incompatible shapes.")
                else:
                    self.dimension = len(bin_size)
                    self.bin_edges = tuple(
                        _bin_size_to_bin_edges_one_dimension(
                            bin_size=bs, bin_range=br, extend_range=extend_range
                        )
                        for bs, br in zip(bin_size, bin_range)
                    )
                    self.bin_range = tuple(
                        (edges[0], edges[-1]) for edges in self.bin_edges
                    )

            else:
                raise TypeError("bin_size and/or bin_range have incorrect shapes.")

        elif _is_2d_inhomogeneous_array(bin_size) or _is_2d_homogeneous_array(
            bin_size
        ):  # _is_2d_array_of_1d_array_of_scalar(bin_size):
            bin_size = cast("Sequence[Sequence[int | float]]", bin_size)
            if _is_1d_array_of_two_scalar(bin_range):
                bin_range = cast("tuple[float, float] | Sequence[float]", bin_range)
                self.dimension = len(bin_size)
                self.bin_edges = tuple(
                    _bin_size_to_bin_edges_one_dimension(
                        bin_size=bs, bin_range=bin_range, extend_range=extend_range
                    )
                    for bs in bin_size
                )
                self.bin_range = tuple(
                    (edges[0], edges[-1]) for edges in self.bin_edges
                )

            elif _is_2d_homogeneous_array(bin_range):
                bin_range = cast("Sequence[Sequence[float]]", bin_range)
                if len(bin_size) != len(bin_range):
                    raise TypeError("bin_size and bin_range have incompatible shapes.")
                else:
                    self.dimension = len(bin_size)
                    self.bin_edges = tuple(
                        _bin_size_to_bin_edges_one_dimension(
                            bin_size=bs, bin_range=br, extend_range=extend_range
                        )
                        for bs, br in zip(bin_size, bin_range)
                    )
                    self.bin_range = tuple(
                        (edges[0], edges[-1]) for edges in self.bin_edges
                    )

        elif np.ndim(bin_size) > 2:
            raise TypeError(
                "`bin_size` must be 0-, 1- or 2-dimensional. \
                            Construct from bin_edges if you need variable bin_sizes in one dimension."
            )

        else:
            raise TypeError("bin_size and/or bin_range have incorrect shapes.")

        self.n_bins = _bin_edges_to_n_bins(self.bin_edges)  # type: ignore
        self.bin_size = _bin_edges_to_bin_size(self.bin_edges)  # type: ignore


# todo: add option for the following bin specifications.
# Bin specifications from an appropriate class or as defined in :func:`numpy.histogramdd`:
# The number of bins for all dimensions n.
# The number of bins for each dimension (nx, ny)
# A sequence of arrays ((edge_x1, edge_x2), (edge_y1, edge_y2)) describing the monotonically
# increasing bin edges along each dimension.
class Bins:
    """
    Bin definitions to be used in histogram and render functions.
    Bin edges are continuous, contiguous and monotonic.
    Bins can be instantiated from specifications for `bins` or `bin_edges`
    or for one of `n_bins` or `bin_size` in combination with `bin_range`.
    One and only one of (`bins`, `bin_edges`, `n_bins`, `bin_size`) must be different
    from None in any instantiating function.
    To pass bin specifications to other functions use an instance of
    `Bins` or `bin_edges`.

    Parameters
    ----------
    bins : Bins | boost_histogram.axis.Axis | boost_histogram.axis.AxesTuple | None
        Specific class specifying the bins.
    bin_edges : Sequence[float] | Sequence[Sequence[float]] | None
        Bin edges for all or each dimension
        with shape (dimension, n_bin_edges).
    bin_range : tuple[float, float] | Sequence[float] | Sequence[Sequence[float]]
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
    n_bins : int | Sequence[int] | None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float | Sequence[float] | Sequence[Sequence[float]] | None
        The size of bins for all or each bin and for all or each dimension
        with shape (dimension,) or (dimension, n_bins).
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
        (1, 3) for the other dimension.
        To specify arbitrary sequence of `bin_size` use `bin_edges` instead.
    labels : list[str] | None
        Names for each bin axis with shape (dimension,)
    extend_range : bool | None
        If for equally-sized bins the final bin_edge is different from the
        maximum bin_range, the last bin_edge will be smaller than the maximum
        bin_range but all bins are equally-sized (None);
        the last bin_edge will be equal to the maximum bin_range but bins
        are not equally-sized (False);
        the last bin_edge will be larger than the maximum bin_range but all
        bins are equally-sized (True).

        If for variable-sized bins the final bin_edge is different from the
        maximum bin_range, the last bin_edge will be smaller than the maximum
        bin_range (None);
        the last bin_edge will be equal to the maximum bin_range (False);
        the last bin_edge will be larger than the maximum bin_range but taken
        from the input sequence (True).

    Attributes
    ----------
    dimension : int
        The number of dimensions for which bins are provided.
    bin_range : tuple[tuple[float, float], ...]
        Minimum and maximum edge for each dimension with shape (dimension, 2).
    bin_edges : tuple[npt.NDArray[np.float_], ...]
        Array(s) with bin edges for each dimension with shape (dimension,)
    n_bins : tuple[int, ...]
        Number of bins for each dimension with shape (dimension,)
    bin_size : tuple[float, ...] | tuple[npt.NDArray[np.float_], ...]
        Size of bins for each dimension with shape (dimension,)
        or with shape (dimension, n_bins).
    bin_centers :  tuple[npt.NDArray[np.float_], ...]
        Array(s) with bin centers for all or each dimension
        with shape (dimension,).
    labels : list[str] | None
        Names for each bin axis.
    boost_histogram_axes : boost_histogram.axis.AxesTuple
        Axis definitions for boost-histogram
    """

    def __init__(
        self,
        bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
        n_bins: int | Sequence[int] | None = None,
        bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
        bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
        bin_range: tuple[float, float]
        | Sequence[float]
        | Sequence[Sequence[float]]
        | None = None,
        labels: list[str] | None = None,
        extend_range: bool | None = None,
    ) -> None:
        self._bins: Bins | _BinsFromBoostHistogramAxis | _BinsFromNumber | _BinsFromSize | _BinsFromEdges
        self._labels: list[str] | None

        # check for correct inputs
        excluding_parameter = (bins, n_bins, bin_size, bin_edges)
        excluding_parameter_strings = ("bins", "n_bins", "bin_size", "bin_edges")
        n_inputs = sum(param is not None for param in excluding_parameter)
        if n_inputs != 1:
            raise ValueError(
                f"One and only one of {excluding_parameter_strings} "
                f"must be different from None."
            )

        # inject builder class
        if bins is not None:
            if bin_range is not None:
                raise ValueError(
                    "The parameter `bin_range` is derived from bins class "
                    "and must be None."
                )
            if isinstance(bins, Bins):
                self._bins = bins
            if isinstance(bins, (bh.axis.Axis, bh.axis.AxesTuple)):
                self._bins = _BinsFromBoostHistogramAxis(bins)
        elif n_bins is not None:
            self._bins = _BinsFromNumber(n_bins, bin_range)  # type: ignore
        elif bin_size is not None:
            self._bins = _BinsFromSize(bin_size, bin_range, extend_range)  # type: ignore
        elif bin_edges is not None:
            if bin_range is not None:
                raise ValueError(
                    "The parameter `bin_range` is derived from `bin_edges` "
                    "and must be None."
                )
            self._bins = _BinsFromEdges(bin_edges)

        self._bin_centers: tuple[npt.NDArray[np.float_], ...] | None = None
        self.labels = labels
        self._boost_histogram_axes: bh.axis.AxesTuple | None = None

    @property
    def dimension(self) -> int:
        return self._bins.dimension

    @property
    def bin_edges(self) -> tuple[npt.NDArray[np.float_], ...]:
        return_value: tuple[npt.NDArray[np.float_], ...] = self._bins.bin_edges
        return return_value

    @property
    def n_bins(self) -> tuple[int, ...]:
        return self._bins.n_bins

    @property
    def bin_size(self) -> tuple[float, ...] | tuple[npt.NDArray[np.float_], ...]:
        return self._bins.bin_size

    @property
    def bin_range(self) -> tuple[tuple[float, float], ...]:
        return self._bins.bin_range

    @property
    def bin_centers(self) -> tuple[npt.NDArray[np.float_], ...]:
        if self._bin_centers is None:
            self._bin_centers = getattr(self._bins, "bin_centers", None)
            if self._bin_centers is None:
                self._bin_centers = _bin_edges_to_bin_centers(self.bin_edges)  # type: ignore
        return_value: tuple[npt.NDArray[np.float_], ...] = self._bin_centers
        return return_value

    @property
    def labels(self) -> list[str] | None:
        if self._labels is None:
            self._labels = getattr(self._bins, "labels", None)
        return self._labels

    @labels.setter
    def labels(self, value: str | Sequence[str] | None) -> None:
        if value is None:
            self._labels = None
        elif isinstance(value, str):
            self._labels = [value]
        elif isinstance(value, (tuple, list)):
            self._labels = list(value)
        else:
            raise TypeError("`labels` must be str or list of str or None.")

        if self._labels is not None and len(self.labels) != self.dimension:  # type: ignore
            self._labels = None
            raise ValueError("`labels` must have a length of `dimension`.")

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self._bins, attr)

    @property
    def is_equally_sized(self) -> tuple[bool, ...]:
        """True for each dimension if all bins are of the same size."""
        if _is_1d_array_of_scalar(self.bin_size):
            return tuple(True for _ in self.bin_size)
        elif _is_2d_array_of_1d_array_of_scalar(self.bin_size):
            return tuple(all(np.isclose(bs, bs[0])) for bs in self.bin_size)  # type: ignore

        elif _is_2d_inhomogeneous_array(self.bin_size):
            result = []
            for bs in self.bin_size:
                if _is_scalar(bs):
                    result.append(True)
                else:
                    result.append(all(np.isclose(bs, bs[0])))  # type: ignore
            return tuple(result)
        else:
            raise TypeError

    def equalize_bin_size(self) -> Bins:
        """
        Return a new instance of `Bins` with bin_size set equal to the
        first bin_size element in each dimension
        and extend_range=None.
        """
        new_bin_size = []
        for bs in self.bin_size:
            try:
                new_bin_size.append(bs[0])  # type: ignore
            except IndexError:
                new_bin_size.append(bs)
        return Bins(bin_size=new_bin_size, bin_range=self.bin_range, extend_range=None)

    @property
    def boost_histogram_axes(self) -> bh.axis.AxesTuple:
        """Axis definitions for boost-histogram"""
        if self._boost_histogram_axes is None:
            axes = []
            for index in range(self.dimension):
                if self.is_equally_sized[index]:
                    axis = bh.axis.Regular(self.n_bins[index], *self.bin_range[index])
                else:
                    axis = bh.axis.Variable(self.bin_edges[index])  # type: ignore
                axes.append(axis)
            self._boost_histogram_axes = bh.axis.AxesTuple(axes)
        return self._boost_histogram_axes


def _histogram_fast_histogram(data: npt.ArrayLike, bins: Bins) -> npt.NDArray[np.int_]:
    """
    Provide histogram with counts in each bin.

    Parameters
    ----------
    data
        Coordinate values with shape (dimensions, n_points) to be binned
    bins
        The bin specification

    Returns
    -------
    npt.NDArray[np.int_]
    """
    data = np.asarray(data)
    if data.shape[0] == 1:
        img: npt.NDArray[np.int_] = fast_histogram.histogram1d(
            data, range=bins.bin_range[0], bins=bins.n_bins[0]
        )
    elif data.shape[0] == 2:
        img = fast_histogram.histogram2d(*data, range=bins.bin_range, bins=bins.n_bins)
    else:
        raise TypeError("Dimension of data must be 1 or 2.")
    return img


def _histogram_boost_histogram(data: npt.ArrayLike, bins: Bins) -> npt.NDArray[np.int_]:
    """
    Provide histogram with counts in each bin.

    Parameters
    ----------
    data
        Coordinate values with shape (n_dimensions, n_points) to be binned
    bins
        The bin specification

    Returns
    -------
    npt.NDArray[np.int_]
    """
    hist = bh.Histogram(*bins.boost_histogram_axes).fill(*data)  # type: ignore
    img: npt.NDArray[np.int_] = hist.view()
    return img


def _histogram_mean_fast_histogram(
    data: npt.ArrayLike, bins: Bins, values: npt.ArrayLike
) -> npt.NDArray[np.float_]:
    """
    Provide histogram with averaged values for all counts in each bin.

    Parameters
    ----------
    data
        Coordinate values with shape (n_dimensions, n_points) to be binned
    bins
        The bin specification
    values
        Values with shape (n_points,) to be averaged in each bin

    Returns
    -------
    npt.NDArray[np.float_]
    """
    data = np.asarray(data)
    if data.shape[0] == 1:
        hist: npt.NDArray[np.float_] = fast_histogram.histogram1d(
            data, range=bins.bin_range[0], bins=bins.n_bins[0]
        )
        hist_w = fast_histogram.histogram1d(
            data, range=bins.bin_range[0], bins=bins.n_bins[0], weights=values
        )
    elif data.shape[0] == 2:
        hist = fast_histogram.histogram2d(*data, range=bins.bin_range, bins=bins.n_bins)
        hist_w = fast_histogram.histogram2d(
            *data, range=bins.bin_range, bins=bins.n_bins, weights=values
        )
    else:
        raise TypeError("Dimension of data must be 1 or 2.")

    with np.errstate(divide="ignore", invalid="ignore"):
        hist = np.true_divide(hist_w, hist)
        hist[hist == np.inf] = 0
        # hist = np.nan_to_num(hist)
    return hist


def _histogram_mean_boost_histogram(
    data: npt.ArrayLike, bins: Bins, values: npt.ArrayLike
) -> npt.NDArray[np.float_]:
    """
    Provide histogram with averaged values for all counts in each bin.

    Parameters
    ----------
    data : npt.ArrayLike
        Coordinate values with shape (n_dimensions, n_points) to be binned
    bins
        The bin specification
    values
        Values with shape (n_points,) to be averaged in each bin

    Returns
    -------
    npt.NDArray[np.float_]
    """
    hist = bh.Histogram(*bins.boost_histogram_axes, storage=bh.storage.Mean()).fill(  # type: ignore
        *data, sample=values
    )
    # bh.Histogram yields zero for mean values in bins with zero counts
    mean_values: npt.NDArray[np.float_] = hist.values()
    mask = hist.counts() == 0
    mean_values[mask] = np.nan
    return mean_values


def histogram(
    locdata: LocData,
    loc_properties: str | Iterable[str] | None = None,
    other_property: str | None = None,
    bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
    n_bins: int | Sequence[int] | None = None,
    bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_range: tuple[float, float]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Literal["zero", "link"]
    | None = None,
) -> tuple[npt.NDArray[np.int_ | np.float_], Bins, list[str]]:
    """
    Make histogram of loc_properties (columns in `locdata.data`)
    by binning all localizations
    or averaging other_property within each bin.

    Parameters
    ----------
    locdata
        Localization data.
    loc_properties
        Localization properties to be grouped into bins.
        If None The coordinate_values of locdata are used.
    other_property
        Localization property that is averaged in each pixel.
        If None localization counts are shown.
    bins
        The bin specification as defined in :class:`Bins`
    bin_edges
        Bin edges for all or each dimension
        with shape (dimension, n_bin_edges).
    bin_range
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
    n_bins
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size
        The size of bins for all or each bin and for all or each dimension
        with shape (dimension,) or (dimension, n_bins).
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
        (1, 3) for the other dimension.
        To specify arbitrary sequence of `bin_size` use `bin_edges` instead.

    Returns
    -------
    namedtuple('Histogram', "data bins labels"): (npt.NDArray[np.int_ | np.float_], Bins, list[str])
    """
    labels_ = _check_loc_properties(locdata, loc_properties)
    data = locdata.data[labels_].values.T
    img: npt.NDArray[np.int_ | np.float_]

    if (
        (bin_range is None or isinstance(bin_range, str))
        and bin_edges is None
        and bins is None
    ):
        bin_range_ = ranges(locdata, loc_properties=labels_, special=bin_range)
    else:
        bin_range_ = bin_range  # type: ignore

    try:
        bins = Bins(
            bins=bins,
            n_bins=n_bins,
            bin_size=bin_size,
            bin_edges=bin_edges,
            bin_range=bin_range_,  # type: ignore
            labels=labels_,
        )
    except ValueError as exc:  # the error is raised again only to adapt the message.
        raise ValueError(
            "Bin dimension and len of `loc_properties` is incompatible."
        ) from exc

    if other_property is None:
        # histogram data by counting points
        if data.shape[0] == 2:
            # we are using fast-histogram for 2D since it is even faster
            # than boost_histogram
            img = _histogram_fast_histogram(data, bins)
        elif data.shape[0] == 1 or data.shape[0] == 3:
            img = _histogram_boost_histogram(data, bins)
        else:
            raise TypeError(
                "loc_properties must contain a string or a list with 1, 2 or 3 "
                "elements."
            )
        labels_.append("counts")

    elif other_property in locdata.data.columns:
        # histogram data by averaging values
        values = locdata.data[other_property].values
        if data.shape[0] == 2:
            img = _histogram_mean_fast_histogram(data=data, bins=bins, values=values)  # type: ignore
        elif data.shape[0] == 1 or data.shape[0] == 3:
            img = _histogram_mean_boost_histogram(data=data, bins=bins, values=values)  # type: ignore
        else:
            raise TypeError("No more than 3 elements in loc_properties are allowed.")
        labels_.append(other_property)
    else:
        raise TypeError(
            f"Parameter for `other_property` {other_property} is not a valid property "
            f"name."
        )

    Histogram = namedtuple("Histogram", "data bins labels")
    return Histogram(img, bins, labels_)


def _accumulate_1d(
    data: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
    return_data: bool = False,
    return_counts: bool = False,
) -> tuple[
    npt.NDArray[np.int_], list[int], list[int] | None, npt.NDArray[np.int_] | None
]:
    """
    Bin data and collect data elements contained in each bin.
    The returned `bin_indices` refer to the given bins including index[0] for
    underflow data and index[n_bins] for overflow data.

    Parameters
    ----------
    data
        Data array of shape (n_points,) or (n_points, dimensions)
        All points are binned with regard to the first dimension.
    bin_edges
        Array of bin edges for corresponding dimension.
    return_data
        If true, grouped data elements are returned.
    return_counts
        If true, counts (number of elements per bin) are returned.

    Note
    ----
    Even though the returned data groups are sorted according to the bins,
    the data within groups is not sorted.

    Returns
    -------
    tuple[npt.NDArray[np.int_], list[int], list[int] | None, npt.NDArray[np.int_] | None]
        bin_indices, data_indices, collection, counts.
    """
    data_ = np.array(data)
    if data_.ndim > 1:
        data_ = data_[:, 0]
    # identify bins indices
    bin_identifier = np.digitize(data_, bins=bin_edges)  # type: ignore
    # bin_identifier 0 and n_bins represent out of bounds data

    # sort data
    sorted_indices = np.argsort(bin_identifier, kind="stable")

    # group data
    bin_indices, n_elements = np.unique(bin_identifier, return_counts=True)
    # bin_indices (like bin_identifier) 0 and n_bins represent out of bounds data

    cumsum = np.cumsum(n_elements)
    start_indices = np.insert(cumsum[:-1], 0, 0)
    stop_indices = cumsum
    data_indices = [
        sorted_indices[start:stop] for start, stop in zip(start_indices, stop_indices)
    ]

    collection = [data[indices_] for indices_ in data_indices] if return_data else None  # type: ignore
    counts = n_elements if return_counts else None

    return bin_indices, data_indices, collection, counts  # type: ignore


def _accumulate_2d(
    data: npt.ArrayLike,
    bin_edges: tuple[npt.ArrayLike, ...],
    return_data: bool = False,
    return_counts: bool = False,
) -> tuple[
    npt.NDArray[np.int_], list[int], list[int] | None, npt.NDArray[np.int_] | None
]:
    """
    Bin data and collect data elements contained in each bin.
    All points are binned with regard to the first and second dimension.
    The returned `bin_indices` refer to the given bins
    including index[0] for underflow data
    and index[n_bins] for overflow data.

    Parameters
    ----------
    data
        Data array of shape (n_points, dimensions)
    bin_edges
        Array of bin edges for corresponding dimensions.
    return_data
        If true, grouped data elements are returned.
    return_counts
        If true, counts (number of elements per bin) are returned.

    Returns
    -------
    tuple[npt.NDArray[np.int_], list[int], list[int] | None, npt.NDArray[np.int_] | None]
        bin_indices, data_indices, collection, counts.
    """
    data_ = np.array(data)
    if data_.size == 0:
        bin_indices = np.array([])
        data_indices: list[int] = []
        collection: list[int] | None | None = [] if return_data else None
        counts = np.array([]) if return_counts else None
        return bin_indices, data_indices, collection, counts

    # accumulate first dimension
    bin_indices_first_dim, data_indices_first_dim, _, _ = _accumulate_1d(
        data=data_[:, 0], bin_edges=bin_edges[0]
    )

    # traverse groups
    bin_indices_ = []
    data_indices = []
    counts_ = []
    for bin_index_first_dim_, data_indices_first_dim_ in zip(
        bin_indices_first_dim, data_indices_first_dim
    ):
        grouped_data_ = data_[:, 1][data_indices_first_dim_]
        bin_indices_group, data_indices_group, _, counts_group = _accumulate_1d(
            data=grouped_data_, bin_edges=bin_edges[1], return_counts=return_counts
        )

        # form multi-dimensional bin_indices
        first = np.repeat(bin_index_first_dim_, len(bin_indices_group))
        new_bin_indices = np.vstack([first, bin_indices_group]).T
        bin_indices_.append(new_bin_indices)
        counts_.append(counts_group)

        # form multi-dimensional data_indices
        new_data_indices = [
            data_indices_first_dim_[idxs] for idxs in data_indices_group  # type: ignore
        ]
        data_indices.extend(new_data_indices)

    bin_indices = np.concatenate(bin_indices_)
    counts = np.concatenate(counts_) if return_counts else None  # type: ignore
    collection = [data_[idxs] for idxs in data_indices] if return_data else None

    return bin_indices, data_indices, collection, counts
