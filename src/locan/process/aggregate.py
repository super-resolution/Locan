"""

Aggregate localization data.

This module provides :func:`histogram` to bin LocData objects into counts or
per-bin mean values that form a histogram or image. The bins are described by
:class:`locan.process.bins.Bins`.

All bins are half-open intervals ``[edge_i, edge_i+1)``. :func:`histogram`
widens a range that is derived from the data by a negligible amount (1e-12 of
the range) so that the localization with the maximum coordinate is counted.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Literal, NamedTuple, cast

import boost_histogram as bh
import fast_histogram
import numpy as np
import numpy.typing as npt

from locan.data.locdata import LocData
from locan.data.validation import _check_loc_properties
from locan.process.bins import (
    BinEdgesInput,
    BinRangeInput,
    Bins,
    BinSizeInput,
    NBinsInput,
)
from locan.process.properties.locdata_statistics import ranges

__all__: list[str] = ["histogram"]

logger = logging.getLogger(__name__)

# Relative widening of a data-derived range so that the maximum value falls
# into the last half-open bin.
_RANGE_WIDENING_RTOL: float = 1e-12


def _data_bin_range(
    locdata: LocData,
    loc_properties: list[str],
    special: Literal["zero", "link"] | None,
) -> npt.NDArray[np.float64] | None:
    """
    Determine bin ranges from data such that the maximum value is included
    in the last half-open bin.
    """
    bin_range = ranges(locdata, loc_properties=loc_properties, special=special)
    if bin_range is None:
        return None
    bin_range_ = np.array(bin_range, dtype=np.float64)
    lower, upper = bin_range_[:, 0], bin_range_[:, 1]
    # One ulp is not enough: the backends compute the bin index as
    # (x - low) / (high - low) * n_bins, which can still round up to n_bins.
    widened = upper + (upper - lower) * _RANGE_WIDENING_RTOL
    bin_range_[:, 1] = np.maximum(widened, np.nextafter(upper, np.inf))
    return bin_range_


def _histogram_fast_histogram(
    data: npt.ArrayLike, bins: Bins
) -> npt.NDArray[np.float64]:
    """
    Provide histogram with counts in each bin.

    Only applicable for equally sized bins in one or two dimensions.

    Parameters
    ----------
    data
        Coordinate values with shape (dimensions, n_points) to be binned
    bins
        The bin specification

    Returns
    -------
    npt.NDArray[np.float64]
    """
    data = np.asarray(data)
    if data.shape[0] == 1:
        img: npt.NDArray[np.float64] = fast_histogram.histogram1d(
            data[0], range=bins.bin_range[0], bins=bins.n_bins[0]
        )
    elif data.shape[0] == 2:
        img = fast_histogram.histogram2d(*data, range=bins.bin_range, bins=bins.n_bins)
    else:
        raise TypeError("Dimension of data must be 1 or 2.")
    return img


def _histogram_boost_histogram(
    data: npt.ArrayLike, bins: Bins
) -> npt.NDArray[np.float64]:
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
    npt.NDArray[np.float64]
    """
    data = np.asarray(data)
    hist = bh.Histogram(*bins.to_boost_histogram_axes()).fill(*data)
    img: npt.NDArray[np.float64] = np.asarray(hist.view(), dtype=np.float64)
    return img


def _histogram_mean_fast_histogram(
    data: npt.ArrayLike, bins: Bins, values: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Provide histogram with averaged values for all counts in each bin.

    Only applicable for equally sized bins in one or two dimensions.
    Bins without counts carry NaN.

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
    npt.NDArray[np.float64]
    """
    data = np.asarray(data)
    if data.shape[0] == 1:
        hist: npt.NDArray[np.float64] = fast_histogram.histogram1d(
            data[0], range=bins.bin_range[0], bins=bins.n_bins[0]
        )
        hist_w = fast_histogram.histogram1d(
            data[0], range=bins.bin_range[0], bins=bins.n_bins[0], weights=values
        )
    elif data.shape[0] == 2:
        hist = fast_histogram.histogram2d(*data, range=bins.bin_range, bins=bins.n_bins)
        hist_w = fast_histogram.histogram2d(
            *data, range=bins.bin_range, bins=bins.n_bins, weights=values
        )
    else:
        raise TypeError("Dimension of data must be 1 or 2.")

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_values: npt.NDArray[np.float64] = np.true_divide(hist_w, hist)
    return mean_values


def _histogram_mean_boost_histogram(
    data: npt.ArrayLike, bins: Bins, values: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Provide histogram with averaged values for all counts in each bin.

    Bins without counts carry NaN.

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
    npt.NDArray[np.float64]
    """
    data = np.asarray(data)
    hist = bh.Histogram(
        *bins.to_boost_histogram_axes(), storage=bh.storage.Mean()
    ).fill(*data, sample=values)
    # bh.Histogram yields zero for mean values in bins with zero counts
    mean_values: npt.NDArray[np.float64] = np.asarray(hist.values(), dtype=np.float64)
    mask = hist.counts() == 0
    mean_values[mask] = np.nan
    return mean_values


class Histogram(NamedTuple):
    data: npt.NDArray[np.float64]
    bins: Bins
    labels: list[str]


def histogram(
    locdata: LocData,
    loc_properties: str | Iterable[str] | None = None,
    other_property: str | None = None,
    bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
    n_bins: NBinsInput | None = None,
    bin_size: BinSizeInput | None = None,
    bin_edges: BinEdgesInput | None = None,
    bin_range: BinRangeInput | Literal["zero", "link"] | None = None,
) -> Histogram:
    """
    Make histogram of loc_properties (columns in `locdata.data`)
    by binning all localizations
    or averaging other_property within each bin.

    Bins are half-open intervals. If `bin_range` is derived from the data,
    the upper bound is widened by a negligible amount (1e-12 of the range)
    so that the localization with the maximum coordinate is counted.

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

    Returns
    -------
    Histogram

    Raises
    ------
    ValueError
        If the bin specification is inconsistent or its dimension does not
        match `loc_properties`.
    TypeError
        If `locdata` is empty, more than three `loc_properties` are given
        or `other_property` is not a valid property.
    """
    labels_ = _check_loc_properties(locdata, loc_properties)
    data = locdata.data[labels_].values.T

    bin_range_: BinRangeInput | None
    if (
        (bin_range is None or isinstance(bin_range, str))
        and bin_edges is None
        and bins is None
    ):
        if bin_range is not None and bin_range not in ("zero", "link"):
            raise ValueError(
                "`bin_range` must be a range, 'zero', 'link' or None, "
                f"got {bin_range!r}."
            )
        special = cast("Literal['zero', 'link'] | None", bin_range)
        bin_range_ = _data_bin_range(locdata, labels_, special=special)
        if bin_range_ is None:
            raise TypeError("`locdata` must contain at least one localization.")
    elif isinstance(bin_range, str):
        raise ValueError(
            "`bin_range` must be None when `bins` or `bin_edges` is given, "
            f"got {bin_range!r}."
        )
    else:
        bin_range_ = bin_range

    bins_ = Bins.from_any(
        bins=bins,
        n_bins=n_bins,
        bin_size=bin_size,
        bin_edges=bin_edges,
        bin_range=bin_range_,
    )
    if bins_.dimension != len(labels_):
        raise ValueError(
            f"The dimension of bins ({bins_.dimension}) does not match the number "
            f"of `loc_properties` ({len(labels_)})."
        )
    bins_ = bins_.with_labels(labels_)

    n_dimensions = data.shape[0]
    if not 1 <= n_dimensions <= 3:
        raise TypeError(
            "loc_properties must contain a string or a list with 1, 2 or 3 elements."
        )
    # fast-histogram is faster than boost-histogram in 2D but requires
    # equally sized bins.
    use_fast_histogram = n_dimensions == 2 and all(bins_.is_equally_sized)

    img: npt.NDArray[np.float64]
    if other_property is None:
        if use_fast_histogram:
            img = _histogram_fast_histogram(data, bins_)
        else:
            img = _histogram_boost_histogram(data, bins_)
        labels_.append("counts")
    elif other_property in locdata.data.columns:
        values = locdata.data[other_property].to_numpy()
        if use_fast_histogram:
            img = _histogram_mean_fast_histogram(data=data, bins=bins_, values=values)
        else:
            img = _histogram_mean_boost_histogram(data=data, bins=bins_, values=values)
        labels_.append(other_property)
    else:
        raise TypeError(
            f"Parameter for `other_property` {other_property} is not a valid property "
            f"name."
        )

    return Histogram(data=img, bins=bins_, labels=labels_)


def _group_by_bin_identifier(
    data: npt.NDArray[Any],
    bin_identifier: npt.NDArray[np.intp],
    return_data: bool,
    return_counts: bool,
) -> tuple[
    npt.NDArray[np.intp],
    list[npt.NDArray[np.intp]],
    list[npt.NDArray[Any]] | None,
    npt.NDArray[np.intp] | None,
]:
    """Group data indices by (multi-dimensional) bin identifier."""
    if bin_identifier.ndim == 1:
        keys = bin_identifier
    else:
        # lexicographic order: first dimension is most significant
        keys = np.zeros(bin_identifier.shape[0], dtype=np.intp)
        for column in bin_identifier.T:
            keys = keys * (int(column.max(initial=0)) + 1) + column
    sorted_indices = np.argsort(keys, kind="stable")
    _, first_positions, n_elements = np.unique(
        keys[sorted_indices], return_index=True, return_counts=True
    )
    bin_indices = bin_identifier[sorted_indices[first_positions]]
    stop_positions = first_positions + n_elements
    data_indices = [
        sorted_indices[start:stop]
        for start, stop in zip(first_positions, stop_positions, strict=True)
    ]
    collection = [data[indices_] for indices_ in data_indices] if return_data else None
    counts = n_elements.astype(np.intp) if return_counts else None
    return bin_indices, data_indices, collection, counts


def _accumulate_1d(
    data: npt.ArrayLike,
    bins: Bins,
    return_data: bool = False,
    return_counts: bool = False,
) -> tuple[
    npt.NDArray[np.intp],
    list[npt.NDArray[np.intp]],
    list[npt.NDArray[Any]] | None,
    npt.NDArray[np.intp] | None,
]:
    """
    Bin data and collect data elements contained in each bin.
    The returned `bin_indices` refer to the given bins including index[0] for
    underflow data and index[n_bins + 1] for overflow data.

    Parameters
    ----------
    data
        Data array of shape (n_points,) or (n_points, dimensions)
        All points are binned with regard to the first dimension.
    bins
        One-dimensional bins.
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
    tuple[npt.NDArray[np.intp], list[npt.NDArray[np.intp]], list[npt.NDArray[Any]] | None, npt.NDArray[np.intp] | None]
        bin_indices, data_indices, collection, counts.
    """
    data_ = np.asarray(data)
    coordinates = data_[:, 0] if data_.ndim > 1 else data_
    bin_identifier = bins.digitize(coordinates)
    return _group_by_bin_identifier(data_, bin_identifier, return_data, return_counts)


def _accumulate_2d(
    data: npt.ArrayLike,
    bins: Bins,
    return_data: bool = False,
    return_counts: bool = False,
) -> tuple[
    npt.NDArray[np.intp],
    list[npt.NDArray[np.intp]],
    list[npt.NDArray[Any]] | None,
    npt.NDArray[np.intp] | None,
]:
    """
    Bin data and collect data elements contained in each bin.
    All points are binned with regard to the first and second dimension.
    The returned `bin_indices` refer to the given bins
    including index[0] for underflow data
    and index[n_bins + 1] for overflow data.

    Parameters
    ----------
    data
        Data array of shape (n_points, dimensions)
    bins
        Two-dimensional bins.
    return_data
        If true, grouped data elements are returned.
    return_counts
        If true, counts (number of elements per bin) are returned.

    Returns
    -------
    tuple[npt.NDArray[np.intp], list[npt.NDArray[np.intp]], list[npt.NDArray[Any]] | None, npt.NDArray[np.intp] | None]
        bin_indices, data_indices, collection, counts.
    """
    data_ = np.asarray(data)
    if data_.size == 0:
        bin_indices = np.empty((0, 2), dtype=np.intp)
        data_indices: list[npt.NDArray[np.intp]] = []
        collection: list[npt.NDArray[Any]] | None = [] if return_data else None
        counts = np.array([], dtype=np.intp) if return_counts else None
        return bin_indices, data_indices, collection, counts
    bin_identifier = bins.digitize(data_[:, :2])
    return _group_by_bin_identifier(data_, bin_identifier, return_data, return_counts)
