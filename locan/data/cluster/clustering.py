"""

Methods for clustering localization data in LocData objects.

"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN

from locan.configuration import N_JOBS
from locan.data.aggregate import Bins, _accumulate_2d
from locan.data.locdata import LocData
from locan.data.locdata_utils import _check_loc_properties
from locan.data.properties.locdata_statistics import ranges
from locan.dependencies import HAS_DEPENDENCY, needs_package

if HAS_DEPENDENCY["hdbscan"]:
    from hdbscan import HDBSCAN

if TYPE_CHECKING:
    import boost_histogram as bh

__all__: list[str] = ["cluster_hdbscan", "cluster_dbscan", "cluster_by_bin"]


@needs_package("hdbscan")
def cluster_hdbscan(
    locdata: LocData,
    min_cluster_size: int = 5,
    loc_properties: list[str] | None = None,
    allow_single_cluster: bool = False,
    **kwargs: Any,
) -> tuple[LocData, LocData]:
    """
    Cluster localizations in locdata using the hdbscan clustering algorithm.

    Parameters
    ----------
    locdata
        Localization data on which to perform the manipulation.
    loc_properties
        The LocData properties to be used for clustering.
        If None, `locdata.coordinates` will be used.
    min_cluster_size
        Minimumm cluster size in HDBSCAN algorithm (default: 5)
    allow_single_cluster
        If True, return single cluster (default: False)
    kwargs
        Other parameters passed to `hdbscan.HDBSCAN`.

    Returns
    -------
    tuple[LocData, LocData]
        A tuple with noise and cluster.
        The first LocData object is a selection of all localizations that are
        defined as noise, in other words all localizations that are not part
        of any cluster.
        The second LocData object is a LocData instance assembling all
        generated selections (i.e. localization cluster).
    """
    parameter = locals()

    if len(locdata) == 0:
        locdata_noise = LocData()
        collection = LocData()

    if len(locdata) < min_cluster_size:
        locdata_noise = copy(locdata)
        collection = LocData()

    else:
        if loc_properties is None:
            fit_data = locdata.coordinates
        else:
            fit_data = locdata.data[loc_properties].to_numpy()

        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            allow_single_cluster=allow_single_cluster,
            gen_min_span_tree=False,
            **kwargs,
        ).fit_predict(fit_data)

        grouped = locdata.data.groupby(labels)
        locdata_index_labels = [
            locdata.data.index[idxs] for idxs in grouped.indices.values()
        ]
        selections = [
            LocData.from_selection(locdata=locdata, indices=idxs)
            for idxs in locdata_index_labels
        ]

        try:
            grouped.get_group(-1)
            locdata_noise = selections[0]
            collection = LocData.from_collection(selections[1:])
        except KeyError:
            locdata_noise = LocData()
            collection = LocData.from_collection(selections)

    # set regions
    if locdata_noise:
        locdata_noise.region = locdata.region
    if collection:
        collection.region = locdata.region

    # metadata
    if locdata_noise:
        del locdata_noise.meta.history[:]
        locdata_noise.meta.history.add(
            name=sys._getframe().f_code.co_name, parameter=str(parameter)
        )
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return locdata_noise, collection


def cluster_dbscan(
    locdata: LocData,
    eps: float = 20,
    min_samples: int = 5,
    loc_properties: list[str] | None = None,
    **kwargs: Any,
) -> tuple[LocData, LocData]:
    """
    Cluster localizations in locdata using the dbscan clustering algorithm as
    implemented in sklearn.

    Parameters
    ----------
    locdata
        specifying the localization data on which to perform the manipulation.
    eps
        The maximum distance between two samples for them to be considered as
        in the same neighborhood.
    min_samples
        The number of samples in a neighborhood for a point to be considered
        as a core point.
        This includes the point itself.
    loc_properties
        The LocData properties to be used for clustering. If None,
        `locdata.coordinates` will be used.
    kwargs
        Other parameters passed to `sklearn.cluster.DBSCAN`.

    Returns
    -------
    tuple[LocData, LocData]
        A tuple with noise and cluster.
        The first LocData object is a selection of all localizations that are
        defined as noise, in other words all localizations that are not part
        of any cluster.
        The second LocData object is a LocData instance assembling all
        generated selections (i.e. localization cluster).
    """
    parameter = locals()

    if len(locdata) == 0:
        locdata_noise = LocData()
        collection = LocData()

    else:
        if loc_properties is None:
            fit_data = locdata.coordinates
        else:
            fit_data = locdata.data[loc_properties].to_numpy()

        labels = DBSCAN(
            eps=eps, min_samples=min_samples, n_jobs=N_JOBS, **kwargs
        ).fit_predict(fit_data)

        grouped = locdata.data.groupby(labels)
        locdata_index_labels = [
            locdata.data.index[idxs] for idxs in grouped.indices.values()
        ]
        selections = [
            LocData.from_selection(locdata=locdata, indices=idxs)
            for idxs in locdata_index_labels
        ]

        try:
            grouped.get_group(-1)
            locdata_noise = selections[0]
            collection = LocData.from_collection(selections[1:])
        except KeyError:
            locdata_noise = LocData()
            collection = LocData.from_collection(selections)

    # set regions
    if locdata_noise:
        locdata_noise.region = locdata.region
    if collection:
        collection.region = locdata.region

    # metadata
    if locdata_noise:
        del locdata_noise.meta.history[:]
        locdata_noise.meta.history.add(
            name=sys._getframe().f_code.co_name, parameter=str(parameter)
        )
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return locdata_noise, collection


def cluster_by_bin(
    locdata: LocData,
    loc_properties: list[str] | None = None,
    min_samples: int = 1,
    bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
    n_bins: int | Sequence[int] | None = None,
    bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_range: tuple[float, float]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | str
    | None = None,
    return_counts: bool = False,
) -> tuple[Bins | None, npt.NDArray[np.int_], LocData, npt.NDArray[np.int_] | None]:
    """
    Cluster localizations in locdata by binning all localizations with regard
    to `loc_properties` and collecting all localizations per bin as cluster.

    Parameters
    ----------
    locdata
        Localization data.
    loc_properties
        Localization properties to be grouped into bins.
        If None The coordinate_values of locdata are used.
    min_samples
        The minimum number of samples per bin to be considered as cluster.
    bins
        The bin specification as defined in :class:`Bins`
    bin_edges
        Array of bin edges with shape (n_bin_edges,)
        or (dimension, n_bin_edges) for all or each dimension.
    n_bins
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size
        The size of bins in units of locdata coordinate units for all or each
        dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        To specify arbitrary sequence of `bin_sizes` use `bin_edges` instead.
    bin_range : tuple[float, ...] | tuple[tuple[float, float], ...] | Literal['zero'] | None
        The data bin_range to be taken into consideration for all or each
        dimension.
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    return_counts
        If true, n_elements per bin are returned.

    Returns
    -------
    tuple[Bins | None, npt.NDArray[np.int_], LocData, npt.NDArray[np.int_] | None]
        Tuple with bins, bin_indices,
        collection of all generated selections (i.e. localization clusters),
        and counts per bin.
    """
    parameter = locals()

    if len(locdata) == 0:
        bins = None
        bin_indices = np.array([])
        collection = LocData()
        counts = np.array([]) if return_counts else None
        return bins, bin_indices, collection, counts

    loc_properties = _check_loc_properties(locdata, loc_properties)
    data = locdata.data[loc_properties].values
    if (bin_range is None or isinstance(bin_range, str)) and bin_edges is None:
        bin_range_ = ranges(locdata, loc_properties=loc_properties, special=bin_range)  # type: ignore
    else:
        bin_range_ = bin_range  # type: ignore

    try:
        bins = Bins(
            bins=bins,
            n_bins=n_bins,
            bin_size=bin_size,
            bin_edges=bin_edges,
            bin_range=bin_range_,  # type: ignore
            labels=loc_properties,
        )
    except ValueError as exc:
        raise ValueError(
            "Bin dimension and len of `loc_properties` is incompatible."
        ) from exc

    bin_indices, data_indices, _, counts = _accumulate_2d(
        data, bin_edges=bins.bin_edges, return_counts=True
    )
    assert counts is not None  # type narrowing # noqa: S101

    if min_samples > 1:
        mask = counts >= min_samples
        counts = counts[mask]
        bin_indices = bin_indices[mask]
        data_indices = [
            data_indices_e
            for data_indices_e, mask_e in zip(data_indices, mask)
            if mask_e
        ]

    selections = [
        LocData.from_selection(locdata=locdata, indices=idxs) for idxs in data_indices
    ]
    collection = LocData.from_collection(selections)

    # set regions
    if collection:
        collection.region = locdata.region

    # metadata
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    if not return_counts:
        counts = None

    return bins, bin_indices, collection, counts
