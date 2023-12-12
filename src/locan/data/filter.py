"""

Filter localization data.

This module provides functions for filtering LocData objects.
The functions take LocData as input and compute new LocData objects.

"""
from __future__ import annotations

import sys
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

from locan.configuration import N_JOBS
from locan.constants import HullType
from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta
from locan.data.region import Interval, Region, Region2D, RoiRegion
from locan.locan_types import RandomGeneratorSeed

__all__: list[str] = [
    "Selector",
    "filter_condition",
    "select_by_condition",
    "select_by_region",
    "select_by_image_mask",
    "exclude_sparse_points",
    "random_subset",
    "localizations_in_cluster_regions",
]


class Selector:
    """
    Define selection interval for a single localization property.

    Parameters
    ----------
    loc_property
        Localization property
    activate
        Indicator to apply the selection or not
    lower_bound
        min fo selection interval
    upper_bound
        max of selection interval

    Attributes
    ----------
    loc_property : str
        Localization property
    activate : bool
        Indicator to apply the selection or not
    lower_bound : int | float
        min fo selection interval
    upper_bound: int | float
        max of selection interval
    interval : Interval
        Class with interval specifications
    condition : str
        specification turned into condition string
    """

    def __init__(
        self,
        loc_property: str,
        activate: bool,
        lower_bound: int | float,
        upper_bound: int | float,
    ) -> None:
        self.loc_property = loc_property
        self.activate = activate
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self.interval = Interval(lower_bound=lower_bound, upper_bound=upper_bound)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loc_property='{self.loc_property}', "
            f"activate={self.activate}, "
            f"lower_bound={self.lower_bound}, "
            f"upper_bound={self.upper_bound}"
            f")"
        )

    @property
    def lower_bound(self) -> int | float:
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, value: int | float) -> None:
        self._lower_bound = value
        self.interval = Interval(lower_bound=value, upper_bound=self.upper_bound)

    @property
    def upper_bound(self) -> int | float:
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, value: int | float) -> None:
        self._upper_bound = value
        self.interval = Interval(lower_bound=self.lower_bound, upper_bound=value)

    @property
    def condition(self) -> str:
        if self.activate is True:
            condition = f"{self.lower_bound} < {self.loc_property} < {self.upper_bound}"
        else:
            condition = ""
        return condition


def filter_condition(selectors: Iterable[Selector]) -> str:
    """
    Get a condition string from selection specifications.

    Parameters
    ----------
    selectors
        Specifications for loc_property selections

    Returns
    -------
    str
    """
    iterable = (selector.condition for selector in selectors if selector.activate)
    condition = " and ".join(iterable)
    return condition


def select_by_condition(locdata: LocData, condition: str) -> LocData:
    """
    Select by specifying conditions on data properties.

    Parameters
    ----------
    locdata
        Specifying the localization data from which to select.
    condition
        Conditions as input in select method.
        More precise: query specifications to be used with pandas query.

    Returns
    -------
    LocData
        A new instance of LocData referring to the specified dataset.
    """
    local_parameter = locals()

    # select
    new_indices = locdata.data.query(condition).index.values.tolist()

    # instantiate
    new_locdata = LocData.from_selection(locdata=locdata, indices=new_indices)

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata


def select_by_region(
    locdata: LocData,
    region: Region,
    loc_properties: list[str] | None = None,
    reduce: bool = True,
) -> LocData:
    """
    Select localizations from `locdata` that are within `region` and return
    a new LocData object.
    Selection is with respect to loc_properties or to localization coordinates
    that correspond to region dimension.

    Parameters
    ----------
    locdata
        Localization data that is tested for being inside the region.
    region
        Tested region
    loc_properties
        Localization properties to be tested.
    reduce
        Return the reduced LocData object or keep references alive.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations within region of
        interest.

    Note
    ----
    Points on boundary of regions are considered outside if region is a
    shapely object, but inside if region is a
    matplotlib (which is the case for RoiRegion) object.
    """
    local_parameter = locals()

    if loc_properties is None:
        loc_properties_ = locdata.coordinate_keys[0 : region.dimension]
    else:
        loc_properties_ = loc_properties

    points = locdata.data[list(loc_properties_)].values

    if isinstance(region, (Region2D, RoiRegion)):
        indices_inside = region.contains(points)
        locdata_indices_to_keep = locdata.data.index[indices_inside]
        new_locdata = LocData.from_selection(
            locdata=locdata, indices=locdata_indices_to_keep
        )
        new_locdata.region = region
    else:
        raise NotImplementedError("Only Region2D has been implemented.")

    # finish
    if reduce:
        new_locdata.reduce()

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata


# todo: implement
def select_by_image_mask(locdata: LocData, mask: npt.ArrayLike) -> LocData:
    """
    Select by masking using a binary image(e.g. generated by thresholding a
    transmitted-light microscopy image.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data from which to select.
    mask
        binary image.

    Returns
    -------
    LocData
        a new instance of Selection referring to the specified dataset.
    """
    raise NotImplementedError


def exclude_sparse_points(
    locdata: LocData,
    other_locdata: LocData | None = None,
    radius: float = 50,
    min_samples: int = 5,
) -> LocData:
    """
    Exclude localizations by thresholding a local density.

    A subset of localizations, that exhibit a small local density of
    localizations from locdata or alternatively from other_locdata,
    is identified as noise and excluded.
    Noise is identified by using a nearest-neighbor search
    (:class:`sklearn.neighbors.NearestNeighbors`) to find all
    localizations within a circle (sphere) of the given `radius`.
    If the number of localizations is below the
    threshold value `min_samples`, the localization is considered to be noise.

    The method identifies the same noise points as done by the clustering
    algorithm DBSCAN [1]_.

    Parameters
    ----------
    locdata
        Specifying the localization data from which to exclude localization
        data.
    other_locdata
        Specifying the localization data on which to compute local density.
    radius
        Radius of a circle or sphere in which neighbors are identified
        (equivalent to epsilon in DBSCAN).
    min_samples
        The minimum number of samples in the neighborhood that need to be
        found for each localization to not be
        identified as noise (equivalent to minPoints in DBSCAN).

    Returns
    -------
    LocData
        All localizations except those identified as sparse (noise) points.

    References
    ----------
    .. [1] Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu,
       A density-based algorithm for discovering clusters in large spatial
       databases with noise.
       In: Evangelos Simoudis, Jiawei Han, Usama M. Fayyad (Hrsg.):
       Proceedings of the Second International Conference
       on Knowledge Discovery and Data Mining (KDD-96). AAAI Press, 1996,
       S. 226-231, ISBN 1-57735-004-9.
    """
    local_parameter = locals()

    if other_locdata is None:
        nn = NearestNeighbors(metric="euclidean", n_jobs=N_JOBS).fit(
            locdata.coordinates
        )
        neighbor_points_list = nn.radius_neighbors(radius=radius, return_distance=False)
        # if points is not provided the query point is not considered its own neighbor.
    else:
        nn = NearestNeighbors(metric="euclidean", n_jobs=N_JOBS).fit(
            other_locdata.coordinates
        )
        neighbor_points_list = nn.radius_neighbors(
            locdata.coordinates, radius=radius, return_distance=False
        )

    indices_to_keep = [len(pts) >= min_samples for pts in neighbor_points_list]
    locdata_indices_to_keep = locdata.data.index[indices_to_keep]
    new_locdata = LocData.from_selection(locdata, locdata_indices_to_keep)

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata


def random_subset(
    locdata: LocData,
    n_points: int,
    replace: bool = True,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Take a random subset of localizations.

    Parameters
    ----------
    locdata
        Specifying the localization data from which to select localization data.
    n_points
        Number of localizations to randomly choose from locdata.
    replace
        Indicate if sampling is with or without replacement
    seed
        Random number generation seed

    Returns
    -------
    LocData
        A new instance of LocData carrying the subset of localizations.
    """
    local_parameter = locals()

    if not locdata:
        return locdata

    rng = np.random.default_rng(seed)

    indices = rng.choice(locdata.data.index, size=n_points, replace=replace)
    new_locdata = LocData.from_selection(locdata, indices)

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata = LocData.from_selection(locdata, indices, meta=meta_)

    return new_locdata


def localizations_in_cluster_regions(
    locdata: LocData,
    collection: LocData | list[LocData],
    hull_type: HullType | str = HullType.CONVEX_HULL,
) -> LocData:
    """
    Identify localizations from `locdata` within the regions of all
    `collection` elements.

    Parameters
    ----------
    locdata
        Localization data that is tested for being inside the region
    collection
        A set of Locdata objects collected in a collection or list.
    hull_type
        The hull type for each LocData object that is used to define the
        region.

    Returns
    --------
    LocData
        A collection of LocData objects with all elements of locdata
        contained by the region.
    """
    locdatas = []
    if isinstance(hull_type, str):
        hull_type = HullType[hull_type.upper()].value
    else:
        hull_type = hull_type.value

    if isinstance(collection, LocData):
        if isinstance(collection.references, list):  # this case covers pure collections
            for ref in collection.references:
                cregion = getattr(ref, hull_type).region
                locdata_selection = select_by_region(locdata=locdata, region=cregion)
                locdatas.append(locdata_selection)
        else:  # this case covers selections of collections
            for index in collection.indices:  # type: ignore
                cregion = getattr(
                    collection.references.references[index], hull_type  # type: ignore
                ).region
                locdata_selection = select_by_region(locdata=locdata, region=cregion)
                locdatas.append(locdata_selection)
    else:  # this case covers list of LocData objects
        for ref in collection:
            cregion = getattr(ref, hull_type).region
            locdata_selection = select_by_region(locdata=locdata, region=cregion)
            locdatas.append(locdata_selection)

    new_collection = LocData.from_collection(locdatas)

    return new_collection
