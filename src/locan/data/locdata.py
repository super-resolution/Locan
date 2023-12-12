"""

A class to carry localization data.

"""
from __future__ import annotations

import copy
import logging
import os
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from itertools import accumulate
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from google.protobuf import json_format

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError  # needed for Python 3.7

import locan.data.hulls
from locan import (  # is required to use locdata_id as global variable  # noqa: F401
    locdata_id,
)
from locan.constants import PROPERTY_KEYS, PropertyKey
from locan.data import metadata_pb2
from locan.data.locdata_utils import _dataframe_to_pandas, _get_linked_coordinates
from locan.data.metadata_utils import (
    _modify_meta,
    merge_metadata,
    metadata_to_formatted_string,
)
from locan.data.region import Region, RoiRegion

if TYPE_CHECKING:
    from locan.locan_types import DataFrame  # noqa F401

__all__: list[str] = ["LocData"]

logger = logging.getLogger(__name__)

T_LocData = TypeVar("T_LocData", bound="LocData")


class LocData:
    """
    This class carries localization data, aggregated properties and meta data.

    Data consist of individual elements being either localizations or other
    `LocData` objects.
    Both, localizations and `Locdata` objects have properties.
    Properties come from the original data or are added by analysis procedures.

    Parameters
    ----------
    references
        A `LocData` reference or an array with references to `LocData` objects
        referring to the selected localizations in dataset.
    dataframe
        Dataframe with localization data.
    indices
        Indices for dataframe in references that makes up the data.
        `indices` refers to index label, not position.
    meta
        Metadata about the current dataset and its history.

    Attributes
    ----------
    references : LocData | list[LocData] | None
        A LocData reference or an array with references to LocData objects
        referring to the selected localizations in dataframe.
    dataframe : pandas.DataFrame
        Dataframe with localization data.
    indices : slice | list[int] | None
        Indices for dataframe in references that makes up the data.
    meta : locan.data.metadata_pb2.Metadata
        Metadata about the current dataset and its history.
    properties : dict[str, Any]
        List of properties generated from data.
    coordinate_keys : list[str]
        The available coordinate properties.
    uncertainty_keys : list[str]
        The available uncertainty properties.
    dimension : int
        Number of coordinates available for each localization
        (i.e. size of `coordinate_keys`).
    """

    count = 0
    """int: A counter for counting LocData instantiations (class attribute)."""

    def __init__(
        self,
        references: LocData | Iterable[LocData] | None = None,
        dataframe: pd.DataFrame | None = None,
        indices: int
        | list[int | bool]
        | npt.NDArray[np.int_ | np.bool_]
        | slice
        | pd.Index[int]
        | None = None,
        meta: metadata_pb2.Metadata
        | dict[str, Any]
        | str
        | bytes
        | os.PathLike[Any]
        | BinaryIO
        | None = None,
    ):
        self.__class__.count += 1

        self.references: LocData | Iterable[LocData] | None = references
        self.dataframe: pd.DataFrame = (
            pd.DataFrame() if dataframe is None else dataframe
        )
        self.indices: int | list[int | bool] | npt.NDArray[
            np.int_ | np.bool_
        ] | slice | pd.Index[int] | None = indices
        self.meta: metadata_pb2.Metadata = metadata_pb2.Metadata()
        self.properties: dict[str, Any] = {}

        # regions and hulls
        self._region: Region | None = None
        self._bounding_box: locan.data.hulls.BoundingBox | None = None
        self._oriented_bounding_box: locan.data.hulls.OrientedBoundingBox | None = None
        self._convex_hull: locan.data.hulls.ConvexHull | None = None
        self._alpha_shape: locan.data.hulls.AlphaShape | None = None
        self._inertia_moments: locan.data.properties.misc.InertiaMoments | None = None

        self.dimension: int = len(self.coordinate_keys)

        self._update_properties()

        # meta
        global locdata_id
        locdata_id += 1
        self.meta.identifier = str(locdata_id)

        self.meta.creation_time.GetCurrentTime()
        self.meta.source = metadata_pb2.DESIGN
        self.meta.state = metadata_pb2.RAW

        self.meta.element_count = len(self.data.index)
        if "frame" in self.data.columns:
            self.meta.frame_count = len(self.data["frame"].unique())

        self.meta = merge_metadata(metadata=self.meta, other_metadata=meta)

    @property
    def coordinate_keys(self) -> list[str]:
        return [
            label_
            for label_ in PropertyKey.coordinate_keys()
            if label_ in self.data.columns
        ]

    @property
    def uncertainty_keys(self) -> list[str]:
        return [
            label_
            for label_ in PropertyKey.uncertainty_keys()
            if label_ in self.data.columns
        ]

    def _update_properties(
        self, update_function: dict[str, Callable[..., Any]] | None = None
    ) -> Self:
        """
        Compute properties from localization data.

        For each loc_property in update_function the supplied callable is used.
        If None the following functions are used if loc_properties are available:

        coordinates and corresponding uncertainties: weighted_mean_variance
        `intensity`: sum
        `local_background`: mean
        `frame`: min

        Parameters
        ----------
        update_function
            mapping of localization property onto callable to compute
            property from corresponding localization data

        Returns
        -------
        Self
        """
        self.properties = dict()
        self.properties["localization_count"] = len(self.data.index)

        properties_for_update = [
            loc_property_
            for loc_property_ in [
                *self.coordinate_keys,
                "frame",
                "intensity",
                "local_background",
            ]
            if loc_property_ in self.data.columns
        ]

        if update_function is not None:
            properties_for_update = [
                loc_property_
                for loc_property_ in properties_for_update
                if loc_property_ not in update_function.keys()
            ]

        # localization coordinates
        if all(c_label_ in properties_for_update for c_label_ in self.coordinate_keys):
            self.properties.update(_get_linked_coordinates(locdata=self.data))

        if "intensity" in properties_for_update:
            self.properties["intensity"] = np.sum(self.data["intensity"])

        if "local_background" in properties_for_update:
            self.properties["local_background"] = np.mean(self.data["local_background"])

        if "frame" in properties_for_update:
            self.properties["frame"] = np.min(self.data["frame"])

        if update_function is not None:
            for loc_property_, function_ in update_function.items():
                self.properties[loc_property_] = function_(self.data[loc_property_])

        self.bounding_box  # update self._bounding_box  # noqa B018
        return self

    def __del__(self) -> None:
        """Updating the counter upon deletion of class instance."""
        self.__class__.count -= 1

    def __len__(self) -> int:
        """
        Return the length of data, i.e. the number of elements
        (localizations or collection elements).
        """
        return len(self.data.index)

    def __getstate__(self) -> dict[str, Any]:
        """Modify pickling behavior."""
        # Copy the object's state from self.__dict__ to avoid modifying the original state.
        state = self.__dict__.copy()
        # Serialize the unpicklable protobuf entries.
        json_string = json_format.MessageToJson(
            self.meta, including_default_value_fields=False
        )
        state["meta"] = json_string
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Modify pickling behavior."""
        # Restore instance attributes.
        self.__dict__.update(state)
        # Restore protobuf class for meta attribute
        self.meta = metadata_pb2.Metadata()
        self.meta = json_format.Parse(state["meta"], self.meta)

    def __copy__(self) -> LocData:
        """
        Create a shallow copy of locdata (keeping all references) with the
        following exceptions:
        (i) The class variable `count` is increased for the copied LocData
        object.
        (ii) Metadata keeps the original meta.creation_time while
        meta.modification_time and meta.history is updated.
        """
        new_locdata = LocData(self.references, self.dataframe, self.indices, meta=None)
        new_locdata._region = self._region
        # meta
        meta_ = _modify_meta(
            self, new_locdata, function_name="LocData.copy", parameter=None, meta=None
        )
        new_locdata.meta = meta_
        return new_locdata

    def __deepcopy__(self, memodict: dict[Any, Any] | None = None) -> LocData:
        """
        Create a deep copy of locdata (including all references) with the
        following exceptions:
        (i) The class variable `count` is increased for all deepcopied LocData
        objects.
        (ii) Metadata keeps the original meta.creation_time while
        meta.modification_time and meta.history is updated.
        """
        if memodict is None:
            memodict = {}
        new_locdata = LocData(
            copy.deepcopy(self.references, memodict),
            copy.deepcopy(self.dataframe, memodict),
            copy.deepcopy(self.indices, memodict),
            meta=None,
        )
        new_locdata._region = self._region
        # meta
        meta_ = _modify_meta(
            self,
            new_locdata,
            function_name="LocData.deepcopy",
            parameter=None,
            meta=None,
        )
        new_locdata.meta = meta_
        return new_locdata

    @property
    def bounding_box(self) -> locan.data.hulls.BoundingBox | None:
        """
        Hull object: Return an object representing the axis-aligned
        minimal bounding box.
        """
        if self._bounding_box is None:
            try:
                self._bounding_box = locan.data.hulls.BoundingBox(self.coordinates)
            except ValueError:
                warnings.warn(
                    "Properties related to bounding box could not be computed.",
                    UserWarning,
                    stacklevel=1,
                )
        self._update_properties_bounding_box()
        return self._bounding_box

    def _update_properties_bounding_box(self) -> None:
        if self._bounding_box is not None:
            self.properties["region_measure_bb"] = self._bounding_box.region_measure
            if self._bounding_box.region_measure:
                self.properties["localization_density_bb"] = (
                    self.properties["localization_count"]
                    / self._bounding_box.region_measure
                )
            if self._bounding_box.subregion_measure:
                self.properties[
                    "subregion_measure_bb"
                ] = self._bounding_box.subregion_measure

    @property
    def convex_hull(self) -> locan.data.hulls.ConvexHull | None:
        """
        Hull object: Return an object representing the convex hull of all
        localizations.
        """
        if self._convex_hull is None:
            try:
                self._convex_hull = locan.data.hulls.ConvexHull(self.coordinates)
            except (TypeError, QhullError):
                warnings.warn(
                    "Properties related to convex hull could not be computed.",
                    UserWarning,
                    stacklevel=1,
                )
        self._update_properties_convex_hull()
        return self._convex_hull

    def _update_properties_convex_hull(self) -> None:
        if self._convex_hull is not None:
            self.properties["region_measure_ch"] = self._convex_hull.region_measure
            if self._convex_hull.region_measure:
                self.properties["localization_density_ch"] = (
                    self.properties["localization_count"]
                    / self._convex_hull.region_measure
                )
            if self._convex_hull.subregion_measure:
                self.properties[
                    "subregion_measure_ch"
                ] = self._convex_hull.subregion_measure

    @property
    def oriented_bounding_box(self) -> locan.data.hulls.OrientedBoundingBox | None:
        """
        Hull object: Return an object representing the oriented minimal
        bounding box.
        """
        if self._oriented_bounding_box is None:
            try:
                self._oriented_bounding_box = locan.data.hulls.OrientedBoundingBox(
                    self.coordinates
                )
            except TypeError:
                warnings.warn(
                    "Properties related to oriented bounding box could not be computed.",
                    UserWarning,
                    stacklevel=1,
                )
        self._update_properties_oriented_bounding_box()
        return self._oriented_bounding_box

    def _update_properties_oriented_bounding_box(self) -> None:
        if self._oriented_bounding_box is not None:
            self.properties[
                "region_measure_obb"
            ] = self._oriented_bounding_box.region_measure
            if self._oriented_bounding_box.region_measure:
                self.properties["localization_density_obb"] = (
                    self.properties["localization_count"]
                    / self._oriented_bounding_box.region_measure
                )
            self.properties["orientation_obb"] = self._oriented_bounding_box.angle
            self.properties["circularity_obb"] = self._oriented_bounding_box.elongation

    @property
    def alpha_shape(self) -> locan.data.hulls.AlphaShape | None:
        """
        Hull object: Return an object representing the alpha-shape of all
        localizations.
        """
        return self._alpha_shape

    def update_alpha_shape(self, alpha: float) -> Self:
        """
        Compute the alpha shape for specific `alpha` and update
        `self.alpha_shape`.

        Parameters
        ----------
        alpha
            Alpha parameter specifying a unique alpha complex.

        Returns
        -------
        Self
            The modified object
        """
        try:
            if self._alpha_shape is None:
                self._alpha_shape = locan.data.hulls.AlphaShape(
                    points=self.coordinates, alpha=alpha
                )
            else:
                self._alpha_shape.alpha = alpha
        except TypeError:
            warnings.warn(
                "Properties related to alpha shape could not be computed.",
                UserWarning,
                stacklevel=1,
            )
        self._update_properties_alpha_shape()
        return self

    def _update_properties_alpha_shape(self) -> None:
        if self._alpha_shape is not None:
            self.properties["region_measure_as"] = self._alpha_shape.region_measure
            try:
                self.properties["localization_density_as"] = (
                    self._alpha_shape.n_points_alpha_shape
                    / self._alpha_shape.region_measure
                )
            except ZeroDivisionError:
                self.properties["localization_density_as"] = float("nan")

    def update_alpha_shape_in_references(self, alpha: float) -> Self:
        """
        Compute the alpha shape for each element in `locdata.references` and
        update `locdata.dataframe`.

        Parameters
        ----------
        alpha
            Alpha parameter specifying a unique alpha complex.

        Returns
        -------
        Self
            The modified object
        """
        if isinstance(self.references, list):
            for reference in self.references:
                reference.update_alpha_shape(alpha=alpha)
            new_df = pd.DataFrame(
                [reference.properties for reference in self.references]
            )
            new_df.index = self.data.index
            if self.dataframe is None:
                self.dataframe = new_df
            else:
                self.dataframe.update(new_df)
            new_columns = [
                column for column in new_df.columns if column in self.dataframe.columns
            ]
            new_df.drop(columns=new_columns, inplace=True, errors="ignore")
            self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
        return self

    @property
    def inertia_moments(self) -> locan.data.properties.misc.InertiaMoments | None:
        """
        Inertia moments are returned as computed by
        :func:`locan.data.properties.inertia_moments`.
        """
        if self._inertia_moments is None:
            try:
                self._inertia_moments = locan.data.properties.inertia_moments(
                    self.coordinates
                )
            except TypeError:
                warnings.warn(
                    "Properties related to inertia_moments could not be computed.",
                    UserWarning,
                    stacklevel=1,
                )
        self._update_properties_inertia_moments()
        return self._inertia_moments

    def _update_properties_inertia_moments(self) -> None:
        if self._inertia_moments is not None:
            self.properties["orientation_im"] = self._inertia_moments.orientation
            self.properties["circularity_im"] = self._inertia_moments.eccentricity

    def update_inertia_moments_in_references(self) -> Self:
        """
        Compute inertia_moments for each element in locdata.references and
        update locdata.dataframe.

        Returns
        -------
        Self
            The modified object
        """
        if isinstance(self.references, list):
            for reference in self.references:
                reference.inertia_moments  # request property to update  # noqa B018
            new_df = pd.DataFrame(
                [reference.properties for reference in self.references]
            )
            new_df.index = self.data.index
            if self.dataframe is None:
                self.dataframe = new_df
            else:
                self.dataframe.update(new_df)
            new_columns = [
                column for column in new_df.columns if column in self.dataframe.columns
            ]
            new_df.drop(columns=new_columns, inplace=True, errors="ignore")
            self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
        return self

    @property
    def region(self) -> Region | None:
        """RoiRegion object: Return the region that supports all localizations."""
        return self._region

    @region.setter
    def region(self, region: Region | None) -> None:
        if region is not None:
            if region.dimension != self.dimension:
                raise TypeError(
                    "Region dimension and coordinates dimension must be identical."
                )
            elif len(self) != len(region.contains(self.coordinates)):
                logger.warning("Not all coordinates are within region.")

        if isinstance(region, (Region, RoiRegion)) or region is None:
            self._region = region

        elif isinstance(
            region, dict
        ):  # legacy code to deal with deprecated RoiLegacy_0
            region_ = RoiRegion(**region)
            if region_ is not None:
                if region_.dimension != self.dimension:
                    raise TypeError(
                        "Region dimension and coordinates dimension must be identical."
                    )
                elif len(self) != len(region_.contains(self.coordinates)):
                    logger.warning("Not all coordinates are within region.")
            self._region = region_

        else:
            raise TypeError

        # property for region measures
        if self._region is not None:
            if self._region.region_measure:
                self.properties["region_measure"] = self._region.region_measure
                self.properties["localization_density"] = (
                    self.meta.element_count / self._region.region_measure
                )
            if self._region.subregion_measure:
                self.properties["subregion_measure"] = self._region.subregion_measure

    @property
    def data(self) -> pd.DataFrame:
        """
        pandas.DataFrame: Return all elements either copied from the reference
        or referencing the current dataframe.
        """
        if isinstance(self.references, LocData):
            # we refer to the localization data by its index label, not position
            # in other words we decided not to use iloc but loc
            # df = self.references.data.loc[self.indices]  ... but this does not work in pandas.
            # also see:
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
            try:
                df = self.references.data.loc[self.indices]  # type: ignore
            except KeyError:
                df = self.references.data.loc[
                    self.references.data.index.intersection(self.indices)  # type: ignore
                ]
            df = pd.merge(
                df, self.dataframe, left_index=True, right_index=True, how="outer"
            )
            return df
        else:
            return self.dataframe

    @property
    def coordinates(self) -> npt.NDArray[np.float_]:
        """npt.NDArray[float]: Return all coordinate values."""
        return_value: npt.NDArray[np.float_] = self.data[self.coordinate_keys].values
        return return_value

    @property
    def centroid(self) -> npt.NDArray[np.float_]:
        """
        npt.NDArray[np.float_]: Return coordinate values of the centroid
        (being the property values for all coordinate labels).
        """
        return np.array(
            [
                self.properties[coordinate_label]
                for coordinate_label in self.coordinate_keys
            ]
        )

    @classmethod
    def from_dataframe(
        cls: type[T_LocData],  # noqa: UP006
        dataframe: DataFrame | None = None,
        meta: metadata_pb2.Metadata
        | dict[str, Any]
        | str
        | bytes
        | os.PathLike[Any]
        | BinaryIO
        | None = None,
    ) -> T_LocData:
        """
        Create new LocData object from DataFrame with localization data.

        Parameters
        ----------
        dataframe
            Localization data.
        meta
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with dataframe representing the
            concatenated data.
        """
        dataframe = _dataframe_to_pandas(dataframe=dataframe, allow_copy=True)
        dataframe = pd.DataFrame() if dataframe is None else dataframe
        meta_ = metadata_pb2.Metadata()

        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.RAW
        meta_.history.add(name="LocData.from_dataframe")

        meta_ = merge_metadata(metadata=meta_, other_metadata=meta)

        return cls(dataframe=dataframe, meta=meta_)

    @classmethod
    def from_coordinates(
        cls: type[T_LocData],  # noqa: UP006
        coordinates: npt.ArrayLike | None = None,
        coordinate_labels: Sequence[str] | None = None,
        meta: metadata_pb2.Metadata
        | dict[str, Any]
        | str
        | bytes
        | os.PathLike[Any]
        | BinaryIO
        | None = None,
    ) -> T_LocData:
        """
        Create new LocData object from a sequence of localization coordinates.

        Parameters
        ----------
        coordinates
            Sequence of tuples with localization coordinates
            with shape (n_loclizations, dimension)
        coordinate_labels
            The available coordinate properties.
        meta
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with dataframe representing the
            oncatenated data.
        """
        if coordinates is None:
            coordinates = np.array([])
        else:
            coordinates = np.asarray(coordinates)

        if np.size(coordinates):
            dimension = len(coordinates[0])

            if coordinate_labels is None:
                coordinate_labels = ["position_x", "position_y", "position_z"][
                    0:dimension
                ]
            else:
                if all(cl in PROPERTY_KEYS for cl in coordinate_labels):
                    coordinate_labels = coordinate_labels
                else:
                    raise ValueError(
                        "The given coordinate_keys are not standard property keys."
                    )

            dataframe = pd.DataFrame.from_records(
                data=coordinates, columns=coordinate_labels
            )

        else:
            dataframe = pd.DataFrame()

        meta_ = metadata_pb2.Metadata()
        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.RAW
        meta_.history.add(name="LocData.from_coordinates")

        meta_ = merge_metadata(metadata=meta_, other_metadata=meta)

        return cls(dataframe=dataframe, meta=meta_)

    @classmethod
    def from_selection(
        cls: type[T_LocData],  # noqa: UP006
        locdata: LocData,
        indices: int
        | list[int | bool]
        | npt.NDArray[np.int_ | np.bool_]
        | slice
        | pd.Index[int]
        | None = None,
        meta: metadata_pb2.Metadata
        | dict[str, Any]
        | str
        | bytes
        | os.PathLike[Any]
        | BinaryIO
        | None = None,
    ) -> T_LocData:
        """
        Create new LocData object from selected elements in another `LocData`.

        Parameters
        ----------
        locdata
            Locdata object from which to select elements.
        indices
            Index labels for elements in locdata that make up the new data.
            Note that contrary to usual python slices, both the start and the
            stop are included (see pandas documentation).
            `Indices` refer to index value not position in list.
        meta
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with dataframe representing the selected
            data.

        Note
        ----
        No error is raised if indices do not exist in locdata.
        """
        references = locdata
        if indices is None:
            indices = slice(0, None)

        meta_ = metadata_pb2.Metadata()
        meta_.CopyFrom(locdata.meta)
        try:
            meta_.ClearField("identifier")
        except ValueError:
            pass

        try:
            meta_.ClearField("element_count")
        except ValueError:
            pass

        try:
            meta_.ClearField("frame_count")
        except ValueError:
            pass

        meta_.modification_time.GetCurrentTime()
        meta_.state = metadata_pb2.MODIFIED
        meta_.ancestor_identifiers.append(locdata.meta.identifier)
        meta_.history.add(name="LocData.from_selection")

        meta_ = merge_metadata(metadata=meta_, other_metadata=meta)

        new_locdata = cls(references=references, indices=indices, meta=meta_)
        new_locdata.region = references.region
        return new_locdata

    @classmethod
    def from_collection(
        cls: type[T_LocData],  # noqa: UP006v
        locdatas: Iterable[LocData],
        meta: metadata_pb2.Metadata
        | dict[str, Any]
        | str
        | bytes
        | os.PathLike[Any]
        | BinaryIO
        | None = None,
    ) -> T_LocData:
        """
        Create new LocData object by collecting LocData objects.

        Parameters
        ----------
        locdatas
            Locdata objects to collect.
        meta
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with dataframe representing the concatenated data.
        """
        references = locdatas
        dataframe = pd.DataFrame([ref.properties for ref in references])

        meta_ = metadata_pb2.Metadata()

        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.RAW
        meta_.ancestor_identifiers[:] = [ref.meta.identifier for ref in references]
        meta_.history.add(name="LocData.from_collection")

        meta_ = merge_metadata(metadata=meta_, other_metadata=meta)

        return cls(references=references, dataframe=dataframe, meta=meta_)

    @classmethod
    def concat(
        cls: type[T_LocData],  # noqa: UP006
        locdatas: Iterable[LocData],
        meta: metadata_pb2.Metadata
        | dict[str, Any]
        | str
        | bytes
        | os.PathLike[Any]
        | BinaryIO
        | None = None,
    ) -> T_LocData:
        """
        Concatenate LocData objects.

        Parameters
        ----------
        locdatas
            Locdata objects to concatenate.
        meta
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with dataframe representing the concatenated data.
        """

        dataframe = pd.concat([i.data for i in locdatas], ignore_index=True, sort=False)

        # concatenate references also if None
        references: list[LocData] = []
        for locdata in locdatas:
            try:
                references.extend(locdata.references)  # type: ignore
            except TypeError:
                references.append(locdata.references)  # type: ignore

        # check if all elements are None
        new_references = None if not any(references) else references

        meta_ = metadata_pb2.Metadata()

        meta_.creation_time.GetCurrentTime()
        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.MODIFIED
        meta_.ancestor_identifiers[:] = [dat.meta.identifier for dat in locdatas]
        meta_.history.add(name="concat")

        meta_ = merge_metadata(metadata=meta_, other_metadata=meta)

        return cls(references=new_references, dataframe=dataframe, meta=meta_)

    @classmethod
    def from_chunks(
        cls: type[T_LocData],  # noqa: UP006
        locdata: LocData,
        chunks: Sequence[tuple[int, ...]] | None = None,
        chunk_size: int | None = None,
        n_chunks: int | None = None,
        order: Literal["successive", "alternating"] = "successive",
        drop: bool = False,
        meta: metadata_pb2.Metadata
        | dict[str, Any]
        | str
        | bytes
        | os.PathLike[Any]
        | BinaryIO
        | None = None,
    ) -> T_LocData:
        """
        Divide locdata in chunks of localization elements.

        Parameters
        ----------
        locdata
            Locdata to divide.
        chunks
            Localization chunks as defined by a list of index-tuples.
            One of `chunks`, `chunk_size` or `n_chunks` must be different
            from None.
        chunk_size
            Number of consecutive localizations to form a single chunk of data.
            One of `chunks`, `chunk_size` or `n_chunks` must be different
            from None.
        n_chunks
            Number of chunks.
            One of `chunks`, `chunk_size` or `n_chunks` must be different
            from None.
        order
            The order in which to select localizations.
            One of 'successive' or 'alternating'.
        drop
            If True the last chunk will be eliminated if it has fewer
            localizations than the other chunks.
        meta
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with references and dataframe elements
            representing the individual chunks.
        """
        n_nones = sum(element is None for element in [chunks, chunk_size, n_chunks])

        if n_nones != 2:
            raise ValueError(
                "One and only one of `chunks`, `chunk_size` or `n_chunks` must "
                "be different from None."
            )
        elif chunks is not None:
            index_lists = list(chunks)
        else:
            if chunk_size is not None:
                if (len(locdata) % chunk_size) == 0:
                    n_chunks = len(locdata) // chunk_size
                else:
                    n_chunks = len(locdata) // chunk_size + 1
            else:  # if n_chunks is not None
                assert n_chunks is not None  # type narrowing # noqa: S101
                if (len(locdata) % n_chunks) == 0:
                    chunk_size = len(locdata) // n_chunks
                else:
                    chunk_size = len(locdata) // (n_chunks - 1)

            if order == "successive":
                if (len(locdata) % chunk_size) == 0:
                    chunk_sizes = [chunk_size] * n_chunks
                else:
                    chunk_sizes = [chunk_size] * (n_chunks - 1) + [
                        (len(locdata) % chunk_size)
                    ]
                cum_chunk_sizes = list(accumulate(chunk_sizes))
                cum_chunk_sizes.insert(0, 0)
                index_lists = [
                    locdata.data.index[slice(lower, upper)]  # type: ignore
                    for lower, upper in zip(cum_chunk_sizes[:-1], cum_chunk_sizes[1:])
                ]

            elif order == "alternating":
                index_lists = [
                    locdata.data.index[slice(i_chunk, None, n_chunks)]  # type: ignore
                    for i_chunk in range(n_chunks)
                ]

            else:
                raise ValueError(f"The order {order} is not implemented.")

        if drop and len(index_lists) > 1 and len(index_lists[-1]) < len(index_lists[0]):
            index_lists = index_lists[:-1]

        references = [
            LocData.from_selection(locdata=locdata, indices=list(index_list))
            for index_list in index_lists
        ]
        dataframe = pd.DataFrame([ref.properties for ref in references])

        meta_ = metadata_pb2.Metadata()

        meta_.creation_time.GetCurrentTime()
        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.RAW
        meta_.ancestor_identifiers[:] = [ref.meta.identifier for ref in references]
        meta_.history.add(name="LocData.chunks")

        meta_ = merge_metadata(metadata=meta_, other_metadata=meta)

        return cls(references=references, dataframe=dataframe, meta=meta_)

    def reset(self, reset_index: bool = False) -> Self:
        """
        Reset hulls and properties. This is needed after the dataframe
        attribute has been modified in place.

        Note
        ----
        Should be used with care because metadata is not updated accordingly.
        The region property is not changed.
        Better to just re-instantiate with `LocData.from_dataframe()` or
        use `locdata.update()`.

        Parameters
        ----------
        reset_index
            Flag indicating if the index is reset to integer values.
            If True the previous index values are discarded.

        Returns
        -------
        Self
            The modified object
        """
        if reset_index is True:
            self.dataframe.reset_index(drop=True, inplace=True)

        self.properties = {}
        self._bounding_box = None
        self._oriented_bounding_box = None
        self._convex_hull = None
        self._alpha_shape = None
        self._inertia_moments = None

        self._update_properties()

        return self

    def update(
        self,
        dataframe: pd.DataFrame | None,
        reset_index: bool = False,
        meta: metadata_pb2.Metadata
        | dict[str, Any]
        | str
        | bytes
        | os.PathLike[Any]
        | BinaryIO
        | None = None,
    ) -> Self:
        """
        Update the dataframe attribute in place.

        Use this function rather than setting locdata.dataframe directly in
        order to automatically update
        the attributes for dimension, hulls, properties, and metadata.

        Parameters
        ----------
        dataframe
            Dataframe with localization data.
        reset_index
            Flag indicating if the index is reset to integer values.
            If True the previous index values are discarded.
        meta : locan.data.metadata_pb2.Metadata | dict | str | bytes | os.PathLike | BinaryIO | None
            Metadata about the current dataset and its history.

        Returns
        -------
        Self
            The modified object
        """
        if dataframe is None:
            return self

        local_parameter = locals()
        del local_parameter[
            "dataframe"
        ]  # dataframe is obvious and possibly large and should not be repeated in meta.

        if self.references is not None:
            self.reduce(reset_index=reset_index)
            logger.warning(
                "LocData.reduce() was applied since self.references was not None."
            )

        self.dataframe = dataframe
        self.dimension = len(self.coordinate_keys)
        self.reset(reset_index=reset_index)  # update hulls and properties

        # update meta
        self.meta.modification_time.GetCurrentTime()
        self.meta.state = metadata_pb2.MODIFIED
        self.meta.history.add(name="LocData.update", parameter=str(local_parameter))

        self.meta.element_count = len(self.data.index)
        if "frame" in self.data.columns:
            self.meta.frame_count = len(self.data["frame"].unique())

        self.meta = merge_metadata(metadata=self.meta, other_metadata=meta)

        return self

    def reduce(self, reset_index: bool = False) -> Self:
        """
        Clean up references.

        This includes to update `Locdata.dataframe` and set
        `LocData.references` and `LocData.indices` to None.

        Parameters
        ----------
        reset_index
            Flag indicating if the index is reset to integer values.
            If True the previous index values are discarded.

        Returns
        -------
        Self
            The modified object
        """
        if self.references is None:
            pass
        elif isinstance(self.references, (LocData, list)):
            self.dataframe = self.data
            self.indices = None
            self.references = None
        else:
            raise ValueError("references has undefined value.")

        if reset_index is True:
            self.dataframe.reset_index(drop=True, inplace=True)

        return self

    def update_convex_hulls_in_references(self) -> Self:
        """
        Compute the convex hull for each element in locdata.references and
        update locdata.dataframe.

        Returns
        -------
        Self
            The modified object
        """
        if isinstance(self.references, list):
            for reference in self.references:
                reference.convex_hull  # request property to update reference._convex_hull  # noqa B018

            new_df = pd.DataFrame(
                [reference.properties for reference in self.references]
            )
            new_df.index = self.data.index
            self.dataframe.update(new_df)
            new_columns = [
                column for column in new_df.columns if column in self.dataframe.columns
            ]
            new_df.drop(columns=new_columns, inplace=True, errors="ignore")
            self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
        return self

    def update_oriented_bounding_box_in_references(self) -> Self:
        """
        Compute the oriented bounding box for each element in
        locdata.references and update locdata.dataframe.

        Returns
        -------
        Self
            The modified object
        """
        if isinstance(self.references, list):
            for reference in self.references:
                reference.oriented_bounding_box  # request property to update reference._convex_hull  # noqa B018
            new_df = pd.DataFrame(
                [reference.properties for reference in self.references]
            )
            new_df.index = self.data.index
            self.dataframe.update(new_df)
            new_columns = [
                column for column in new_df.columns if column in self.dataframe.columns
            ]
            new_df.drop(columns=new_columns, inplace=True, errors="ignore")
            self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
        return self

    def projection(self, coordinate_labels: str | list[str]) -> LocData:
        """
        Reduce dimensions by projecting all localization coordinates onto
        selected coordinates.

        Parameters
        ----------
        coordinate_labels
            The coordinate labels to project onto.

        Returns
        -------
            LocData
        """
        local_parameter = locals()

        if isinstance(coordinate_labels, str):
            coordinate_labels = [coordinate_labels]

        new_locdata: LocData = copy.deepcopy(self)

        # reduce coordinate dimensions
        coordinate_labels_to_drop = [
            label for label in self.coordinate_keys if label not in coordinate_labels
        ]
        columns = self.data.columns
        new_columns = [
            column for column in columns if column not in coordinate_labels_to_drop
        ]
        dataframe = new_locdata.data[new_columns]

        # update
        _meta = metadata_pb2.Metadata()
        _meta.history.add(name="LocData.projection", parameter=str(local_parameter))
        # other updates are done in the coming update call.

        new_locdata = new_locdata.update(dataframe=dataframe, meta=_meta)

        return new_locdata

    def print_meta(self) -> None:
        """
        Print Locdata.metadata.

        See Also
        --------
        :func:`locan.data.metadata_utils.metadata_to_formatted_string`
        """
        print(metadata_to_formatted_string(self.meta))

    def print_summary(self) -> None:
        """
        Print a summary containing the most common metadata keys.
        """
        meta_ = metadata_pb2.Metadata()
        if self.meta.HasField("file"):
            meta_.file.CopyFrom(self.meta.file)
        meta_.identifier = self.meta.identifier
        meta_.comment = self.meta.comment
        meta_.creation_time.CopyFrom(self.meta.creation_time)
        if self.meta.HasField("modification_time"):
            meta_.modification_time.CopyFrom(self.meta.modification_time)
        meta_.source = self.meta.source
        meta_.state = self.meta.state
        meta_.element_count = self.meta.element_count
        meta_.frame_count = self.meta.frame_count

        print(metadata_to_formatted_string(meta_))

    def update_properties_in_references(
        self,
        properties: dict[str, Iterable[Any]]
        | pd.Series[Any]
        | pd.DataFrame
        | Callable[..., Any]
        | None = None,
    ) -> Self:
        """
        Add properties for each element in self.references
        and update self.dataframe.

        Parameters
        ----------
        properties
            new property values for each reference or
            function to compute property for LocData object.

        Returns
        -------
        Self
        """
        if not isinstance(self.references, list):
            raise TypeError("self.references must be a list of Locdata")

        if properties is None:
            pass
        elif isinstance(properties, dict):
            for key, values in properties.items():
                for reference, value_ in zip(self.references, values):
                    reference.properties.update({key: value_})
        elif isinstance(properties, pd.Series):
            if any(properties.index != range(len(self.references))):
                raise ValueError(
                    f"properties must have a range index of length {len(self.references)}"
                )
            for reference, value_ in zip(self.references, properties.to_numpy()):
                reference.properties.update({properties.name: value_})
        elif isinstance(properties, pd.DataFrame):
            if any(properties.index != range(len(self.references))):
                raise ValueError(
                    f"properties must have a range index of length {len(self.references)}"
                )
            for name in properties.columns:
                for reference, value_ in zip(
                    self.references, properties[name].to_numpy()
                ):
                    reference.properties.update({name: value_})
        elif callable(properties):
            for reference in self.references:
                reference.properties.update(properties(reference))

        new_df = pd.DataFrame([reference.properties for reference in self.references])
        new_df.index = self.data.index

        references_ = self.references
        self.update(dataframe=new_df)
        self.references = references_

        return self
