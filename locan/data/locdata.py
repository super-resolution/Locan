"""

A class to carry localization data.

"""
import time
import warnings
from itertools import accumulate
import copy
import logging

from google.protobuf import text_format, json_format
import numpy as np
import pandas as pd
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError # needed for Python 3.7

from locan.constants import LOCDATA_ID  # is required to use LOCDATA_ID as global variable
from locan import PROPERTY_KEYS
from locan.data.region import Region, RoiRegion
import locan.data.hulls
from locan.data import metadata_pb2
from locan.data.metadata_utils import _modify_meta, metadata_to_formatted_string
from locan.utils.format import _time_string
from locan.data.properties import inertia_moments


__all__ = ['LocData']

logger = logging.getLogger(__name__)


class LocData:
    """
    This class carries localization data, aggregated properties and meta data.

    Data consist of individual elements being either localizations or other `LocData` objects. Both, localizations and
    `Locdata` objects have properties. Properties come from the original data or are added by analysis procedures.

    Parameters
    ----------
    references : LocData, list(LocData), None
        A `LocData` reference or an array with references to `LocData` objects referring to the selected localizations
        in dataset.
    dataframe : pandas.DataFrame, None
        Dataframe with localization data.
    indices : slice object, list(int), None
        Indices for dataframe in references that makes up the data. `indices` refers to index label, not position.
    meta : locan.data.metadata_pb2.Metadata, dictionary
        Metadata about the current dataset and its history.

    Attributes
    ----------
    references : LocData, list(LocData), None
        A LocData reference or an array with references to LocData objects referring to the selected localizations
        in dataframe.
    dataframe : pandas.DataFrame, None
        Dataframe with localization data.
    indices : slice object, list(int), None
        Indices for dataframe in references that makes up the data.
    meta : locan.data.metadata_pb2.Metadata
        Metadata about the current dataset and its history.
    properties : pandas.DataFrame
        List of properties generated from data.
    coordinate_labels : list of str
        The available coordinate properties.
    dimension : int
        Number of coordinates available for each localization (i.e. size of `coordinate_labels`).
    """
    count = 0
    """int: A counter for counting LocData instantiations (class attribute)."""

    def __init__(self, references=None, dataframe=pd.DataFrame(), indices=None,
                 meta=None):
        self.__class__.count += 1

        self.references = references
        self.dataframe = dataframe
        self.indices = indices
        self.meta = metadata_pb2.Metadata()
        self.properties = {}

        # regions and hulls
        self._region = None
        self._bounding_box = None
        self._oriented_bounding_box = None
        self._convex_hull = None
        self._alpha_shape = None
        self._inertia_moments = None

        self.coordinate_labels = sorted(list(set(self.data.columns).intersection({'position_x',
                                                                                  'position_y',
                                                                                  'position_z'})))

        self.dimension = len(self.coordinate_labels)

        self._update_properties()

        # meta
        global LOCDATA_ID
        LOCDATA_ID += 1
        self.meta.identifier = str(LOCDATA_ID)

        self.meta.creation_time.GetCurrentTime()
        self.meta.source = metadata_pb2.DESIGN
        self.meta.state = metadata_pb2.RAW

        self.meta.element_count = len(self.data.index)
        if 'frame' in self.data.columns:
            self.meta.frame_count = len(self.data['frame'].unique())

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(self.meta, key, value)
        else:
            self.meta.MergeFrom(meta)

    def _update_properties(self):
        self.properties['localization_count'] = len(self.data.index)

        # property for mean spatial coordinates (centroids)
        self.properties.update(dict(self.data[self.coordinate_labels].mean()))

        self.bounding_box  # update self._bounding_box

    def __del__(self):
        """Updating the counter upon deletion of class instance."""
        self.__class__.count -= 1

    def __len__(self):
        """Return the length of data, i.e. the number of elements (localizations or collection elements)."""
        return len(self.data.index)

    def __getstate__(self):
        """Modify pickling behavior."""
        # Copy the object's state from self.__dict__ to avoid modifying the original state.
        state = self.__dict__.copy()
        # Serialize the unpicklable protobuf entries.
        json_string = json_format.MessageToJson(self.meta, including_default_value_fields=False)
        state['meta'] = json_string
        return state

    def __setstate__(self, state):
        """Modify pickling behavior."""
        # Restore instance attributes.
        self.__dict__.update(state)
        # Restore protobuf class for meta attribute
        self.meta = metadata_pb2.Metadata()
        self.meta = json_format.Parse(state['meta'], self.meta)

    def __copy__(self):
        """
        Create a shallow copy of locdata (keeping all references) with the following exceptions:
        (i) The class variable `count` is increased for the copied LocData object.
        (ii) Metadata keeps the original meta.creation_time while meta.modification_time and meta.history is updated.
        """
        new_locdata = LocData(self.references,
                              self.dataframe,
                              self.indices,
                              meta=None)
        new_locdata._region = self._region
        # meta
        meta_ = _modify_meta(self, new_locdata, function_name='LocData.copy',
                             parameter=None, meta=None)
        new_locdata.meta = meta_
        return new_locdata

    def __deepcopy__(self, memodict=None):
        """
        Create a deep copy of locdata (including all references) with the following exceptions:
        (i) The class variable `count` is increased for all deepcopied LocData objects.
        (ii) Metadata keeps the original meta.creation_time while meta.modification_time and meta.history is updated.
        """
        if memodict is None:
            memodict = {}
        new_locdata = LocData(copy.deepcopy(self.references, memodict),
                              copy.deepcopy(self.dataframe, memodict),
                              copy.deepcopy(self.indices, memodict),
                              meta=None)
        new_locdata._region = self._region
        # meta
        meta_ = _modify_meta(self, new_locdata, function_name='LocData.deepcopy',
                             parameter=None, meta=None)
        new_locdata.meta = meta_
        return new_locdata

    @property
    def bounding_box(self):
        """Hull object: Return an object representing the axis-aligned minimal bounding box."""
        if self._bounding_box is None:
            try:
                self._bounding_box = locan.data.hulls.BoundingBox(self.coordinates)
                self.properties['region_measure_bb'] = self._bounding_box.region_measure
                if self._bounding_box.region_measure:
                    self.properties['localization_density_bb'] = \
                        self.properties['localization_count'] / self._bounding_box.region_measure
                if self._bounding_box.subregion_measure:
                    self.properties['subregion_measure_bb'] = self._bounding_box.subregion_measure
            except ValueError:
                warnings.warn('Properties related to bounding box could not be computed.', UserWarning)
        return self._bounding_box

    @property
    def convex_hull(self):
        """Hull object: Return an object representing the convex hull of all localizations."""
        if self._convex_hull is None:
            try:
                self._convex_hull = locan.data.hulls.ConvexHull(self.coordinates)
                self.properties['region_measure_ch'] = self._convex_hull.region_measure
                if self._convex_hull.region_measure:
                    self.properties['localization_density_ch'] = self.properties['localization_count'] \
                                                                      / self._convex_hull.region_measure
            except (TypeError, QhullError):
                warnings.warn('Properties related to convex hull could not be computed.', UserWarning)
        return self._convex_hull

    @property
    def oriented_bounding_box(self):
        """Hull object: Return an object representing the oriented minimal bounding box."""
        if self._oriented_bounding_box is None:
            try:
                self._oriented_bounding_box = locan.data.hulls.OrientedBoundingBox(self.coordinates)
                self.properties['region_measure_obb'] = self._oriented_bounding_box.region_measure
                if self._oriented_bounding_box.region_measure:
                    self.properties['localization_density_obb'] = self.properties['localization_count'] \
                                                                      / self._oriented_bounding_box.region_measure
                self.properties['orientation_obb'] = self._oriented_bounding_box.angle
                self.properties['circularity_obb'] = self._oriented_bounding_box.elongation
            except TypeError:
                warnings.warn('Properties related to oriented bounding box could not be computed.', UserWarning)
        return self._oriented_bounding_box

    @property
    def alpha_shape(self):
        """Hull object: Return an object representing the alpha-shape of all localizations."""
        return self._alpha_shape

    def update_alpha_shape(self, alpha):
        """Compute the alpha shape for specific `alpha` and update `self.alpha_shape`.

        Parameters
        ----------
        alpha : float
            Alpha parameter specifying a unique alpha complex.

        Returns
        -------
        LocData
            The modified object
        """
        try:
            if self._alpha_shape is None:
                self._alpha_shape = locan.data.hulls.AlphaShape(points=self.coordinates, alpha=alpha)
            else:
                self._alpha_shape.alpha = alpha

            self.properties['region_measure_as'] = self._alpha_shape.region_measure
            try:
                self.properties['localization_density_as'] = self._alpha_shape.n_points_alpha_shape \
                                                             / self._alpha_shape.region_measure
            except ZeroDivisionError:
                self.properties['localization_density_as'] = float('nan')

        except TypeError:
            warnings.warn('Properties related to alpha shape could not be computed.', UserWarning)
        return self

    def update_alpha_shape_in_references(self, alpha):
        """
        Compute the alpha shape for each element in `locdata.references` and update `locdata.dataframe`.

        Returns
        -------
        LocData
            The modified object
        """
        if isinstance(self.references, list):
            for reference in self.references:
                reference.update_alpha_shape(alpha=alpha)
            new_df = pd.DataFrame([reference.properties for reference in self.references])
            new_df.index = self.data.index
            self.dataframe.update(new_df)
            new_columns = [column for column in new_df.columns if column in self.dataframe.columns]
            new_df.drop(columns=new_columns, inplace=True, errors='ignore')
            self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
        return self

    @property
    def inertia_moments(self):
        """Inertia moments are returned as computed by :func:`locan.data.properties.inertia_moments`."""
        if self._inertia_moments is None:
            try:
                self._inertia_moments = locan.data.properties.inertia_moments(self.coordinates)
                self.properties['orientation_im'] = self._inertia_moments.orientation
                self.properties['circularity_im'] = self._inertia_moments.eccentricity
            except TypeError:
                warnings.warn('Properties related to inertia_moments could not be computed.', UserWarning)
        return self._inertia_moments

    def update_inertia_moments_in_references(self):
        """
        Compute inertia_moments for each element in locdata.references and update locdata.dataframe.

        Returns
        -------
        LocData
            The modified object
        """
        if isinstance(self.references, list):
            for reference in self.references:
                reference.inertia_moments  # request property to update
            new_df = pd.DataFrame([reference.properties for reference in self.references])
            new_df.index = self.data.index
            self.dataframe.update(new_df)
            new_columns = [column for column in new_df.columns if column in self.dataframe.columns]
            new_df.drop(columns=new_columns, inplace=True, errors='ignore')
            self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
        return self

    @property
    def region(self):
        """RoiRegion object: Return the region that supports all localizations."""
        return self._region

    @region.setter
    def region(self, region):
        if region is not None:
            if region.dimension != self.dimension:
                raise TypeError("Region dimension and coordinates dimension must be identical.")
            elif len(self) != len(region.contains(self.coordinates)):
                logger.warning("Not all coordinates are within region.")

        if isinstance(region, (Region, RoiRegion)) or region is None:
            self._region = region

        elif isinstance(region, dict):  # legacy code to deal with deprecated RoiLegacy_0
            region_ = RoiRegion(**region)
            if region_ is not None:
                if region_.dimension != self.dimension:
                    raise TypeError("Region dimension and coordinates dimension must be identical.")
                elif len(self) != len(region_.contains(self.coordinates)):
                    logger.warning("Not all coordinates are within region.")
            self._region = region_

        else:
            raise TypeError

        # property for region measures
        if self._region is not None:
            if self._region.region_measure:
                self.properties['region_measure'] = self._region.region_measure
                self.properties['localization_density'] = self.meta.element_count / self._region.region_measure
            if self._region.subregion_measure:
                self.properties['subregion_measure'] = self._region.subregion_measure

    @property
    def data(self):
        """pandas.DataFrame: Return all elements either copied from the reference or referencing the current
        dataframe. """
        if isinstance(self.references, LocData):
            # we refer to the localization data by its index label, not position
            # in other words we decided not to use iloc but loc
            # df = self.references.data.loc[self.indices]  ... but this does not work in pandas.
            # also see:
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
            try:
                df = self.references.data.loc[self.indices]
            except KeyError:
                df = self.references.data.loc[self.references.data.index.intersection(self.indices)]
            df = pd.merge(df, self.dataframe, left_index=True, right_index=True, how='outer')
            return df
        else:
            return self.dataframe

    @property
    def coordinates(self):
        """ndarray: Return all coordinate values. """
        return self.data[self.coordinate_labels].values

    @property
    def centroid(self):
        """ndarray: Return coordinate values of the centroid
        (being the property values for all coordinate labels)."""
        return np.array([self.properties[coordinate_label] for coordinate_label in self.coordinate_labels])

    @classmethod
    def from_dataframe(cls, dataframe=pd.DataFrame(), meta=None):
        """
        Create new LocData object from pandas.DataFrame with localization data.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Localization data.
        meta : locan.data.metadata_pb2.Metadata
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with dataframe representing the concatenated data.
        """
        dataframe = dataframe
        meta_ = metadata_pb2.Metadata()

        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.RAW
        meta_.history.add(name='LocData.from_dataframe')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(dataframe=dataframe, meta=meta_)

    @classmethod
    def from_coordinates(cls, coordinates=(), coordinate_labels=None, meta=None):
        """
        Create new LocData object from a sequence of localization coordinates.

        Parameters
        ----------
        coordinates : sequence of tuples with shape (n_loclizations, dimension)
            Sequence of tuples with localization coordinates
        coordinate_labels : sequence of str
            The available coordinate properties.
        meta : locan.data.metadata_pb2.Metadata
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with dataframe representing the concatenated data.
        """
        if np.size(coordinates):
            dimension = len(coordinates[0])

            if coordinate_labels is None:
                coordinate_labels = ['position_x', 'position_y', 'position_z'][0:dimension]
            else:
                if all(cl in PROPERTY_KEYS for cl in coordinate_labels):
                    coordinate_labels = coordinate_labels
                else:
                    raise ValueError('The given coordinate_labels are not standard property keys.')

            dataframe = pd.DataFrame.from_records(data=coordinates, columns=coordinate_labels)

        else:
            dataframe = pd.DataFrame()

        meta_ = metadata_pb2.Metadata()
        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.RAW
        meta_.history.add(name='LocData.from_coordinates')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(dataframe=dataframe, meta=meta_)

    @classmethod
    def from_selection(cls, locdata, indices=slice(0, None), meta=None):
        """
        Create new LocData object from selected elements in another `LocData`.

        Parameters
        ----------
        locdata : LocData
            Locdata object from which to select elements.
        indices : slice object, list(int), None
            Index labels for elements in locdata that make up the new data.
            Note that contrary to usual python slices, both the start and the stop are included
            (see pandas documentation). `Indices` refer to index value not position in list.
        meta : locan.data.metadata_pb2.Metadata
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with dataframe representing the selected data.

        Note
        ----
        No error is raised if indices do not exist in locdata.
        """
        references = locdata
        indices = indices
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
        meta_.history.add(name='LocData.from_selection')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        new_locdata = cls(references=references, indices=indices, meta=meta_)
        new_locdata.region = references.region
        return new_locdata

    @classmethod
    def from_collection(cls, locdatas, meta=None):
        """
        Create new LocData object by collecting LocData objects.

        Parameters
        ----------
        locdatas : list of LocData
            Locdata objects to collect.
        meta : locan.data.metadata_pb2.Metadata
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
        meta_.history.add(name='LocData.from_collection')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(references=references, dataframe=dataframe, meta=meta_)

    @classmethod
    def concat(cls, locdatas, meta=None):
        """
        Concatenate LocData objects.

        Parameters
        ----------
        locdatas : list of LocData
            Locdata objects to concatenate.
        meta : locan.data.metadata_pb2.Metadata
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with dataframe representing the concatenated data.
        """

        dataframe = pd.concat([i.data for i in locdatas], ignore_index=True, sort=False)

        # concatenate references also if None
        references = []
        for locdata in locdatas:
            try:
                references.extend(locdata.references)
            except TypeError:
                references.append(locdata.references)

        # check if all elements are None
        if not any(references):
            references = None

        meta_ = metadata_pb2.Metadata()

        meta_.creation_time.GetCurrentTime()
        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.MODIFIED
        meta_.ancestor_identifiers[:] = [dat.meta.identifier for dat in locdatas]
        meta_.history.add(name='concat')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(references=references, dataframe=dataframe, meta=meta_)

    @classmethod
    def from_chunks(cls, locdata, chunks=None, chunk_size=None, n_chunks=None, order='successive', drop=False, meta=None):
        """
        Divide locdata in chunks of localization elements.

        Parameters
        ----------
        locdatas : list of LocData
            Locdata objects to concatenate.
        chunks : list[tuples]
            Localization chunks as defined by a list of index-tuples
        chunk_size : int, None
            Number of localizations per chunk. One of `chunk_size` or `n_chunks` must be different from None.
        n_chunks : int, None
            Number of chunks. One of `chunk_size` or `n_chunks` must be different from None.
        order : str
            The order in which to select localizations. One of 'successive' or 'alternating'.
        drop : bool
            If True the last chunk will be eliminated if it has fewer localizations than the other chunks.
        meta : locan.data.metadata_pb2.Metadata
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            A new LocData instance with references and dataframe elements representing the individual chunks.
        """
        n_nones = sum(element is None for element in [chunks, chunk_size, n_chunks])

        if n_nones != 2:
            raise ValueError("One and only one of `chunks`, `chunk_size` or `n_chunks` must be different from None.")
        elif chunks is not None:
            index_lists = list(chunks)
        else:
            if chunk_size is not None:
                if (len(locdata) % chunk_size) == 0:
                    n_chunks = len(locdata) // chunk_size
                else:
                    n_chunks = len(locdata) // chunk_size + 1
            else:  # if n_chunks is not None
                if (len(locdata) % n_chunks) == 0:
                    chunk_size = len(locdata) // n_chunks
                else:
                    chunk_size = len(locdata) // (n_chunks-1)

            if order == 'successive':
                if (len(locdata) % chunk_size) == 0:
                    chunk_sizes = [chunk_size] * n_chunks
                else:
                    chunk_sizes = [chunk_size] * (n_chunks-1) + [(len(locdata) % chunk_size)]
                cum_chunk_sizes = list(accumulate(chunk_sizes))
                cum_chunk_sizes.insert(0, 0)
                index_lists = [locdata.data.index[slice(lower, upper)]
                               for lower, upper in zip(cum_chunk_sizes[:-1], cum_chunk_sizes[1:])]

            elif order == 'alternating':
                index_lists = [locdata.data.index[slice(i_chunk, None, n_chunks)] for i_chunk in range(n_chunks)]

            else:
                raise ValueError(f"The order {order} is not implemented.")

        if drop and len(index_lists) > 1 and len(index_lists[-1]) < len(index_lists[0]):
            index_lists = index_lists[:-1]

        references = [LocData.from_selection(locdata=locdata, indices=index_list) for index_list in index_lists]
        dataframe = pd.DataFrame([ref.properties for ref in references])

        meta_ = metadata_pb2.Metadata()

        meta_.creation_time.GetCurrentTime()
        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.RAW
        meta_.ancestor_identifiers[:] = [ref.meta.identifier for ref in references]
        meta_.history.add(name='LocData.chunks')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(references=references, dataframe=dataframe, meta=meta_)

    def reset(self, reset_index=False):
        """
        Reset hulls and properties. This is needed after the dataframe attribute has been modified in place.

        Note
        ----
        Should be used with care because metadata is not updated accordingly.
        The region property is not changed.
        Better to just re-instantiate with `LocData.from_dataframe()` or use `locdata.update()`.

        Parameters
        ----------
        reset_index : Bool
            Flag indicating if the index is reset to integer values. If True the previous index values are discarded.

        Returns
        -------
        LocData
            The modified object
        """
        if reset_index is True:
            self.dataframe.reset_index(drop=True, inplace=True)

        self.properties = {}
        self._bounding_box = None
        self._oriented_bounding_box = None
        self._convex_hull = None
        self._alpha_shape = None

        self._update_properties()

        return self

    def update(self, dataframe, reset_index=False, meta=None):
        """
        Update the dataframe attribute in place.

        Use this function rather than setting locdata.dataframe directly in order to automatically update
        the attributes for dimension, coordinate_labels, hulls, properties, and metadata.

        Parameters
        ----------
        dataframe : pandas.DataFrame, None
            Dataframe with localization data.
        reset_index : Bool
            Flag indicating if the index is reset to integer values. If True the previous index values are discarded.
        meta : locan.data.metadata_pb2.Metadata
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData
            The modified object
        """
        local_parameter = locals()
        del local_parameter['dataframe']  # dataframe is obvious and possibly large and should not be repeated in meta.

        if self.references is not None:
            self.reduce(reset_index=reset_index)
            logger.warning("LocData.reduce() was applied since self.references was not None.")

        self.dataframe = dataframe
        self.coordinate_labels = sorted(list(set(self.data.columns).intersection({'position_x',
                                                                                  'position_y',
                                                                                  'position_z'})))
        self.dimension = len(self.coordinate_labels)
        self.reset(reset_index=reset_index)  # update hulls and properties

        # update meta
        self.meta.modification_time.GetCurrentTime()
        self.meta.state = metadata_pb2.MODIFIED
        self.meta.history.add(name='LocData.update', parameter=str(local_parameter))

        self.meta.element_count = len(self.data.index)
        if 'frame' in self.data.columns:
            self.meta.frame_count = len(self.data['frame'].unique())

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(self.meta, key, value)
        else:
            self.meta.MergeFrom(meta)

        return self

    def reduce(self, reset_index=False):
        """
        Clean up references.

        This includes to update `Locdata.dataframe` and set `LocData.references` and `LocData.indices` to None.

        Parameters
        ----------
        reset_index : Bool
            Flag indicating if the index is reset to integer values. If True the previous index values are discarded.

        Returns
        -------
        LocData
            The modified object
        """
        if self.references is None:
            pass
        elif isinstance(self.references, (LocData, list)):
            self.dataframe = self.data
            self.indices = None
            self.references = None
        else:
            raise ValueError('references has undefined value.')

        if reset_index is True:
            self.dataframe.reset_index(drop=True, inplace=True)

        return self

    def update_convex_hulls_in_references(self):
        """
        Compute the convex hull for each element in locdata.references and update locdata.dataframe.

        Returns
        -------
        LocData
            The modified object
        """
        if isinstance(self.references, list):
            for reference in self.references:
                reference.convex_hull  # request property to update reference._convex_hull

            new_df = pd.DataFrame([reference.properties for reference in self.references])
            new_df.index = self.data.index
            self.dataframe.update(new_df)
            new_columns = [column for column in new_df.columns if column in self.dataframe.columns]
            new_df.drop(columns=new_columns, inplace=True, errors='ignore')
            self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
        return self

    def update_oriented_bounding_box_in_references(self):
        """
        Compute the oriented bounding box for each element in locdata.references and update locdata.dataframe.

        Returns
        -------
        LocData
            The modified object
        """
        if isinstance(self.references, list):
            for reference in self.references:
                reference.oriented_bounding_box  # request property to update reference._convex_hull
            new_df = pd.DataFrame([reference.properties for reference in self.references])
            new_df.index = self.data.index
            self.dataframe.update(new_df)
            new_columns = [column for column in new_df.columns if column in self.dataframe.columns]
            new_df.drop(columns=new_columns, inplace=True, errors='ignore')
            self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
        return self

    def projection(self, coordinate_labels):
        """
        Reduce dimensions by projecting all localization coordinates onto selected coordinates.

        Parameters
        ----------
        coordinate_labels : str, list
            The coordinate labels to project onto.

        Returns
        -------
            LocData
        """
        local_parameter = locals()

        if isinstance(coordinate_labels, str):
            coordinate_labels = [coordinate_labels]

        new_locdata = copy.deepcopy(self)

        # reduce coordinate dimensions
        coordinate_labels_to_drop = [label for label in self.coordinate_labels if label not in coordinate_labels]
        columns = self.data.columns
        new_columns = [column for column in columns if column not in coordinate_labels_to_drop]
        dataframe = new_locdata.data[new_columns]

        # update
        _meta = metadata_pb2.Metadata()
        _meta.history.add(name='LocData.projection', parameter=str(local_parameter))
        # other updates are done in the coming update call.

        new_locdata = new_locdata.update(dataframe=dataframe, meta=_meta)

        return new_locdata

    def print_meta(self):
        """
        Print Locdata.metadata.
        """
        print(metadata_to_formatted_string(self.meta))

    def print_summary(self):
        """
        Print a summary containing the most common metadata keys.
        """
        meta_ = metadata_pb2.Metadata()
        meta_.file.CopyFrom(self.meta.file)
        meta_.identifier = self.meta.identifier
        meta_.comment = self.meta.comment
        meta_.creation_time.CopyFrom(self.meta.creation_time)
        meta_.modification_time.CopyFrom(self.meta.modification_time)
        meta_.source = self.meta.source
        meta_.state = self.meta.state
        meta_.element_count = self.meta.element_count
        meta_.frame_count = self.meta.frame_count

        print(metadata_to_formatted_string(meta_))
