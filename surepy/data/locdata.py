"""

A class to carry localization data.

"""
import time
import warnings
from itertools import accumulate

from google.protobuf import text_format, json_format
import numpy as np
import pandas as pd
from scipy.spatial.qhull import QhullError

from surepy.constants import LOCDATA_ID  # is required to use LOCDATA_ID as global variable
from surepy import PROPERTY_KEYS
from surepy.data.region import RoiRegion
import surepy.data.hulls
from surepy.data import metadata_pb2
from surepy.data.metadata_utils import _modify_meta
from surepy.utils.format import _time_string


__all__ = ['LocData']


class LocData:
    """
    This class carries localization data, aggregated properties and meta data.

    Data consist of individual elements being either localizations or other LocData objects. Both, localizations and
    Locdata objects have properties. Properties come from the original data or are added by analysis procedures.

    Parameters
    ----------
    references : LocData, list(LocData), or None
        A locData reference or an array with references to locData objects referring to the selected localizations
        in dataset.
    dataframe : Pandas DataFrame or None
        Dataframe with localization data.
    indices : slice object or list(int) or None
        Indices for dataframe in references that makes up the data. `Indices` refers to index label, not position.
    meta : Metadata protobuf message or dictionary
        Metadata about the current dataset and its history.

    Attributes
    ----------
    references : LocData, list(LocData) or None
        A locData reference or an array with references to locData objects referring to the selected localizations
        in dataframe.
    dataframe : Pandas DataFrame or None
        Dataframe with localization data.
    indices : slice object or list(int) or None
        Indices for dataframe in references that makes up the data.
    meta : Metadata protobuf message
        Metadata about the current dataset and its history.
    properties : Pandas DataFrame
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
        self._time = time.time()

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

        self.coordinate_labels = sorted(list(set(self.data.columns).intersection({'position_x',
                                                                                  'position_y',
                                                                                  'position_z'})))

        self.dimension = len(self.coordinate_labels)

        self._update_properties()

        # meta
        global LOCDATA_ID
        LOCDATA_ID += 1
        self.meta.identifier = str(LOCDATA_ID)

        self.meta.creation_date = _time_string(self._time)
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

    @property
    def bounding_box(self):
        """Hull object: Return an object representing the axis-aligned minimal bounding box."""
        if self._bounding_box is None:
            try:
                self._bounding_box = surepy.data.hulls.BoundingBox(self.coordinates)
                if self._bounding_box.region_measure:
                    self.properties['region_measure_bb'] = self._bounding_box.region_measure
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
                self._convex_hull = surepy.data.hulls.ConvexHull(self.coordinates)
                self.properties['region_measure_ch'] = self._convex_hull.region_measure
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
                self._oriented_bounding_box = surepy.data.hulls.OrientedBoundingBox(self.coordinates)
                self.properties['region_measure_obb'] = self._oriented_bounding_box.region_measure
                self.properties['localization_density_obb'] = self.properties['localization_count'] \
                                                                  / self._oriented_bounding_box.region_measure
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
                self._alpha_shape = surepy.data.hulls.AlphaShape(points=self.coordinates, alpha=alpha)
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
                reference.update_alpha_shape(alpha=alpha)  # request property to update reference._convex_hull
            self.dataframe = pd.DataFrame([reference.properties for reference in self.references])
        return self

    @property
    def region(self):
        """RoiRegion object: Return the region that supports all localizations."""
        return self._region

    @region.setter
    def region(self, region):
        # todo add sc_check if all localizations are within region. If not put out a warning.
        if isinstance(region, RoiRegion) or region is None:
            self._region = region
        elif isinstance(region, dict):
            self._region = RoiRegion(**region)
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
        """pandas DataFrame: Return all elements either copied from the reference or referencing the current
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
        Create new LocData object from pandas DataFrame with localization data.

        Parameters
        ----------
        dataframe : pandas DataFrame
            Localization data.
        meta : Metadata protobuf message
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData object
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
        meta : Metadata protobuf message
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData object
            A new LocData instance with dataframe representing the concatenated data.
        """
        dimension = len(coordinates[0])

        if coordinate_labels is None:
            coordinate_labels = ['position_x', 'position_y', 'position_z'][0:dimension]
        else:
            if all(cl in PROPERTY_KEYS for cl in coordinate_labels):
                coordinate_labels = coordinate_labels
            else:
                raise ValueError('The given coordinate_labels are not standard property keys.')

        dataframe = pd.DataFrame.from_records(data=coordinates, columns=coordinate_labels)

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
        locdata : LocData object
            Locdata object from which to select elements.
        indices : slice object or list(int) or None
            Index labels for elements in locdata that make up the new data.
            Note that contrary to usual python slices, both the start and the stop are included
            (see pandas documentation). `Indices` refer to index value not position in list.
        meta : Metadata protobuf message
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData object
            A new LocData instance with dataframe representing the selected data.
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

        meta_.modification_date = _time_string(time.time())
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
        locdatas : list of LocData objects
            Locdata objects to collect.
        meta : Metadata protobuf message
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData object
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
        locdatas : list of LocData objects
            Locdata objects to concatenate.
        meta : Metadata protobuf message
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData object
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

        # sc_check if all eleements are None
        if not any(references):
            references = None

        meta_ = metadata_pb2.Metadata()

        meta_.creation_date = _time_string(time.time())
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
    def from_chunks(cls, locdata, chunk_size, meta=None):
        """
        Divide locdata in chunks of successive elements.

        Parameters
        ----------
        locdatas : list of LocData objects
            Locdata objects to concatenate.
        chunk_size : int
            Number of localizations per chunk
        meta : Metadata protobuf message
            Metadata about the current dataset and its history.

        Returns
        -------
        LocData object
            A new LocData instance with references and dataframe elements representing the individual chunks.
        """
        chunk_sizes = [chunk_size] * (len(locdata) // chunk_size) + [(len(locdata) % chunk_size)]
        cum_chunk_sizes = list(accumulate(chunk_sizes))
        cum_chunk_sizes.insert(0, 0)
        index_lists = [locdata.data.index[slice(lower, upper)]
                       for lower, upper in zip(cum_chunk_sizes[:-1], cum_chunk_sizes[1:])]
        references = [LocData.from_selection(locdata=locdata, indices=index_list) for index_list in index_lists]

        dataframe = pd.DataFrame([ref.properties for ref in references])

        meta_ = metadata_pb2.Metadata()

        meta_.creation_date = _time_string(time.time())
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
        hulls, properties, and metadata.

        Parameters
        ----------
        dataframe : Pandas DataFrame or None
            Dataframe with localization data.
        reset_index : Bool
            Flag indicating if the index is reset to integer values. If True the previous index values are discarded.
        meta : Metadata protobuf message
            Metadata about the current dataset and its history.

        Returns
        -------
        self (LocData)
            The modified object
        """
        local_parameter = locals()
        del local_parameter['dataframe']  # dataframe is obvious and possibly large and should not be repeated in meta.

        if self.references is not None:
            self.reduce(reset_index=reset_index)
            warnings.warn("LocData.reduce() was applied since self.references was not None.")

        self._time = time.time()
        self.dataframe = dataframe

        # update hulls and properties
        self.reset(reset_index=reset_index)

        # update meta
        self.meta.modification_date = _time_string(self._time)
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
        if isinstance(self.references, (LocData, list)):
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
            self.dataframe = pd.DataFrame([reference.properties for reference in self.references])
        return self

    def print_meta(self):
        """
        Print Locdata.metadata.
        """
        print(text_format.MessageToString(self.meta))

    def print_summary(self):
        """
        Print a summary containing the most common metadata keys.
        """
        meta_ = metadata_pb2.Metadata()
        meta_.file_path = self.meta.file_path
        meta_.file_type = self.meta.file_type
        meta_.identifier = self.meta.identifier
        meta_.comment = self.meta.comment
        meta_.creation_date = self.meta.creation_date
        meta_.modification_date = self.meta.modification_date
        meta_.source = self.meta.source
        meta_.state = self.meta.state
        meta_.element_count = self.meta.element_count
        meta_.frame_count = self.meta.frame_count

        print(text_format.MessageToString(meta_))
