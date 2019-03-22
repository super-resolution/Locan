"""

A class to carry localization data.

"""

import time
import warnings

from google.protobuf import text_format
import pandas as pd

from surepy.data.region import RoiRegion
import surepy.data.hulls
from surepy.data import metadata_pb2


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
        Indices for dataframe in references that makes up the data.
    meta : Metadata protobuf message or dictionary
        Metadata about the current dataset and its history.

    Attributes
    ----------
    count : int
        A counter for counting LocData instantiations (class attribute).

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

    region : RoiRegion object
        Object representing the region that supports all localizations.
    bounding_box : Hull object
        Object representing the axis-aligned minimal bounding box.
    oriented_bounding_box : Hull object
        Object representing the oriented minimal bounding box.
    convex_hull : Hull object
        Object representing the convex hull of all localizations.
    alpha_shape : Hull object
        Object representing the alpha-shape of all localizations.

    """
    count = 0
    """ A counter for counting LocData instantiations (class attribute). """

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

        # meta
        self.meta.identifier = str(self.__class__.count)
        self.meta.creation_date = int(time.time())
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

        # coordinate labels
        self.coordinate_labels = sorted(list(set(self.data.columns).intersection({'position_x',
                                                                                  'position_y',
                                                                                  'position_z'})))

        # properties
        self.properties['localization_count'] = len(self.data.index)

        # property for mean spatial coordinates (centroids)
        self.properties.update(dict(self.data[self.coordinate_labels].mean()))

        # property for bounding box measures
        if len(self.data.index) > 1:
            try:
                self.bounding_box = surepy.data.hulls.BoundingBox(self.coordinates)
                if self.bounding_box.region_measure:
                    self.properties['region_measure_bb'] = self.bounding_box.region_measure
                    self.properties['localization_density_bb'] = self.meta.element_count / self.bounding_box.region_measure
                if self.bounding_box.subregion_measure:
                    self.properties['subregion_measure_bb'] = self.bounding_box.subregion_measure
            except ValueError:
                warnings.warn('Properties related to bounding box could not be computed.', UserWarning)
                pass

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region):
        # todo add check if all localizations are within region. If not put out a warning.
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

        meta_.creation_date = int(time.time())
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
    def from_selection(cls, locdata, indices=slice(0, None), meta=None):
        """
        Create `LocData` from selected elements in another `LocData`.

        Parameters
        ----------
        locdata : LocData object
            Locdata object from which to select elements.
        indices : slice object or list(int) or None
            Indices for elements in locdata that make up the new data.
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

        meta_.modification_date = int(time.time())
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

        meta_.creation_date = int(time.time())
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

        # check if all eleements are None
        if not any(references):
            references = None

        meta_ = metadata_pb2.Metadata()

        meta_.creation_date = int(time.time())
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


    @property
    def data(self):
        """ Return a pandas dataframe with all elements either copied from the reference or referencing the current
        dataframe. """
        if isinstance(self.references, LocData):
            df = self.references.data.iloc[self.indices]
            df = pd.merge(df, self.dataframe, left_index=True, right_index=True, how='outer')
            return df
        else:
            return self.dataframe


    @property
    def coordinates(self):
        """ Return a numpy ndarray with all coordinate values. """
        return self.data[self.coordinate_labels].values


    def __del__(self):
        """ Updating the counter upon deletion of class instance. """
        self.__class__.count -= 1


    def __len__(self):
        """ Return the length of data, i.e. the number of elements (localizations or collection elements)."""
        return len(self.data.index)


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
        Int
            Flag set to 1 indicating if reference was changed, or set to 0 if no change was applied.
        """
        if self.references is None:
            return_value = 0
        elif isinstance(self.references, LocData):
            self.dataframe = self.data
            self.indices = None
            self.references = None
            return_value = 1
        else:
            raise ValueError('reference has undefined value.')

        if reset_index is True:
            self.dataframe.reset_index(drop=True, inplace=True)

        return return_value


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
        meta_.identifier = self.meta.identifier
        meta_.comment = self.meta.comment
        meta_.creation_date = self.meta.creation_date
        meta_.modification_date = self.meta.modification_date
        meta_.source = self.meta.source
        meta_.state = self.meta.state
        meta_.element_count = self.meta.element_count
        meta_.frame_count = self.meta.frame_count

        print(text_format.MessageToString(meta_))
