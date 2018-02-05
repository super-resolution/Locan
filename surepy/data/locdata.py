import time

from google.protobuf import text_format
import pandas as pd

import surepy.data.hulls
from surepy.data import metadata_pb2


class LocData():
    """
    This class carries localization data, aggregated properties and meta data.

    Data consist of individual elements being either localizations or other LocData objects. Both, localizations and
    Locdata objects have properties. Properties come from the original data or are added by analysis procedures.
    Analysis classes can take LocData objects as input on which to perform their action.

    Parameters
    ----------
    references : LocData or list(LocData) or None
        A locData reference or an array with references to locData objects referring to the selected localizations
        in dataset.
    dataframe : Pandas DataFrame or None
        Dataframe with localization data.
    indices : slice object or list(int) or None
        Indices for dataframe in references that makes up the data.
    meta : Metadata protobuf message
        Metadata about the current dataset and its history.

    Attributes
    ----------
    count : int (class attribute)
        A counter for counting locdata instantiations.

    references : LocData or list(LocData) or None
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

    bounding_box : Hull object
        Object representing the axis-aligned minimal bounding box.
    oriented_bounding_box : Hull object
        Object representing the oriented minimal bounding box.
    convex_hull : Hull object
        Object representing the convex hull of all localizations.
    alpha_shape : Hull object
        Object representing the alpha-shape of all localizations.

    """
    count=0

    # todo: delete kwargs
    # todo: change dataframe to _dataframe

    def __init__(self, references=None, dataframe=pd.DataFrame(), indices=None,
                 meta=None, **kwargs):
        self.__class__.count += 1

        self.references = references
        self.dataframe = dataframe
        self.indices = indices
        self.meta = metadata_pb2.Metadata()
        self.properties = {}

        # meta
        self.meta.identifier = str(self.__class__.count)
        self.meta.creation_date = int(time.time())
        self.meta.source = metadata_pb2.DESIGN
        self.meta.state = metadata_pb2.RAW
        self.meta.history.add(name = 'instantiate')

        self.meta.element_count = len(self.data.index)
        if 'Frame' in self.data.columns:
            self.meta.frame_count = len(self.data['Frame'].unique())

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(self.meta, key, value)
        else:
            self.meta.MergeFrom(meta)

        # coordinate labels
        self.coordinate_labels = sorted(list(set(self.data.columns).intersection({'Position_x', 'Position_y', 'Position_z'})))

        # property for mean spatial coordinates (centroids)
        self.properties.update(dict(self.data[self.coordinate_labels].mean()))

        # property for bounding box measures
        self.bounding_box = surepy.data.hulls.Bounding_box(self.coordinates)
        self.properties['Region_measure_bb'] = self.bounding_box.region_measure
        self.properties['Subregion_measure_bb'] = self.bounding_box.subregion_measure
        self.properties['Localization_density_bb'] = self.meta.element_count / self.bounding_box.region_measure


    @classmethod
    def from_dataframe(cls, dataframe=pd.DataFrame(), meta=None, **kwargs):

        dataframe = dataframe
        meta_ = metadata_pb2.Metadata()

        meta_.creation_date = int(time.time())
        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.RAW
        meta_.history.add(name = 'LocData.from_dataframe')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(dataframe=dataframe, meta=meta_, **kwargs)


    @classmethod
    def from_selection(cls, locdata, indices=slice(0, None), meta=None, **kwargs):

        references = locdata
        indices = indices
        meta_ = metadata_pb2.Metadata()
        meta_.CopyFrom(locdata.meta)
        meta_.ClearField("identifier")

        meta_.modification_date = int(time.time())
        meta_.state = metadata_pb2.MODIFIED
        meta_.ancestor_identifiers.append(locdata.meta.identifier)
        meta_.history.add(name = 'LocData.from_selection')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(references=references, indices=indices, meta=meta_, **kwargs)


    @classmethod
    def from_collection(cls, *args, meta=None, **kwargs):

        references = args
        dataframe = pd.DataFrame([ref.properties for ref in references])

        meta_ = metadata_pb2.Metadata()

        meta_.creation_date = int(time.time())
        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.RAW
        meta_.ancestor_identifiers[:] = [ref.meta.identifier for ref in references]
        meta_.history.add(name = 'LocData.from_collection')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(references=references, dataframe=dataframe, meta=meta_, **kwargs)


    @classmethod
    def concat(cls, *locdata, meta=None, **kwargs):
        """
        Concatenate LocData objects.

        Parameters
        ----------
        locdata : LocData object
            Locdata objects to concatenate.
        meta : Metadata protobuf message
            Metadata about the current dataset and its history.

        Returns
        -------
        locdata object
            A new locdata instance with dataframe representing the concatenated data.
        """

        dataframe = pd.concat([i.data for i in locdata], ignore_index=True)
        meta_ = metadata_pb2.Metadata()

        meta_.creation_date = int(time.time())
        meta_.source = metadata_pb2.DESIGN
        meta_.state = metadata_pb2.MODIFIED
        meta_.ancestor_identifiers[:] = [dat.meta.identifier for dat in locdata]
        meta_.history.add(name = 'concat')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(dataframe=dataframe, meta=meta_, **kwargs)


    @property
    def data(self):
       """ Return a pandas dataframe with all elements either copied from the reference or referencing the current
       dataframe. """
       if isinstance(self.references, LocData):
           df = self.references.data.iloc[self.indices]
           df = df.reset_index()
           df = pd.merge(df, self.dataframe, left_index=True, right_index=True, how='outer')
           # todo: alternative: df = df.assign(**self.dataframe.dict())
           return df
       else:
           return self.dataframe


    @property
    def coordinates(self):
       """ Return a numpy ndarray with all coordinate values. """
       return self.data[self.coordinate_labels].values


    def __del__(self):
        """ updating the counter upon deletion of class instance. """
        self.__class__.count -= 1


    def __len__(self):
        """ Return the length of data, i.e. the number of elements (localizations or collection elements)."""
        return len(self.data.index)


    def save(self, path):
        """
        Save LocData object in an appropriate way.

        Parameters
        ----------
        path : str
            Filepath for saving LocData.
        """
        raise NotImplementedError


    def reduce(self):
        """
        Update dataframe, reset dataframe.index, delete all references, set indices to None.

        Returns
        -------
        Int
            a flag set to 1 indicating if reference was changed, or set to 0 if no change was applied.
        """
        if self.references is None:
            return 0
        elif isinstance(self.references, LocData):
            self.dataframe = self.references.data.iloc[self.indices]
            self.dataframe = self.dataframe.reset_index()
            self.indices = None
            self.references = None
            return 1
        else:
            self.dataframe = self.dataframe.reset_index()
            self.indices = None
            self.references = None
            return 1


    def print_meta(self):
        print (text_format.MessageToString(self.meta))
