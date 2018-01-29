import pandas as pd
from datetime import datetime
import surepy.data.hulls


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
    dataframe : Pandas dataframe or None
        dataframe with localization data
    indices : slice object or list(int) or None
        Indices for dataframe in references that makes up the data.
    meta : dict
        Metadata about the current dataset and its history.

    Attributes
    ----------
    count : int (class attribute)
        A counter for counting locdata instantiations.

    references : LocData or list(LocData) or None
        A locData reference or an array with references to locData objects referring to the selected localizations
        in dataset.
    dataframe : Pandas dataframe or None
        dataframe with localization data
    indices : slice object or list(int) or None
        Indices for dataframe in references that makes up the data.
    meta : dict
        Metadata about the current dataset and its history.
    properties : pandas data frame
        list of properties generated from data.

    bounding_box : Hull object
        object representing the axis-aligned minimal bounding box
    oriented_bounding_box : Hull object
        object representing the oriented minimal bounding box
    convex_hull : Hull object
        object representing the convex hull of all localizations
    alpha_shape : Hull object
        object representing the alpha-shape of all localizations

    """
    count=0

    # todo: delete kwargs

    def __init__(self, references=None, dataframe=pd.DataFrame(), indices=None, meta=None, **kwargs):
        # self.__class__.count += 1
        self.__class__.count += 1

        self.references = references
        self.dataframe = dataframe
        self.meta = {}
        self.indices = indices
        self.properties = {}

        # meta
        self.meta['Identifier'] = str(self.__class__.count)
        self.meta['Production date'] = datetime.now().isoformat(' ')
        self.meta['Source'] = 'design'
        self.meta['State'] = 'raw'
        self.meta['History'] = ['instantiated']
        self.meta['Number of elements'] = len(self.data.index)
        self.meta['Number of frames'] = len(self.data['Frame'].unique()) if 'Frame' in self.data.columns else ''

        if meta is not None:
            self.meta.update(meta)

        # coordinate labels
        self.coordinate_labels = sorted(list(set(self.data.columns).intersection({'Position_x', 'Position_y', 'Position_z'})))

        # property for mean spatial coordinates (centroids)
        self.properties.update(dict(self.data[self.coordinate_labels].mean()))

        # property for bounding box measures
        self.bounding_box = surepy.data.hulls.Bounding_box(self.coordinates)
        self.properties['Region_measure_bb'] = self.bounding_box.region_measure
        self.properties['Subregion_measure_bb'] = self.bounding_box.subregion_measure
        self.properties['Localization_density_bb'] = self.meta['Number of elements'] / self.bounding_box.region_measure


    @classmethod
    def from_dataframe(cls, dataframe=pd.DataFrame(), meta=None, **kwargs):

        dataframe = dataframe
        meta_ = {}
        meta_['Production date'] = datetime.now().isoformat(' ')
        meta_['Source'] = 'design'
        meta_['State'] = 'raw'
        meta_['History'] = ['instantiated from dataframe']

        if meta is not None:
            meta_.update(meta)

        return cls(dataframe=dataframe, meta=meta, **kwargs)


    @classmethod
    def from_selection(cls, locdata, indices=slice(0, None), meta=None, **kwargs):

        references = locdata
        indices = indices
        meta_ = locdata.meta.copy()
        meta_.pop('Identifier', None)

        meta_['Modification date'] = datetime.now().isoformat(' ')
        meta_['State'] = 'modified'
        meta_['History'] = ['selection']
        meta_['Ancestor id'] = locdata.meta['Identifier']

        if meta is not None:
            meta_.update(meta)

        return cls(references=references, indices=indices, meta=meta_, **kwargs)


    @classmethod
    def from_collection(cls, *args, meta=None, **kwargs):

        references = args
        dataframe = pd.DataFrame([ref.properties for ref in references])

        meta_ = {}
        meta_['Modification date'] = datetime.now().isoformat(' ')
        meta_['State'] = 'modified'
        meta_['History'] = ['collection']
        meta_['Ancestor id'] = [ref.meta['Identifier'] for ref in references]

        if meta is not None:
            meta_.update(meta)

        return cls(references=references, dataframe=dataframe, meta=meta_, **kwargs)


    @classmethod
    def concat(cls, *locdata, meta=None, **kwargs):
        """
        Concatenate locdata objects.

        Parameters
        ----------
        locdata : LocData object
            Locdata objects to concatenate.
        meta : dict
            Metadata about the current dataset and its history.

        Returns
        -------
        locdata object
            a new locdata instance with dataframe representing the concatenated data.
        """


        dataframe = pd.concat([i.data for i in locdata], ignore_index=True)
        if meta is not None:
            meta_ = meta
        else:
            meta_ = {}

        meta_['Modification date'] = datetime.now().isoformat(' ')
        meta_['State'] = 'modified'
        meta_['History'] = ['collection']
        meta_['Ancestor id'] = [dat.meta['Identifier'] for dat in locdata]

        return cls(dataframe=dataframe, meta=meta_, **kwargs)


    @property
    def data(self):
       """ Return a pandas dataframe with all elements either copied from the reference or referencing the current
       dataframe. """
       if isinstance(self.references, LocData):
           df = self.references.data.iloc[self.indices]
           df = df.reset_index()
           df = pd.merge(df, self.dataframe, left_index=True, right_index=True, how='outer')
           # alternative: df = df.assign(**self.dataframe.dict())
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
        Update dataframe, reste dataframe.index, delete all references, set indices to None.

        Returns
        -------
        Int
            a flag set to 1 indicating if reference was changed, or set to 0 if no change was applied.
        """
        if isinstance(self.references, LocData):
            self.dataframe = self.references.data.iloc[self.indices]
            self.dataframe = self.dataframe.reset_index()
            self.indices = None
            self.references = None
            return 1
        else:
            self.dataframe = self.dataframe.reset_index()
            return 0


    def print_meta(self):
        for k, v in self.meta.items():
            print (k + ": " + str(v))


