"""
This module provides methods for building an analysis pipeline. Pipeline refers to sequential analysis steps applied to
a single localization data object.
"""

from surepy import LocData
from surepy.data.hulls import Convex_hull_scipy
from surepy.analysis import Localization_precision, Localizations_per_frame, Localization_property
from surepy.data.filter import select_by_condition, random_subset
from surepy.data.clustering import clustering_hdbscan, clustering_dbscan


# todo: define and use _Pipeline with metadata similar to locdata
class _Pipeline():
    """
    The base class for specialized analysis pipelines to be used on LocData objects.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    kwargs :
        Parameter that are passed to the algorithm.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    locdata : LocData object
        Localization data.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.
    """
    count = 0

    def __init__(self, locdata, meta, **kwargs):
        self.__class__.count += 1

        self.locdata = locdata
        self.parameter = kwargs
        self.meta = _init_meta(self)
        self.meta = _update_meta(self, meta)
        self.results = None


    def save_protocol(self, path):
        import inspect
        with open(path, 'w') as handle:
            handle.write('Analysis Pipeline: {}\n\n'.format(self.__class__.__name__))
            handle.write(inspect.getsource(self.compute))


    def compute(self):
        """ Define analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError


class Pipeline():
    """
    The base class for specialized analysis pipelines to be used on LocData objects.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.

    Attributes
    ----------
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.

    and others
    """

    def __init__(self, locdata):
        self.locdata = locdata
        self.identifier = None
        self.roi = None

        self.rhf = None
        self.noise = None
        self.clust = None
        self.clust_selection = None

    def compute(self):
        """ Define analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError

    def save_protocol(self, path):
        import inspect
        with open(path, 'w') as handle:
            handle.write('Analysis Pipeline: {}\n\n'.format(self.__class__.__name__))
            handle.write(inspect.getsource(self.compute))


class Pipeline_test(Pipeline):
    """
    A Pipeline definition for testing.
    """

    def __init__(self, locdata):
        super().__init__(locdata)
        self.test = None

    def compute(self):
        print('This is running ok!')
        self.test = True


class Pipeline_cluster():
    """
    A Pipeline definition for standard cluster analysis.
    """

    def __init__(self, locdata):
        super().__init__(locdata)
        self.rhf = None
        self.noise = None
        self.clust = None
        self.clust_selection = None


    def compute(self):

        # compute cluster
        self.noise, self.clust = clustering_hdbscan(self.locdata, min_cluster_size=5, allow_single_cluster=False, noise=True)

        # compute Localization_count
        localization_count = [len(self.clust.references[i]) for i in range(len(self.clust))]
        self.clust.dataframe = self.clust.dataframe.assign(Localization_count=localization_count)

        # compute convex hull
        Hs = [Convex_hull_scipy(self.clust.references[i].coordinates) for i in range(len(self.clust))]
        self.clust.dataframe = self.clust.dataframe.assign(Region_measure_ch=[H.region_measure for H in Hs])

        # select cluster
        self.clust_selection = select_by_condition(self.clust, condition='Region_measure_ch < 10_000')

        # free memory
        self.clust_selection.reduce()
