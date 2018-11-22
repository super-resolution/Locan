"""
Methods for building an analysis pipeline.

Pipeline refers to sequential analysis steps applied to a single LocData object. Pipeline thus include true piped
analysis, where a preliminary result serves as input to the next analysis step, but also workflows that provide
different results in parallel.
"""

from surepy import LocData
from surepy.io.io_locdata import load_locdata
from surepy.data.rois import Roi, load_from_roi_file
from surepy.data.hulls import Convex_hull_scipy
from surepy.data.filter import select_by_condition
from surepy.data.cluster.clustering import clustering_hdbscan
from surepy.analysis.analysis_tools import _init_meta, _update_meta


class Pipeline():
    """
    The base class for specialized analysis pipelines to be used on LocData objects.

    The custom analysis routine has to be added by implementing the compute(self) method. Results are provided as
    customized attributes. We suggest abbreviated standard names for the most common procedures:

    * rhf - Ripley H function
    * clust - locdata with clustered elements

    Parameters
    ----------
    locdata : LocData object, Roi object, or dict
        Localization data or a dict with keys `file_path` and `file_type` for a path pointing to a localization file and
        an integer indicating the file type. The integer should be according to surepy.data.metadata_pb2.file_type.
        If the `file_type` is "roi" a Roi object is loaded from the given `file_path`.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    """
    # todo check on input of meta

    count = 0

    def __init__(self, locdata, meta=None):
        self.__class__.count += 1

        self.parameter = None # is needed to init metadata_analysis_pb2.
        self.meta = _init_meta(self)
        self.meta = _update_meta(self, meta)

        # prepare locdata as general input
        if isinstance(locdata, LocData):
            self.locdata = locdata
        elif isinstance(locdata, Roi):
            self.locdata = locdata.locdata()
        elif isinstance(locdata, dict) and locdata['file_type']=='roi':
            self.locdata = load_from_roi_file(path=locdata['file_path'])
        elif isinstance(locdata, dict):
            self.locdata = load_locdata(path=locdata['file_path'], type=locdata['file_type'])


    def __del__(self):
        """ updating the counter upon deletion of class instance. """
        self.__class__.count -= 1


    def compute(self):
        """ The analysis routine to be applied on locdata."""
        raise NotImplementedError


    def save_protocol(self, path):
        '''
        Save the analysis routine (i.e. the compute() method) as human readable text.

        Parameters
        ----------
        path : str or Path object
            Path and file name for saving the text file.
        '''
        import inspect
        with open(path, 'w') as handle:
            handle.write('Analysis Pipeline: {}\n\n'.format(self.__class__.__name__))
            handle.write(inspect.getsource(self.compute))


def compute_test(self):
    ''' A pipeline definition for testing.'''
    print('compute_test() is running ok!')
    self.test = True

    return self


def compute_clust(self):
    '''A Pipeline definition for standard cluster analysis. '''

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

    return self
