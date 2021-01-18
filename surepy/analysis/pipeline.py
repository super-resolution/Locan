"""
Building an analysis pipeline.

Pipeline refers to sequential analysis steps applied to a single LocData object. Pipeline thus include true piped
analysis, where a preliminary result serves as input to the next analysis step, but also workflows that provide
different results in parallel.

Note
----
The implementation of an analysis pipeline is in an exploratory state.
"""

from surepy.data.locdata import LocData
from surepy.io.io_locdata import load_locdata
from surepy.data.rois import Roi
from surepy.data.hulls import ConvexHull
from surepy.data.filter import select_by_condition
from surepy.data.cluster.clustering import cluster_hdbscan
from surepy.analysis.analysis_base import _init_meta, _update_meta


__all__ = ['Pipeline']


class Pipeline():
    """
    The base class for specialized analysis pipelines to be used on LocData objects.

    The custom analysis routine has to be added by implementing the compute(self) method. Results are provided as
    customized attributes. We suggest abbreviated standard names for the most common procedures:

    * lp - Localization Precision
    * lprop - Localization Property
    * lpf - Localizations per Frame
    * rhf - Ripley H function
    * clust - locdata with clustered elements

    Parameters
    ----------
    locdata : LocData, Roi object, dict
        Localization data or a dict with keys `file_path` and `file_type` for a path pointing to a localization file and
        an integer indicating the file type. The integer should be according to surepy.data.metadata_pb2.Metadata.file_type.
        If the `file_type` is "roi" a Roi object is loaded from the given `file_path`.
    meta : surepy.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    locdata : LocData
        Localization data.
    meta : surepy.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    """
    # todo sc_check on input of meta

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
            self.locdata = Roi.from_yaml(path=locdata['file_path']).locdata()
        elif isinstance(locdata, dict):
            self.locdata = load_locdata(path=locdata['file_path'], file_type=locdata['file_type'])


    def __del__(self):
        """ updating the counter upon deletion of class instance. """
        self.__class__.count -= 1


    def compute(self):
        """ The analysis routine to be applied on locdata."""
        raise NotImplementedError


    def save_protocol(self, path):
        """
        Save the analysis routine (i.e. the compute() method) as human readable text.

        Parameters
        ----------
        path : str, os.PathLike
            Path and file name for saving the text file.
        """
        import inspect
        with open(path, 'w') as handle:
            handle.write('Analysis Pipeline: {}\n\n'.format(self.__class__.__name__))
            handle.write(inspect.getsource(self.compute))


def compute_test(self):
    """ A pipeline definition for testing."""
    self.test = True

    return self


def compute_clust(self):
    """ A Pipeline definition for standard cluster analysis. """

    # import required modules
    from pathlib import Path
    from surepy.data.cluster import cluster_dbscan
    from surepy.data.hulls import ConvexHull
    from surepy.data.filter import select_by_condition

    # compute cluster
    self.noise, self.clust = cluster_hdbscan(self.locdata, min_cluster_size=5, allow_single_cluster=False)

    # compute convex hull
    Hs = [ConvexHull(self.clust.references[i].coordinates) for i in range(len(self.clust))]
    self.clust.dataframe = self.clust.dataframe.assign(region_measure_ch=[H.region_measure for H in Hs])

    # select cluster
    self.clust_selection = select_by_condition(self.clust, condition='region_measure_ch < 10_000')

    # free memory
    self.clust_selection.reduce()

    # epilogue
    self.indicator = self.count - 1  # indices should start with 0
    self.file_indicator = Path(self.locdata.meta.file_path).stem
    print(f'Finished: {self.indicator}')

    return self
