"""

Check for the presence of clusters.

The existence of clusters is tested by analyzing variations in cluster area and localization density within clusters.

The analysis routine follows the ideas in [1]_ and [2]_.


References
----------
.. [1] First publication
.. [2] Second publication

"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from surepy import LocData
from surepy.analysis.analysis_base import _Analysis
from surepy.data.cluster.clustering import clustering_hdbscan, clustering_dbscan
from surepy.data.filter import random_subset
from surepy.data.hulls import Convex_hull_scipy


#### The algorithms

def _density_based_cluster_check_parameter_for_single_dataset(locdata, algorithm=clustering_hdbscan, algo_parameter={}, hull='bb'):
    """
    Compute localization density, relative area coverage by the clusters (eta), average density of localizations
    within apparent clusters (rho) for a single localization dataset.
    """
    # total region
    if 'Region_measure' in locdata.properties:
        region_measure = locdata.properties['Region_measure']
    else:
        warnings.warn('Since no region of interest is defined, Region_measure_bb is used.', UserWarning)
        region_measure = locdata.properties['Region_measure_bb']

    # localization density
    if 'Localization_density' in locdata.properties:
        localization_density = locdata.properties['Localization_density']
    else:
        warnings.warn('Since no region of interest is defined, Localization_density_bb is used.', UserWarning)
        localization_density = locdata.properties['Localization_density_bb']

    # compute clusters
    noise, clust = algorithm(locdata, noise=True, **algo_parameter)

    if len(clust)==0:
        raise ValueError('There were no clusters found. Use different clustering parameter.')

    # compute cluster regions and densities
    if hull == 'bb':
        # Region_measure_bb has been computed upon instantiation

        # relative area coverage by the clusters
        eta = clust.data['Region_measure_bb'].sum() / region_measure

        # average_localization_density_in_cluster
        rho = clust.data['Localization_density_bb'].mean()

    elif hull == 'ch':
        # compute hulls
        Hs = [Convex_hull_scipy(ref.coordinates) for ref in clust.references]
        clust.dataframe = clust.dataframe.assign(Region_measure_ch=[H.region_measure for H in Hs])

        localization_density_ch = clust.data['Localization_count'] / clust.data['Region_measure_ch']
        clust.dataframe = clust.dataframe.assign(Localization_density_ch=localization_density_ch)

        # relative area coverage by the clusters
        eta = clust.data['Region_measure_ch'].sum() / region_measure

        # average_localization_density_in_cluster
        rho = clust.data['Localization_density_ch'].mean()

    else:
        raise TypeError('Computation for the specified hull is not implemented.')

    return localization_density, eta, rho


def _density_based_cluster_check(locdata, meta=None, algorithm=clustering_hdbscan, algo_parameter={}, hull='bb',
                                 bins=10, divide='random'):
    """
    Compute localization density, relative area coverage by the clusters (eta), average density of localizations
    within apparent clusters (rho) for the sequence of divided localization datasets.
    """
    numbers_loc = np.round( len(locdata) / (np.arange(bins) + 1))
    numbers_loc = np.flip(numbers_loc).astype(int)

    # take random subsets of localizations
    if divide == 'random':
        locdatas = [random_subset(locdata, number_points=n_pts) for n_pts in numbers_loc]
    elif divide == 'sequential':
        locdatas = [LocData.from_selection(locdata, indices=range(n_pts)) for n_pts in numbers_loc]
    else:
        raise TypeError(f'String input for divide {divide} is not valid.')

    results_ = [_density_based_cluster_check_parameter_for_single_dataset(locd, algorithm=algorithm, algo_parameter=algo_parameter, hull=hull)
                for locd in locdatas]

    results = pd.DataFrame(data = results_,
                           columns=['localization_density', 'eta', 'rho'])
    return results


def _normalize_rho(results):
    pass
    # localization_density, eta, rho = 0, 0, 0
    #
    #
    # rho_zero = 0
    #
    # return rho_zero


##### The specific analysis classes

class DensityBasedClusterCheck(_Analysis):
    """
    Check for the presence of clusters in localization data by analyzing variations in cluster area and localization
    density within clusters.

    Parameters
    ----------
    locdata : LocData object
        Localization data that might be clustered.
    algorithm : callable
        Clustering algorithm.
    algo_parameter : dict
        Dictionary with kwargs for `algorithm`.
    hull : str
        Hull computation that is used to compute cluster region measures (area or volume). The identifier string can
        be one of the defined hulls.
    bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range (10, by default).
        If bins is a sequence, it defines the bin edges, including the rightmost edge, allowing for non-uniform bin
        widths.
    divide: str
        Identifier to choose how to partition the localization data. For `random` localizations are selected randomly.
        For `sequential` localizations are selected as chuncks of increasing size always starting from the first
        element.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : pandas data frame
        Data frame with  localization density, relative area coverage by the clusters (eta), average density of
        localizations within apparent clusters (rho) and rho normalized to the extrapolated value for eta=0 (rho_zero).
    """
    count = 0

    def __init__(self, locdata, meta=None, algorithm=clustering_hdbscan, algo_parameter={}, hull='bb', bins=10,
                 divide='random'):
        super().__init__(locdata=locdata, meta=meta,
                         algorithm=algorithm, algo_parameter=algo_parameter, hull=hull, bins=bins, divide=divide)

    def compute(self):
        #self.results = pd.DataFrame({'radius': self.parameter['radii'], 'Ripley_k_data': ripley})
        self.results = _density_based_cluster_check(self.locdata, **self.parameter)
        return self


    def plot(self, ax=None, show=True):
        '''
        Provide plot of results as matplotlib axes object.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image
        show : bool
            Flag indicating if plt.show() is active.
        kwargs : dict
            Other parameters passed to `matplotlib.pyplot.plot()`.
        '''
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        self.results.plot(x='eta', y='rho', ax=ax)

        ax.set(title = 'Label-density variation of ',
               xlabel = 'Relative clustered area',
               ylabel = 'Localization density within cluster'
               )

        # show figure
        if show:
            plt.show()

        return None
