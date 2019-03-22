"""
Localization-density variation analysis to characterize localization cluster.

The existence of clusters is tested by analyzing variations in cluster area and localization density within clusters.
The analysis routine follows the ideas in [1]_ and [2]_.

References
----------
.. [1] Baumgart F., Arnold AM., Leskovar K., Staszek K., Fölser M., Weghuber J., Stockinger H., Schütz GJ.,
   Varying label density allows artifact-free analysis of membrane-protein nanoclusters.
   Nat Methods. 2016 Aug;13(8):661-4. doi: 10.1038/nmeth.3897
.. [2] Spahn C., Herrmannsdörfer F., Kuner T., Heilemann M.
   Temporal accumulation analysis provides simplified artifact-free analysis of membrane-protein nanoclusters.
   Nat Methods. 2016 Nov 29;13(12):963-964. doi: 10.1038/nmeth.406

"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surepy import LocData
from surepy.analysis.analysis_base import _Analysis
from surepy.data.cluster.clustering import clustering_hdbscan
from surepy.data.filter import random_subset
from surepy.data.hulls import ConvexHull


#### The algorithms


def _accumulation_cluster_check_for_single_dataset(locdata, region_measure, algorithm=clustering_hdbscan,
                                                              algo_parameter=None, hull='bb'):
    """
    Compute localization density, relative area coverage by the clusters (eta), average density of localizations
    within apparent clusters (rho) for a single localization dataset.
    """
    # localization density
    localization_density = len(locdata) / region_measure

    # compute clusters
    if algo_parameter is None:
        algo_parameter = {}
    noise, clust = algorithm(locdata, noise=True, **algo_parameter)

    if len(clust)==0:
        # return localization_density, eta, rho
        return np.nan, np.nan, np.nan

    # compute cluster regions and densities
    if hull == 'bb':
        # Region_measure_bb has been computed upon instantiation
        if not 'Region_measure_bb' in clust.data.columns:
            # return localization_density, eta, rho
            return np.nan, np.nan, np.nan

        else:
            # relative area coverage by the clusters
            eta = clust.data['Region_measure_bb'].sum() / region_measure

            # average_localization_density_in_cluster
            rho = clust.data['Localization_density_bb'].mean()

    elif hull == 'ch':
        # compute hulls
        Hs = [ConvexHull(ref.coordinates) for ref in clust.references]
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


def _accumulation_cluster_check(locdata, region_measure='bb', algorithm=clustering_hdbscan,
                                algo_parameter=None, hull='bb', n_loc=10, divide='random', n_extrapolate=5):
    """
    Compute localization density, relative area coverage by the clusters (eta), average density of localizations
    within apparent clusters (rho) for the sequence of divided localization datasets.
    """
    # total region
    if isinstance(region_measure, str):
        region_measure_ = locdata.properties['Region_measure_' + region_measure]
    else:
        region_measure_ = region_measure

    if isinstance(n_loc, (list, tuple, np.ndarray)):
        if max(n_loc) <= len(locdata):
            numbers_loc = n_loc
        else:
            raise ValueError('The bins must be smaller than the total number of localizations in locdata.')
    else:
        numbers_loc = np.linspace(0, len(locdata), n_loc+1, dtype=int)[1:]

    # take random subsets of localizations
    if divide == 'random':
        locdatas = [random_subset(locdata, number_points=n_pts) for n_pts in numbers_loc]
    elif divide == 'sequential':
        locdatas = [LocData.from_selection(locdata, indices=range(n_pts)) for n_pts in numbers_loc]
    else:
        raise TypeError(f'String input {divide} for divide is not valid.')

    results_ = [_accumulation_cluster_check_for_single_dataset(locd,
                                                               region_measure=region_measure_, algorithm=algorithm,
                                                               algo_parameter=algo_parameter, hull=hull)
                for locd in locdatas]

    # linear regression to extrapolate rho_0
    results_ = np.asarray(results_)
    idx = np.all(np.isfinite(results_), axis=1)
    x = results_[idx, 0]  # position 0 being localization_density
    y = results_[idx, 2]  # position 2 being rho

    fit_coefficients = np.polyfit(x[0:n_extrapolate], y[0:n_extrapolate], deg=1)
    rho_zero = fit_coefficients[-1]

    if rho_zero <=0.:
        warnings.warn('Extrapolation of rho yields a negative value.', UserWarning)
        rho_zero = 1

    # combine results
    results_ = [np.append(entry, [rho_zero, entry[2]/rho_zero]) for entry in results_]
    results = pd.DataFrame(data = results_,
                           columns=['localization_density', 'eta', 'rho', 'rho_0', 'rho/rho_0'])
    return results


##### The specific analysis classes

class AccumulationClusterCheck(_Analysis):
    """
    Check for the presence of clusters in localization data by analyzing variations in cluster area and localization
    density within clusters.

    Parameters
    ----------
    locdata : LocData object
        Localization data that might be clustered.
    region_measure : float or str
        Region measure (area or volume) for the support of locdata. String can be any of standard hull identifiere.
    algorithm : callable
        Clustering algorithm.
    algo_parameter : dict
        Dictionary with kwargs for `algorithm`.
    hull : str
        Hull computation that is used to compute cluster region measures (area or volume). The identifier string can
        be one of the defined hulls.
    n_loc : int or sequence of scalars
        If n_loc is an int, it defines the number of localization subsets into which the total number of localizations
        are distributed.
        If n_loc is a sequence, it defines the number of localizations used for each localization subset.
    divide: str
        Identifier to choose how to partition the localization data. For `random` localizations are selected randomly.
        For `sequential` localizations are selected as chuncks of increasing size always starting from the first
        element.
    n_extrapolate : int
        The number of rho values taken to extrapolate rho_zero.

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
        localizations within apparent clusters (rho), and rho normalized to the extrapolated value of rho for
        localization_density=0 (rho_zero). If the extrapolation of rho yields a negative value rho_zero is set to 1.
    """
    count = 0

    def __init__(self, locdata, meta=None, region_measure='bb', algorithm=clustering_hdbscan, algo_parameter=None,
                 hull='bb', n_loc=10, divide='random', n_extrapolate=5):
        super().__init__(locdata=locdata, meta=meta, region_measure=region_measure,
                         algorithm=algorithm, algo_parameter=algo_parameter, hull=hull, n_loc=n_loc, divide=divide,
                         n_extrapolate=n_extrapolate)

    def compute(self):
        self.results = _accumulation_cluster_check(self.locdata, **self.parameter)
        return self

    def plot(self, ax=None, show=True, **kwargs):
        """
        Provide plot of results as matplotlib axes object.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image
        show : bool
            Flag indicating if plt.show() is active.
        kwargs : dict
            Other parameters passed to `matplotlib.pyplot.plot()`.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.plot().
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        self.results.plot(x='eta', y='rho/rho_0', ax=ax, **kwargs)

        ax.set(title = '',
               xlabel = 'Relative clustered area',
               ylabel = 'Normalized localization density within cluster'
               )

        # show figure
        if show:
            plt.show()

        return None
