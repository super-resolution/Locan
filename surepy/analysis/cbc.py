"""
Coordinate-based colocalization.

Colocalization is estimated by computing a colocalization index for each localization
using the so-called coordinate-based colocalization algorithm [1]_.

References
----------
.. [1] Malkusch S, Endesfelder U, Mondry J, Gelléri M, Verveer PJ, Heilemann M.,
   Coordinate-based colocalization analysis of single-molecule localization microscopy data.
   Histochem Cell Biol. 2012, 137(1):1-10.
   doi: 10.1007/s00418-011-0880-5
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

from surepy.analysis.analysis_base import _Analysis


##### The algorithm

def _coordinate_based_colocalization(points, other_points=None, radius=100, n_steps=10):
    """
    Compute a colocalization index for each localization by coordinate-based colocalization.

    Parameters
    ----------
    points : array of tuple
        Array of points (each point represented by a tuple with coordinates) for which CBC values are computed.

    other_points : array of tuple or None
        Array of points (each represented by a tuple with coordinates) to be compared with points. If None other_points
        are set to points.

    radius : int or float
        The maximum radius up to which nearest neighbors are determined

    n_steps : int
        The number of bins from which Spearman correlation is computed.

    Returns
    -------
    np.array
        An array with coordinate-based colocalization coefficients for each input point.
    """
    # sampled radii
    radii = np.linspace(0, radius, n_steps+1)

    # nearest neighbors within radius
    nneigh_1 = NearestNeighbors(radius=radius, metric='euclidean').fit(points)
    distances_1 = np.array(nneigh_1.radius_neighbors()[0])

    if other_points is None:
        nneigh_2 = NearestNeighbors(radius=radius, metric='euclidean').fit(points)
    else:
        nneigh_2 = NearestNeighbors(radius=radius, metric='euclidean').fit(other_points)
    distances_2 = np.array(nneigh_2.radius_neighbors(points)[0])

    # CBC for each point
    correlation = np.empty(len(points))
    for i, (d_1, d_2) in enumerate(zip(distances_1, distances_2)):
        if len(d_1) and len(d_2):
            # binning
            hist_1 = np.histogram(d_1, bins=radii, range=(0, radius))[0]
            hist_2 = np.histogram(d_2, bins=radii, range=(0, radius))[0]

            # normalization
            values_1 = np.cumsum(hist_1) * radius ** 2 / radii[1:] ** 2 / len(d_1)
            values_2 = np.cumsum(hist_2) * radius ** 2 / radii[1:] ** 2 / len(d_2)

            # Spearman rank correlation
            rho, pval = spearmanr(values_1, values_2)
            correlation[i] = rho
        else:
            correlation[i] = np.nan

    # CBC normalization for each point
    max_distances = np.array([np.max(d, initial=0) for d in distances_2])  # max is set to 0 for empty arrays.
    norm_spearmanr = np.exp(-1 * max_distances / radius)
    correlation = correlation * norm_spearmanr

    return correlation


##### The specific analysis classes

class CoordinateBasedColocalization(_Analysis):
    """
    Compute a colocalization index for each localization by coordinate-based colocalization (CBC).

    The colocalization index is calculated for each localization in `locdata` by finding nearest neighbors in
    `locdata` or `other_locdata` within `radius`. A normalized number of nearest neighbors at a certain radius is
    computed for `n_steps` equally-sized steps of increasing radii ranging from 0 to `radius`.
    The Spearman rank correlation coefficent is computed for these values and weighted by
    Exp[-nearestNeighborDistance/distanceMax].

    Parameters
    ----------
    locdata : LocData object
        Localization data for which CBC values are computed.
    other_locdata : LocData object or None
        Localization data to be colocalized. If None other_locdata is set to locdata.
    radius : int or float
        The maximum radius up to which nearest neighbors are determined
    n_steps : int
        The number of bins from which Spearman correlation is computed.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : pandas DataFrame
        Coordinate-based colocalization coefficients for each input point.
    """
    count = 0

    def __init__(self, locdata, other_locdata=None, radius=100, n_steps=10, meta=None):
        super().__init__(locdata=locdata, other_locdata=other_locdata, radius=radius, n_steps=n_steps, meta=meta)
        self.results = None

    def compute(self):
        points = self.locdata.coordinates

        # turn other_locdata into other_points
        new_parameter = {key: self.parameter[key] for key in self.parameter if key is not 'other_locdata'}

        if self.parameter['other_locdata'] is not None:
            other_points = self.parameter['other_locdata'].coordinates
            id = self.parameter['other_locdata'].meta.identifier
        else:
            other_points = None
            id = 'self'

        self.results = pd.DataFrame({f'colocalization_cbc_{id}':
                                         _coordinate_based_colocalization(points, other_points, **new_parameter)})
        return self
