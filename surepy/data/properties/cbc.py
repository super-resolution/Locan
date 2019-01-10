"""
Coordinate-based colocalization.

Colocalization is estimated by computing a colocalization index for each localization
using the so-called coordinate-based colocalization algorithm [1]_.

References
----------
.. [1]: Malkusch S, Endesfelder U, Mondry J, Gelléri M, Verveer PJ, Heilemann M.,
   Coordinate-based colocalization analysis of single-molecule localization microscopy data.
   Histochem Cell Biol. 2012, 137(1):1-10.
   doi: 10.1007/s00418-011-0880-5
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr


def coordinate_based_colocalization(points_1, points_2, radius=100, n_steps=10):
    """
    Compute a colocalization index for each localization by coordinate-based colocalization.

    The colocalization index is calculated for each point in <points_1> by finding nearest neighbors in <points_1>
    or <points_2> within radius. A normalized number of nearest neighbors at a certain radius is computer for
    n_steps of increasing radii ranging from 0 to <radius> in steps of <n_steps>.
    The Spearman rank correlation coefficent is computed for these values and weighted by
    Exp[-nearestNeighborDistance/distanceMax].

    Parameters
    ----------
    points_1 : array of tuple
        Array of points (each point represented by a tuple with coordinates) for which CBC values are computed.

    points_2 : array of tuple
        Array of points (each represented by a tuple with coordinates) to be compared with points_1

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
    nneigh_1 = NearestNeighbors(radius=radius, metric='euclidean').fit(points_1)
    distances_1 = np.array(nneigh_1.radius_neighbors()[0])

    nneigh_2 = NearestNeighbors(radius=radius, metric='euclidean').fit(points_2)
    distances_2 = np.array(nneigh_2.radius_neighbors(points_1)[0])

    # CBC for each point
    correlation = np.empty(len(points_1))
    for i, (d_1, d_2) in enumerate(zip(distances_1, distances_2)):
        # binning
        hist_1 = np.histogram(d_1, bins=radii, range=(0, radius))[0]
        hist_2 = np.histogram(d_2, bins=radii, range=(0, radius))[0]

        # normalization
        values_1 = np.cumsum(hist_1) * radius**2 / radii[1:]**2 / len(d_1)
        values_2 = np.cumsum(hist_2) * radius**2 / radii[1:]**2 / len(d_2)

        # Spearman rank correlation
        rho, pval = spearmanr(values_1, values_2)
        correlation[i] = rho

    # CBC normalization for each point
    max_distances = np.array([np.max(d) for d in distances_2])
    norm_spearmanr = np.exp(-1 * max_distances / radius)
    correlation = correlation * norm_spearmanr

    return correlation
