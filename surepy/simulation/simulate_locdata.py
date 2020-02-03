"""

Simulate localization data.

This module provides functions to simulate localization data and return LocData objects.

Functions that are named as make_* provide point data arrays. Functions that are named as simulate_* provide
locdata.

Use simulate_csr to get localizations that are spatially distributed by a Poisson process (i.e. homogeneous).

Use simulate_Matern to get localizations that are homogeneously distributed in spherical clusters with cluster centers
being homogeneously distributed.

Use simulate_blobs to get localizations that are normally distributed in spherical clusters with cluster centers
being homogeneously distributed.


Localizations are often distributed either by a spatial process of complete-spatial randomness or following a
Neyman-Scott process [1]_. For a Neyman-Scott process parent events (representing single emitters) yield a random number
of offspring events (representing localizations due to repeated blinking). The total number of emitters is specified
by <Parent Event Number>. The number of offspring events is Poisson distributed with mean <Offspring Number>.
The spatial distribution of parent events is distributed according to complete-spatial randomness;
the spatial distribution of offspring events is Gauss-distributed with the given <Standard Deviation> for each emitter.

Intensity distributions are often simulated in the following way: Each localization is given an emission strength drawn
from an intensity distribution that is either "Exponential" or "Poisson" with <Mean Intensity>.

Parts of this code is adapted from scikit-learn/sklearn/datasets/samples_generator.py .

References
----------
.. [1] Neyman, J. & Scott, E. L.,
   A Theory of the Spatial Distribution of Galaxies.
   Astrophysical Journal 1952, vol. 116, p.144.

"""

import sys
import time

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from surepy.data.locdata import LocData
from surepy.data import metadata_pb2
from surepy.data.region import RoiRegion


__all__ = ['make_csr', 'simulate_csr',
           'make_csr_on_disc', 'simulate_csr_on_disc',
           'make_Matern', 'simulate_Matern',
           'make_Thomas', 'simulate_Thomas',
           'make_csr_on_region', 'simulate_csr_on_region',
           'make_Thomas_on_region', 'simulate_Thomas_on_region',
           'simulate_tracks', 'resample']


def make_csr(n_samples=100, n_features=2, feature_range=(0, 1.), seed=None):
    """
    Provide points that are spatially-distributed by complete spatial
    randomness within the boundarys given by `feature_range`.

    Parameters
    ----------
    n_samples : int
        total number of localizations
    n_features : int
        The number of features for each sample.
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each feature. If sequence the number of elements but be equal to n_features.
    seed : int
        random number generation seed

    Returns
    -------
    array of shape [n_samples, n_features]
        The generated samples.
    """
    if seed is not None:
        np.random.seed(seed)

    if len(np.shape(feature_range)) == 1:
        samples = np.random.uniform(*feature_range, size=(n_samples, n_features))
    else:
        if np.shape(feature_range)[0] != n_features:
            raise ValueError(f'The number of feature_range elements (if sequence) must be equal to n_features.')
        else:
            samples = np.random.rand(n_samples, n_features)
            for i, (low, high) in enumerate(feature_range):
                if low < high:
                    samples[:, i] = samples[:, i] * (high - low) + low
                else:
                    raise ValueError(f'The first value of feature_range {low} must be smaller than the second one '
                                     f'{high}.')
    return samples


def simulate_csr(n_samples=100, n_features=2, feature_range=(0, 1.), seed=None):
    """
    Provide a dataset of localizations with coordinates that are spatially-distributed by complete spatial
    randomness within the boundarys given by `feature_range`.

    Parameters
    ----------
    n_samples : int
        total number of localizations
    n_features : int
        The number of features for each sample. The first three features are taken as `Position_x`, `Position_y`, and
        `Position_z`.
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each feature. If sequence the number of elements but be equal to n_features.
    seed : int
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    parameter = locals()

    samples = make_csr(n_samples=n_samples, n_features=n_features, feature_range=feature_range, seed=seed)

    property_names = []
    for i in range(n_features):
        if i == 0:
            property_names.append('position_x')
        elif i == 1:
            property_names.append('position_y')
        elif i == 2:
            property_names.append('position_z')
        else:
            property_names.append(f'feature_{i - 3}')

    dict_ = {}
    for name, data in zip(property_names, samples.T):
        dict_.update({name: data})

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(dict_))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return locdata


def make_csr_on_disc(n_samples=100, radius=1.0, seed=None):
    """
    Provide points that are spatially-distributed on a disc by complete spatial randomness.

    Parameters
    ----------
    n_samples : int
       total number of localizations
    radius : float
        radius of the disc
    seed : int
       random number generation seed

    Returns
    -------
    array of shape [n_samples, 2]
       The generated samples.
    """
    if seed is not None:
        np.random.seed(seed)

    # angular and radial coordinates of Poisson points
    theta = np.random.rand(n_samples) * 2 * np.pi
    rho = radius * np.sqrt(np.random.rand(n_samples))

    # Convert from polar to Cartesian coordinates
    xx = rho * np.cos(theta)
    yy = rho * np.sin(theta)

    samples = np.array((xx, yy)).T
    return samples


def simulate_csr_on_disc(n_samples=100, radius=1.0, seed=None):
    """
    Provide a dataset of localizations with coordinates that are spatially-distributed on a disc by complete spatial
    randomness..

    Parameters
    ----------
    n_samples : int
       total number of localizations
    radius : float
        radius of the disc
    seed : int
       random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    parameter = locals()

    samples = make_csr_on_disc(n_samples=n_samples, radius=radius, seed=seed)

    property_names = ['position_x', 'position_y']

    dict_ = {}
    for name, data in zip(property_names, samples.T):
        dict_.update({name: data})

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(dict_))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return locdata


def make_Matern(n_samples=100, n_features=2, centers=None, radius=1.0, feature_range=(-10.0, 10.0),
                shuffle=True, seed=None):
    """
    Generate spots with equally distributed points inside. Centers are spatially-distributed by complete spatial
    randomness within the boundaries given by `feature_range`. The number of dimensions is specified by `n_features`.

    Parameters
    ----------
    n_samples : int or array-like
        If int, it is the total number of points equally divided among clusters.
        If array-like, each element of the sequence indicates the number of samples per cluster.
    n_features : int
        The number of features for each sample. One of (1, 2, 3).
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
        If centers is an array, n_features is taken from centers shape.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be either None or an array of length equal to the length of n_samples.
    radius : float or sequence of floats
        The radius for the spots. If tuple, the number of elements must be equal to the number of centers.
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each cluster center when centers are
        generated at random. If sequence the number of elements must be equal to n_features.
    shuffle : boolean
        Shuffle the samples.
    seed : int
        random number generation seed

    Returns
    -------
    samples : array of shape [n_samples, n_features]
        The generated samples.
    labels : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """
    if seed is not None:
        np.random.seed(seed)

    # check n_feature consistent with feature_range
    if (len(np.shape(feature_range)) != 1) and (np.shape(feature_range)[0] != n_features):
        raise ValueError(f'The number of feature_range elements (if sequence) must be equal to n_features.')

    # n_samples, centers, n_centers
    if isinstance(n_samples, (int, np.integer)):
        # Set n_centers by looking at centers arg
        if centers is None:
            centers = 3

        if isinstance(centers, (int, np.integer)):
            n_centers = centers
            centers = make_csr(n_samples=n_centers, n_features=n_features, feature_range=feature_range, seed=seed)
        else:  # if centers is array
            if n_features != np.shape(centers)[1]:
                raise ValueError(f'n_features must be the same as the dimensions for each center. '
                                 f'Got n_features: {n_features} and center dimensions: {np.shape(centers)[1]} instead.')
            n_centers = np.shape(centers)[0]

    else:  # if n_samples is array
        n_centers = len(n_samples)  # Set n_centers by looking at [n_samples] arg
        if centers is None:
            centers = make_csr(n_samples=n_centers, n_features=n_features, feature_range=feature_range, seed=seed)
        elif isinstance(centers, (int, np.integer)):
            if centers != len(n_samples):
                raise ValueError(f"Length of `n_samples` not consistent"
                                 f" with number of centers. Got length of n_samples = {n_centers} "
                                 f"and centers = {centers}")
            centers = make_csr(n_samples=centers, n_features=n_features, feature_range=feature_range, seed=seed)
        else:  # if centers is array
            try:
                assert len(centers) == n_centers
            except TypeError:
                raise ValueError(f"Parameter `centers` must be array-like or None. "
                                 f"Got {centers} instead")
            except AssertionError:
                raise ValueError(f"Length of `n_samples` not consistent"
                                 f" with number of centers. Got length of n_samples = {n_centers} "
                                 f"and number of centers = {len(centers)}")
            if n_features != np.shape(centers)[1]:
                raise ValueError(f'n_features must be the same as the dimensions for each center. '
                                 f'Got n_features: {n_features} and center dimensions: {centers.shape[1]} instead.')

    # set n_samples_per_center
    if isinstance(n_samples, (int, np.integer)):
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
    else:
        n_samples_per_center = n_samples

    # radius: if radius is given as list, it must be consistent with the n_centers
    if hasattr(radius, "__len__"):
        if len(radius) != n_centers:
            raise ValueError(f"Length of `radius` not consistent with "
                             f"number of centers. Got number of centers = {n_centers} "
                             f"and radius = {radius}")
        else:
            radii = radius
    else:  # if isinstance(radius, (float, np.float)):
        radii = np.full(len(centers), radius)

    # discs
    disk_samples = []
    labels = []
    if n_features == 1:
        raise NotImplementedError
    elif n_features == 3:
        raise NotImplementedError
    elif n_features == 2:
        for i, (number, r, center) in enumerate(zip(n_samples_per_center, radii, centers)):
            pts = make_csr_on_disc(n_samples=number, radius=r, seed=seed)
            pts = pts + center
            disk_samples.append(pts)
            labels += [i] * number

    # shift and concatenate
    samples = np.concatenate(disk_samples)
    labels = np.array(labels)

    # shuffle
    if shuffle:
        total_n_samples = np.sum(n_samples)
        indices = np.arange(total_n_samples)
        np.random.shuffle(indices)
        samples = samples[indices]
        labels = labels[indices]

    return samples, labels


def simulate_Matern(n_samples=100, n_features=2, centers=None, radius=1.0, feature_range=(-10.0, 10.0),
                    shuffle=True, seed=None):
    """
    Provide a dataset of localizations with coordinates and labels that are spatially-distributed spots with
    homogeneously distributed points inside. Centers are spatially-distributed by complete spatial
    randomness within the boundaries given by `feature_range`. The number of dimensions is specified by `n_features`.

    Parameters
    ----------
    n_samples : int or array-like
        If int, it is the total number of points equally divided among clusters.
        If array-like, each element of the sequence indicates the number of samples per cluster.
    n_features : int
        The number of features for each sample. One of (1, 2, 3).
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
        If centers is an array, n_features is taken from centers shape.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be either None or an array of length equal to the length of n_samples.
    radius : float or sequence of floats
        The radius for the spots. If tuple, the number of elements must be equal to the number of centers.
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each cluster center when centers are
        generated at random. If sequence the number of elements must be equal to n_features.
    shuffle : boolean
        Shuffle the samples.
    seed : int
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    parameter = locals()

    if seed is not None:
        np.random.seed(seed)

    samples, labels = make_Matern(n_samples=n_samples, n_features=n_features, centers=centers, radius=radius,
                                  feature_range=feature_range, shuffle=shuffle, seed=seed)

    property_names = []
    for i in range(n_features):
        if i == 0:
            property_names.append('position_x')
        elif i == 1:
            property_names.append('position_y')
        elif i == 2:
            property_names.append('position_z')
        else:
            property_names.append(f'feature_{i - 3}')

    dict_ = {}
    for name, data in zip(property_names, samples.T):
        dict_.update({name: data})
    dict_.update({'cluster_label': labels})

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(dict_))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return locdata


def make_Thomas(n_samples=100, n_features=2, centers=None, cluster_std=1.0, feature_range=(-10.0, 10.0),
                shuffle=True, seed=None):
    """
    Generate spots with normally distributed points inside. Centers are spatially-distributed by complete spatial
    randomness within the boundaries given by `feature_range`. The number of dimensions is specified by `n_features`.

    Parameters
    ----------
    n_samples : int or array-like
        If int, it is the total number of points equally divided among clusters.
        If array-like, each element of the sequence indicates the number of samples per cluster.
    n_features : int
        The number of features for each sample. One of (1, 2, 3).
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
        If centers is an array, n_features is taken from centers shape.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be either None or an array of length equal to the length of n_samples.
    cluster_std : float or sequence of floats
        The standard deviation for the spots. If sequence, the number of elements must be equal to the number of centers.
        # todo add different std for each feature
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each cluster center when centers are
        generated at random. If sequence the number of elements must be equal to n_features.
    shuffle : boolean
        Shuffle the samples.
    seed : int
        random number generation seed

    Returns
    -------
    samples : array of shape [n_samples, n_features]
        The generated samples.
    labels : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """
    if seed is not None:
        np.random.seed(seed)

    # check n_feature consistent with feature_range
    if (len(np.shape(feature_range)) != 1) and (np.shape(feature_range)[0] != n_features):
        raise ValueError(f'The number of feature_range elements (if sequence) must be equal to n_features.')

    # n_samples, centers, n_centers
    if isinstance(n_samples, (int, np.integer)):
        # Set n_centers by looking at centers arg
        if centers is None:
            centers = 3

        if isinstance(centers, (int, np.integer)):
            n_centers = centers
            centers = make_csr(n_samples=n_centers, n_features=n_features, feature_range=feature_range, seed=seed)
        else:  # if centers is array
            if n_features != np.shape(centers)[1]:
                raise ValueError(f'n_features must be the same as the dimensions for each center. '
                                 f'Got n_features: {n_features} and center dimensions: {np.shape(centers)[1]} instead.')
            n_centers = np.shape(centers)[0]

    else:  # if n_samples is array
        n_centers = len(n_samples)  # Set n_centers by looking at [n_samples] arg
        if centers is None:
            centers = make_csr(n_samples=n_centers, n_features=n_features, feature_range=feature_range, seed=seed)
        elif isinstance(centers, (int, np.integer)):
            if centers != len(n_samples):
                raise ValueError(f"Length of `n_samples` not consistent"
                                 f" with number of centers. Got length of n_samples = {n_centers} "
                                 f"and centers = {centers}")
            centers = make_csr(n_samples=centers, n_features=n_features, feature_range=feature_range, seed=seed)
        else:  # if centers is array
            try:
                assert len(centers) == n_centers
            except TypeError:
                raise ValueError(f"Parameter `centers` must be array-like or None. "
                                 f"Got {centers} instead")
            except AssertionError:
                raise ValueError(f"Length of `n_samples` not consistent"
                                 f" with number of centers. Got length of n_samples = {n_centers} "
                                 f"and number of centers = {len(centers)}")
            if n_features != np.shape(centers)[1]:
                raise ValueError(f'n_features must be the same as the dimensions for each center. '
                                 f'Got n_features: {n_features} and center dimensions: {centers.shape[1]} instead.')

    # set n_samples_per_center
    if isinstance(n_samples, (int, np.integer)):
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
    else:
        n_samples_per_center = n_samples

    # cluster_std: if cluster_std is given as list, it must be consistent with the n_centers
    if hasattr(cluster_std, "__len__"):
        if len(cluster_std) != n_centers:
            raise ValueError(f"Length of `cluster_std` not consistent with "
                             f"number of centers. Got number of centers = {n_centers} "
                             f"and cluster_std = {cluster_std}")
        else:
            cluster_std_list = cluster_std
    else:  # if isinstance(radius, (float, np.float)):
        cluster_std_list = np.full(len(centers), cluster_std)

    # normal-distributed spots
    spot_samples = []
    labels = []
    for i, (center, std, number) in enumerate(zip(centers, cluster_std_list, n_samples_per_center)):
        pts = np.random.normal(loc=center, scale=std, size=(number, n_features))
        spot_samples.append(pts)
        labels += [i] * number

    # concatenate
    samples = np.concatenate(spot_samples)
    labels = np.array(labels)

    # shuffle
    if shuffle:
        total_n_samples = np.sum(n_samples)
        indices = np.arange(total_n_samples)
        np.random.shuffle(indices)
        samples = samples[indices]
        labels = labels[indices]

    return samples, labels


def simulate_Thomas(n_samples=100, n_features=2, centers=None, cluster_std=1.0, feature_range=(-10.0, 10.0),
                    shuffle=True, seed=None):
    """
    Provide a dataset of localizations with coordinates and labels that are spatially-distributed spots with
    with normally distributed points inside. Centers are spatially-distributed by complete spatial
    randomness within the boundaries given by `feature_range`. The number of dimensions is specified by `n_features`.

    Parameters
    ----------
    n_samples : int or array-like
        If int, it is the total number of points equally divided among clusters.
        If array-like, each element of the sequence indicates the number of samples per cluster.
    n_features : int
        The number of features for each sample. One of (1, 2, 3).
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
        If centers is an array, n_features is taken from centers shape.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be either None or an array of length equal to the length of n_samples.
    cluster_std : float or sequence of floats
        The standard deviation for the spots. If sequence, the number of elements must be equal to the number of centers.
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each cluster center when centers are
        generated at random. If sequence the number of elements must be equal to n_features.
    shuffle : boolean
        Shuffle the samples.
    seed : int
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    parameter = locals()

    if seed is not None:
        np.random.seed(seed)

    samples, labels = make_Thomas(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std,
                                  feature_range=feature_range, shuffle=shuffle, seed=seed)

    property_names = []
    for i in range(n_features):
        if i == 0:
            property_names.append('position_x')
        elif i == 1:
            property_names.append('position_y')
        elif i == 2:
            property_names.append('position_z')
        else:
            property_names.append(f'feature_{i - 3}')

    dict_ = {}
    for name, data in zip(property_names, samples.T):
        dict_.update({name: data})
    dict_.update({'cluster_label': labels})

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(dict_))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return locdata


# todo add 3D
def make_csr_on_region(region, n_samples=100, seed=None):
    """
    Provide points that are spatially-distributed inside the specified region by complete spatial randomness.

    Parameters
    ----------
    region : RoiRegion Object, or dict
        Region of interest as specified by RoiRegion or dictionary with keys `region_specs` and `region_type`.
        Allowed values for `region_specs` and `region_type` are defined in the docstrings for `Roi` and `RoiRegion`.
    n_samples : int
       total number of localizations
    seed : int
       random number generation seed

    Returns
    -------
    array of shape [n_samples, 2]
       The generated samples.
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(region, dict):
        region_ = RoiRegion(region_specs=region['region_specs'], region_type=region['region_type'])
    else:
        region_ = region

    # todo: add function .bounds to RoiRegion
    shapely_polygon = Polygon(region_.polygon)
    min_x, min_y, max_x, max_y = shapely_polygon.bounds
    bounding_box = ((min_x, max_x), (min_y, max_y))

    n_remaining = n_samples
    samples = []
    while n_remaining > 0:
        new_samples = np.random.rand(n_samples, region_.dimension)
        for i, (low, high) in enumerate(bounding_box):
            if low < high:
                new_samples[:, i] = new_samples[:, i] * (high - low) + low
            else:
                raise ValueError(f'The first value of feature_range {low} must be smaller than the second one '
                                 f'{high}.')
        new_samples = new_samples[region_.contains(new_samples)]
        samples.append(new_samples)
        n_remaining = n_remaining - len(new_samples)

    samples = np.concatenate(samples)
    samples = samples[0: n_samples]
    return samples


def simulate_csr_on_region(region, n_samples=100, seed=None):
    """
    Provide a dataset of localizations with coordinates that are spatially-distributed inside teh specified region by
    complete spatial randomness..

    Parameters
    ----------
    region : RoiRegion Object, or dict
        Region of interest as specified by RoiRegion or dictionary with keys `region_specs` and `region_type`.
        Allowed values for `region_specs` and `region_type` are defined in the docstrings for `Roi` and `RoiRegion`.
    n_samples : int
       total number of localizations
    seed : int
       random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    parameter = locals()

    samples = make_csr_on_region(region=region, n_samples=n_samples, seed=seed)

    property_names = []
    for i in range(np.shape(samples)[-1]):
        if i == 0:
            property_names.append('position_x')
        elif i == 1:
            property_names.append('position_y')
        elif i == 2:
            property_names.append('position_z')

    dict_ = {}
    for name, data in zip(property_names, samples.T):
        dict_.update({name: data})

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(dict_))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return locdata


def make_Thomas_on_region(region, n_samples=100, centers=None, cluster_std=1.0,
                          shuffle=True, seed=None):
    """
    Provide points that are spatially-distributed spots
    with normally distributed points inside. Centers are spatially-distributed by complete spatial
    randomness within the region. The number of dimensions also called `n_features` is given by the `region` dimensions.

    Parameters
    ----------
    region : RoiRegion Object, or dict
        Region of interest as specified by RoiRegion or dictionary with keys `region_specs` and `region_type`.
        Allowed values for `region_specs` and `region_type` are defined in the docstrings for `Roi` and `RoiRegion`.
    n_samples : int or array-like
        If int, it is the total number of points equally divided among clusters.
        If array-like, each element of the sequence indicates the number of samples per cluster.
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
        If centers is an array, n_features is taken from centers shape.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be either None or an array of length equal to the length of n_samples.
    cluster_std : float or sequence of floats
        The standard deviation for the spots. If sequence, the number of elements must be equal to the number of centers.
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each cluster center when centers are
        generated at random. If sequence the number of elements must be equal to n_features.
    shuffle : boolean
        Shuffle the samples.
    seed : int
        random number generation seed

    Returns
    -------
    array of shape [n_samples, 2]
       The generated samples.
    """
    if isinstance(region, dict):
        region_ = RoiRegion(region_specs=region['region_specs'], region_type=region['region_type'])
    else:
        region_ = region

    if isinstance(n_samples, (int, np.integer)):
        # Set n_centers by looking at centers arg
        if centers is None:
            centers = 3

        if isinstance(centers, (int, np.integer)):
            n_centers = centers
            centers = make_csr_on_region(region=region, n_samples=n_centers, seed=seed)
        else:  # if centers is array
            if region_.dimension != np.shape(centers)[1]:
                raise ValueError(f'Region dimensions must be the same as the dimensions for each center. '
                                 f'Got region dimension: {region_.dimension} and '
                                 f'center dimensions: {np.shape(centers)[1]} instead.')
            n_centers = np.shape(centers)[0]
            # todo add check if centers in region else raise ValueError

    else:  # if n_samples is array
        n_centers = len(n_samples)  # Set n_centers by looking at [n_samples] arg
        if centers is None:
            centers = make_csr_on_region(region=region, n_samples=n_centers, seed=seed)
        elif isinstance(centers, (int, np.integer)):
            if centers != len(n_samples):
                raise ValueError(f"Length of `n_samples` not consistent"
                                 f" with number of centers. Got length of n_samples = {n_centers} "
                                 f"and centers = {centers}")
            centers = make_csr_on_region(region=region, n_samples=centers, seed=seed)
        else:  # if centers is array
            try:
                assert len(centers) == n_centers
            except TypeError:
                raise ValueError(f"Parameter `centers` must be array-like or None. "
                                 f"Got {centers} instead")
            except AssertionError:
                raise ValueError(f"Length of `n_samples` not consistent"
                                 f" with number of centers. Got length of n_samples = {n_centers} "
                                 f"and number of centers = {len(centers)}")
            if region_.dimension != np.shape(centers)[1]:
                raise ValueError(f'Region dimensions must be the same as the dimensions for each center. '
                                 f'Got Region dimensions: {region_.dimension} and '
                                 f'center dimensions: {centers.shape[1]} instead.')

        # set n_samples_per_center
    if isinstance(n_samples, (int, np.integer)):
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
    else:
        n_samples_per_center = n_samples

    samples, labels = make_Thomas(n_samples=n_samples_per_center, n_features=region_.dimension, centers=centers,
                                  cluster_std=cluster_std, shuffle=shuffle, seed=seed)

    return samples, labels


def simulate_Thomas_on_region(region, n_samples=100, centers=None, cluster_std=1.0,
                              shuffle=True, seed=None):
    """
    Provide a dataset of localizations with coordinates and labels that are spatially-distributed spots
    with normally distributed points inside. Centers are spatially-distributed by complete spatial
    randomness within the region. The number of dimensions also called `n_features` is given by the `region` dimensions.

    Parameters
    ----------
    region : RoiRegion Object, or dict
        Region of interest as specified by RoiRegion or dictionary with keys `region_specs` and `region_type`.
        Allowed values for `region_specs` and `region_type` are defined in the docstrings for `Roi` and `RoiRegion`.
    n_samples : int or array-like
        If int, it is the total number of points equally divided among clusters.
        If array-like, each element of the sequence indicates the number of samples per cluster.
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
        If centers is an array, n_features is taken from centers shape.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be either None or an array of length equal to the length of n_samples.
    cluster_std : float or sequence of floats
        The standard deviation for the spots. If sequence, the number of elements must be equal to the number of centers.
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each cluster center when centers are
        generated at random. If sequence the number of elements must be equal to n_features.
    shuffle : boolean
        Shuffle the samples.
    seed : int
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    parameter = locals()

    samples, labels = make_Thomas_on_region(region=region, n_samples=n_samples, centers=centers,
                                            cluster_std=cluster_std, shuffle=shuffle, seed=seed)

    property_names = []
    for i in range(np.shape(samples)[-1]):
        if i == 0:
            property_names.append('position_x')
        elif i == 1:
            property_names.append('position_y')
        elif i == 2:
            property_names.append('position_z')

    dict_ = {}
    for name, data in zip(property_names, samples.T):
        dict_.update({name: data})

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(dict_))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return locdata


def _random_walk(n_walks=1, n_steps=10, dimensions=2, diffusion_constant=1, time_step=10):
    """
    Random walk simulation

    Parameters
    ----------
    n_walks: int
        Number of walks
    n_steps : int
        Number of time steps (i.e. frames)
    dimensions : int
        spatial dimensions to simulate
    diffusion_constant : int or float
        Diffusion constant in units length per seconds^2
    time_step : float
        Time per frame (or simulation step) in seconds.

    Returns
    -------
    tuple of arrays
        (times, positions), where shape(times) is 1 and shape of positions is (n_walks, n_steps, dimensions)
    """
    # equally spaced time steps
    times = np.arange(n_steps) * time_step

    # random step sizes according to the diffusion constant
    random_numbers = np.random.randint(0, 2, size=(n_walks, n_steps, dimensions))
    step_size = np.sqrt(2 * dimensions * diffusion_constant * time_step)
    steps = np.where(random_numbers == 0, -step_size, +step_size)

    # walker positions
    positions = np.cumsum(steps, axis=1)

    return times, positions


def simulate_tracks(n_walks=1, n_steps=10, ranges=((0, 10000), (0, 10000)), diffusion_constant=1,
                    time_step=10, seed=None):
    """
    Provide a dataset of localizations representing random walks with starting points being spatially-distributed
    on a rectangular shape or cubic volume by complete spatial randomness.

    Parameters
    ----------
    n_walks: int
        Number of walks
    n_steps : int
        Number of time steps (i.e. frames)
    ranges : tuple of tuples of two ints
        the range for valid x[, y, z]-coordinates
    diffusion_constant : int or float
        Diffusion constant with unit length per seconds^2
    time_step : float
        Time per frame (or simulation step) in seconds.
    seed : int
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    parameter = locals()

    if seed is not None:
        np.random.seed(seed)

    start_positions = np.array([np.random.uniform(*_range, size=n_walks) for _range in ranges]).T

    times, positions = _random_walk(n_walks=n_walks, n_steps=n_steps, dimensions=len(ranges),
                                    diffusion_constant=diffusion_constant, time_step=time_step)

    new_positions = np.concatenate([
        start_position + position
        for start_position, position in zip(start_positions, positions)
    ])

    locdata_dict = {
        'position_' + label: position_values
        for _, position_values, label in zip(ranges, new_positions.T, ('x', 'y', 'z'))
    }

    locdata_dict.update(frame=np.tile(range(len(times)), n_walks))

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(locdata_dict))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return locdata


def resample(locdata, n_samples=10):
    """
    Resample locdata according to localization uncertainty. Per localization *n_samples* new localizations
    are simulated normally distributed around the localization coordinates with a standard deviation set to the
    uncertainty in each dimension.
    The resulting LocData object carries new localizations with the following properties: 'Position_x',
    'Position_y'[, 'Position_z'], 'Origin_index'

    Parameters
    ----------
    locdata : LocData object
        Localization data to be resampled
    n_samples : int
        The number of localizations generated for each original localization.

    Returns
    -------
    locdata : LocData object
        New localization data with simulated coordinates.
    """

    # generate dataframe
    list = []
    for i in range(len(locdata)):
        new_d = {}
        new_d.update({'origin_index': np.full(n_samples, i)})
        x_values = np.random.normal(loc=locdata.data.iloc[i]['position_x'],
                                    scale=locdata.data.iloc[i]['uncertainty_x'],
                                    size=n_samples
                                    )
        new_d.update({'position_x': x_values})

        y_values = np.random.normal(loc=locdata.data.iloc[i]['position_y'],
                                    scale=locdata.data.iloc[i]['uncertainty_y'],
                                    size=n_samples
                                    )
        new_d.update({'position_y': y_values})

        try:
            z_values = np.random.normal(loc=locdata.data.iloc[i]['position_z'],
                                        scale=locdata.data.iloc[i]['uncertainty_z'],
                                        size=n_samples
                                        )
            new_d.update({'position_z': z_values})
        except KeyError:
            pass

        list.append(pd.DataFrame(new_d))

    dataframe = pd.concat(list, ignore_index=True)

    # metadata
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
    meta_.history.add(name='resample',
                      parameter='locdata={}, n_samples={}'.format(locdata, n_samples))

    # instantiate
    new_locdata = LocData.from_dataframe(dataframe=dataframe, meta=meta_)

    return new_locdata
