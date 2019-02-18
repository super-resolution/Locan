"""

Simulate localization data.

This module provides functions to simulate localization data and return LocData objects.

Functions that are named as make_* provide point data arrays. Functions that are named as simulate_* provide
locdata.

Use simulate_csr to get localizations that are spatially distributed by a Poisson process (i.e. homogeneous).

Use simulate_spots to get localizations that are homogeneously distributed in spherical clusters with cluster centers
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
from sklearn.datasets.samples_generator import make_blobs

from surepy import LocData
from surepy.data import metadata_pb2

#todo: correct history updates


def make_csr(n_samples=100, n_features=2, feature_range=(0, 1.), seed = None):
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
                    samples[:,i] = samples[:,i] * (high - low) + low
                else:
                    raise ValueError(f'The first value of feature_range {low} must be smaller than the second one '
                                     f'{high}.')
    return samples


def simulate_csr(n_samples=100, n_features=2, feature_range=(0, 1.), seed = None):
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
            property_names.append('Position_x')
        elif i == 1:
            property_names.append('Position_y')
        elif i == 2:
            property_names.append('Position_z')
        else:
            property_names.append(f'Feature_{i-3}')

    dict = {}
    for name, data in zip(property_names, samples.T):
        dict.update({name: data})

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(dict))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return locdata



def simulate_csr_(n_samples = 10000, x_range = (0,10000), y_range = (0,10000), z_range = None, seed=None):
    """
    Provide a dataset of localizations that are spatially-distributed on a rectangular shape or cubic volume by
    complete spatial randomness.

    Parameters
    ----------
    n_samples : int
        total number of localizations

    x_range : tuple of two ints
        the range for valid x-coordinates

    y_range : tuple of two ints
        the range for valid y-coordinates

    z_range : tuple of two ints
        the range for valid z-coordinates

    seed : int
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """

    if seed is not None:
        np.random.seed(seed)

    dict = {}
    for i, j in zip(('x', 'y', 'z'), (x_range, y_range, z_range)):
        if j is not None:
            dict.update({'Position_' + i: np.random.uniform(*j, n_samples)})

    dat = LocData.from_dataframe(dataframe=pd.DataFrame(dict))

    # metadata
    dat.meta.source = metadata_pb2.SIMULATION
    del dat.meta.history[:]
    dat.meta.history.add(name='simulate_csr',
                         parameter='n_samples={}, x_range={}, y_range={}, z_range={}, seed={}'.format(
                             n_samples, x_range, y_range, z_range, seed))

    return dat







def simulate_csr_(n_samples=100, n_features=2, feature_range=(-10.0, 10.0), seed = None):
    """
    Provide a dataset of localizations that are spatially-distributed by complete spatial
    randomness within the boundarys given by `feature_range`.

    Parameters
    ----------
    n_samples : int
        total number of localizations
    n_features : int
        The number of features for each sample.
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each cluster center when centers are
        generated at random. If sequence the number of elements but be equal to n_features.
    seed : int
        random number generation seed

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """

    if seed is not None:
        np.random.seed(seed)

    dict = {}
    for i, j in zip(('x', 'y', 'z'), (x_range, y_range, z_range)):
        if j is not None:
            dict.update({'Position_' + i: np.random.uniform(*j, n_samples)})

    dat = LocData.from_dataframe(dataframe=pd.DataFrame(dict))

    # metadata
    dat.meta.source = metadata_pb2.SIMULATION
    del dat.meta.history[:]
    dat.meta.history.add(name='simulate_csr',
                         parameter='n_samples={}, x_range={}, y_range={}, z_range={}, seed={}'.format(
                             n_samples, x_range, y_range, z_range, seed))

    return dat


def _make_spots(n_samples=100, n_features=2, centers=None, radius=1.0, feature_range=(-10.0, 10.0),
                shuffle = True, seed = None):
    """
    Generate spots with equally distributed points inside. Centers are spatially-distributed by complete spatial
    randomness within the boundarys given by `feature_range`. The number of dimensions is specified by `n_features`.

    Parameters
    ----------
    n_samples : int or array-like
        If int, it is the total number of points equally divided among
        clusters.
        If array-like, each element of the sequence indicates
        the number of samples per cluster.
    n_features : int
        The number of features for each sample.
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.
    radius : tuple of floats or sequence of tuple of floats
        The radius for the spots. If tuple, the number of elements must be equal to n_features. If sequence of tuples,
        the length of sequence must be equal to the number of centers and the length of each tuple must be equal to
        n_features.
    feature_range : pair of floats (min, max) or sequence of pair of floats
        The bounding box for each cluster center when centers are
        generated at random. If sequence the number of elements but be equal to n_features.
    shuffle : boolean, optional (default=True)
        Shuffle the samples.
    seed : int
        random number generation seed

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """
    from collections.abc import Iterable

    if np.issubdtype(n_samples, np.signedinteger):  # or use: if isinstance(n_samples, (int, np.integer)):
        # Set n_centers by looking at centers arg
        if centers is None:
            centers = 3

        if np.issubdtype(centers, np.signedinteger):
            n_centers = centers
            centers = np.random.rand((n_centers, n_features)) * center_box
            # centers = generator.uniform(center_box[0], center_box[1],
            #                             size=(n_centers, n_features))

        else:
            #todo: centers = check_array(centers)
            n_features = centers.shape[1]
            n_centers = centers.shape[0]

    else:
        # Set n_centers by looking at [n_samples] arg
        n_centers = len(n_samples)
        if centers is None:
            centers = np.random.rand((n_centers, n_features)) * center_box
            # centers = generator.uniform(center_box[0], center_box[1],
            #                             size=(n_centers, n_features))
        try:
            assert len(centers) == n_centers
        except TypeError:
            raise ValueError("Parameter `centers` must be array-like. "
                             "Got {!r} instead".format(centers))
        except AssertionError:
            raise ValueError("Length of `n_samples` not consistent"
                             " with number of centers. Got n_samples = {} "
                             "and centers = {}".format(n_samples, centers))
        else:
            pass
            # todo: centers = check_array(centers)
        n_features = centers.shape[1]


    # radius: if radius is given as list, it must be consistent with the n_centers
    if (hasattr(radius, "__len__") and len(radius) != n_centers):
        raise ValueError("Length of `radii` not consistent with "
                         "number of centers. Got centers = {} "
                         "and radii = {}".format(centers, radius))

    if isinstance(radius, (float, np.float)):
        radius = np.full(len(centers), radius)

    X = []
    y = []

    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers

        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
#
#     for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
#         X.append(generator.normal(loc=centers[i], scale=std,
#                                   size=(n, n_features)))
#         y += [i] * n
#
#     X = np.concatenate(X)
#     y = np.array(y)
#
#     if shuffle:
#         total_n_samples = np.sum(n_samples)
#         indices = np.arange(total_n_samples)
#         generator.shuffle(indices)
#         X = X[indices]
#         y = y[indices]
#
#   return X, y


def simulate_spots(n_samples = 100, n_features = 2, centers = None, cluster_std = 1.0, center_box = (-10.0, 10.0),
                   shuffle = True, seed = None):
    """
    Provide a dataset of localizations with coordinates and labels that are spatially-distributed spots (circular or
    elliptic) with equally distributed points inside. The spot centers are distributed by complete spatial randomness.

    Parameters
    ----------
    n_samples : int or array-like
        If int, it is the total number of points equally divided among
        clusters.
        If array-like, each element of the sequence indicates
        the number of samples per cluster.
    n_features : int
        The number of features for each sample.
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.
    radii : tuple of floats or sequence of tuple of floats
        The radii for the spots. If tuple, the number of elements must be equal to n_features. If sequence of tuples,
        the length of sequence must be equal to the number of centers and the length of each tuple must be equal to
        n_features.
    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    shuffle : boolean, optional (default=True)
        Shuffle the samples.
    seed : int
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    if seed is not None:
        np.random.seed(seed)

    points, labels = _make_spots(n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=cluster_std,
                                center_box=center_box, random_state = seed)

    if n_features == 1:
        dataframe = pd.DataFrame(np.stack((points[:, 0], labels), axis=-1),
                                 columns=['Position_x', 'Cluster_label'])
    if n_features == 2:
        dataframe = pd.DataFrame(np.stack((points[:, 0], points[:, 1], labels), axis=-1),
                                 columns=['Position_x', 'Position_y', 'Cluster_label'])
    if n_features == 3:
        dataframe = pd.DataFrame(np.stack((points[:, 0], points[:, 1], points[:, 2], labels), axis=-1),
                                 columns=['Position_x', 'Position_y', 'Position_z', 'Cluster_label'])

    dat = LocData.from_dataframe(dataframe=dataframe)

    # metadata
    dat.meta.source = metadata_pb2.SIMULATION
    del dat.meta.history[:]
    dat.meta.history.add(name='simulate_blobs',
                         parameter='n_centers={}, n_samples={}, n_features={}, center_box={}, cluster_std={}, '
                                   'seed={}'.format(
                             n_centers, n_samples, n_features, center_box, cluster_std, seed))

    return dat


def simulate_blobs(n_centers=100, n_samples=10000, n_features=2, center_box=(0,10000), cluster_std=10, seed=None):
    """
    Provide a dataset of localizations with coordinates and labels that are spatially-distributed on a rectangular
    shape.

    The data consists of  normal distributed point cluster with blob centers being distributed by complete
    spatial randomness. The simulation uses the make_blobs method from sklearn with corresponding parameters.

    Parameters
    ----------
    n_centers : int
        number of blobs

    n_samples : int
        total number of localizations

    n_features : int
        the number of dimensions

    center_box : tuple of two ints
        the range for valid coordinates

    cluster_std : int
        standard deviation for Gauss-distributed positions in each blob

    seed : int (default: None)
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    if seed is not None:
        np.random.seed(seed)

    points, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=cluster_std,
                                center_box=center_box, random_state = seed)

    if n_features == 1:
        dataframe = pd.DataFrame(np.stack((points[:, 0], labels), axis=-1),
                                 columns=['Position_x', 'Cluster_label'])
    if n_features == 2:
        dataframe = pd.DataFrame(np.stack((points[:, 0], points[:, 1], labels), axis=-1),
                                 columns=['Position_x', 'Position_y', 'Cluster_label'])
    if n_features == 3:
        dataframe = pd.DataFrame(np.stack((points[:, 0], points[:, 1], points[:, 2], labels), axis=-1),
                                 columns=['Position_x', 'Position_y', 'Position_z', 'Cluster_label'])

    dat = LocData.from_dataframe(dataframe=dataframe)

    # metadata
    dat.meta.source = metadata_pb2.SIMULATION
    del dat.meta.history[:]
    dat.meta.history.add(name='simulate_blobs',
                         parameter='n_centers={}, n_samples={}, n_features={}, center_box={}, cluster_std={}, '
                                   'seed={}'.format(
                             n_centers, n_samples, n_features, center_box, cluster_std, seed))

    return dat


def _random_walk(number_walks=1, number_steps=10, dimensions=2, diffusion_constant=1, time_step=10):
    '''
    Random walk simulation

    Parameters
    ----------
    number_walks: int
        Number of walks
    number_steps : int
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
        (times, positions), where shape(times) is 1 and shape of positions is (number_walks, number_steps, dimensions)
    '''

    # equally spaced time steps
    times = np.arange(number_steps) * time_step

    # random step sizes according to the diffusion constant
    random_numbers = np.random.randint(0, 2, size=(number_walks, number_steps, dimensions))
    step_size = np.sqrt(2 * dimensions * diffusion_constant * time_step)
    steps = np.where(random_numbers == 0, -step_size, +step_size)

    # walker positions
    positions = np.cumsum(steps, axis=1)

    return times, positions


def simulate_tracks(number_walks=1, number_steps=10, ranges = ((0,10000),(0,10000)), diffusion_constant=1,
                    time_step=10, seed=None):
    """
    Provide a dataset of localizations representing random walks with starting points being spatially-distributed
    on a rectangular shape or cubic volume by complete spatial randomness.

    Parameters
    ----------
    number_walks: int
        Number of walks
    number_steps : int
        Number of time steps (i.e. frames)
    ranges : tuple of tuples of two ints
        the ranges for valid x[, y, z]-coordinates
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

    start_positions = np.array([np.random.uniform(*_range, size=number_walks) for _range in ranges]).T

    times, positions = _random_walk(number_walks=number_walks, number_steps=number_steps, dimensions=len(ranges),
                                    diffusion_constant=diffusion_constant, time_step=time_step)

    new_positions = np.concatenate([
        start_position + position
        for start_position, position in zip(start_positions, positions)
    ])

    locdata_dict = {
        'Position_' + label: position_values
        for _, position_values, label in zip(ranges, new_positions.T, ('x', 'y', 'z'))
    }

    locdata_dict.update(Frame=np.tile(range(len(times)), (number_walks)))

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(locdata_dict))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return locdata



def resample(locdata, number_samples=10):
    """
    Resample locdata according to localization uncertainty. Per localization *number_samples* new localizations
    are simulated normally distributed around the localization coordinates with a standard deviation set to the
    uncertainty in each dimension.
    The resulting LocData object carries new localizations with the following properties: 'Position_x',
    'Position_y'[, 'Position_z'], 'Origin_index'

    Parameters
    ----------
    locdata : LocData object
        Localization data to be resampled
    number_samples : int
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
        new_d.update({'Origin_index': np.full(number_samples, i)})
        x_values = np.random.normal(loc=locdata.data.iloc[i]['Position_x'],
                                    scale=locdata.data.iloc[i]['Uncertainty_x'],
                                    size=number_samples
                                    )
        new_d.update({'Position_x': x_values})

        y_values = np.random.normal(loc=locdata.data.iloc[i]['Position_y'],
                                    scale=locdata.data.iloc[i]['Uncertainty_y'],
                                    size=number_samples
                                    )
        new_d.update({'Position_y': y_values})

        try:
            z_values = np.random.normal(loc=locdata.data.iloc[i]['Position_z'],
                                        scale=locdata.data.iloc[i]['Uncertainty_z'],
                                        size=number_samples
                                        )
            new_d.update({'Position_z': z_values})
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
                      parameter='locdata={}, number_samples={}'.format(locdata, number_samples))

    # instantiate
    new_locdata = LocData.from_dataframe(dataframe=dataframe, meta=meta_)

    return new_locdata