'''

This module provides functions to simulate localization data and return LocData objects.

'''

import sys
import time

import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

from surepy import LocData
from surepy.data import metadata_pb2

#todo: correct history updates

def simulate(**kwargs):
    """
    Provide data by running a simulation.

    Parameters
    ----------
    kwargs :
        Simulation parameter

    Returns
    -------
    LocData
        A new instance of LocData with simulated localization data.
    """
    raise NotImplementedError


def simulate_csr(n_samples = 10000, x_range = (0,10000), y_range = (0,10000), z_range = None, seed=None):
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