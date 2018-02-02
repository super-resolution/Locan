'''

Methods for simulating localization data.

'''

import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from surepy import LocData


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


def simulate_csr(n_samples = 10000, x_range = (0,10000), y_range = (0,10000), z_range = None, seed=0):
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
            dict.update({'Position_' + i: None if j is None else np.random.uniform(*j, n_samples)})

    dat = LocData.from_dataframe(dataframe=pd.DataFrame(dict))

    dat.meta['Source'] = 'simulation'
    dat.meta['State'] = 'raw'
    dat.meta['History'] = [].append({'Method': 'simulate_csr',
                                     'Parameter': [n_samples, x_range, y_range, z_range, seed]})

    return dat


def simulate_blobs(n_centers=100, n_samples=10000, n_features=2, center_box=(0,10000), cluster_std=10, seed=None):
    """
    Provide a dataset of localizations with coordinates and labels that are spatially-distributed on a rectangular
    shape.

    The data consists of  normal distributed point cluster with blob centers being distributed by complete
    spatial randomness. Teh simulation uses the make_blobs method from sklearn with corresponding parameters.

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

    dat.meta['Source'] = 'simulation'
    dat.meta['State'] = 'raw'
    dat.meta['History'].append({'Method:': 'simulate_blobs',
                                'Parameter': [n_centers, n_samples, n_features, n_centers, cluster_std, seed]
                                })

    return dat



