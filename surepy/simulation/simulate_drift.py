"""
Simulate drift and apply it to localization data.

Drift can be linear in time or resemble a random walk.
"""
import sys

import numpy as np
import pandas as pd

from surepy.data.locdata import LocData
from surepy.data.metadata_utils import _modify_meta


__all__ = ['add_drift']


def _random_walk_drift(n_steps, diffusion_constant, velocity, seed=None):
    """
    Transform locdata coordinates according to a simulated drift.

    Position deltas are computed as function of frame number.
    Within a single time unit delta_t the probability for a diffusion step of length delta_x is:

    p(delta_x, delta_t) = Norm(-velocity*delta_t, sigma**2) where the standard deviation sigma**2 = 2 * D * delta_t

    Parameters
    ----------
    n_steps : int
        The number of time steps for the random walk.
    diffusion_constant : tuple of float with shape (point_dimension,)
        Diffusion constant for each dimension specifying the drift velocity.
        The diffusion constant has the unit of square of localization coordinate unit per frame unit.
    velocity : tuple of float with shape (point_dimension,)
        Drift velocity in units of localization coordinate unit per frame unit
    seed : int
        random number generation seed

    Returns
    -------
    numpy.ndarray with shape (n_points, diffusion_constant.shape)
        Position deltas that have to be added to the original localizations.
    """
    if seed is not None:
        np.random.seed(seed)
    sigmas = np.sqrt(2 * np.asarray(diffusion_constant))  # * delta_t which is 1 frame
    steps = np.array([np.random.normal(loc=-vel, scale=sigma, size=n_steps)
                      for vel, sigma in zip(velocity, sigmas)])
    return np.cumsum(steps, axis=1)


def _drift(frames, diffusion_constant=None, velocity=None, seed=None):
    """
    Compute position deltas as function of frame number.

     Parameters
    ----------
    frames : array-like of int
        array with frame numbers
    diffusion_constant : tuple of float with shape (point_dimension,), None
        Diffusion constant for each dimension specifying the drift velocity.
        The diffusion constant has the unit of square of localization coordinate unit per frame unit.
        If None only linear drift is computed.
    velocity : tuple of float with shape (point_dimension,)
        Drift velocity in units of localization coordinate unit per frame unit
    seed : int
        random number generation seed

    Returns
    -------
    numpy.ndarray with shape (n_points, diffusion_constant.shape)
        Position deltas that have to be added to the original localizations.
    """
    frames_ = np.asarray(frames)

    if diffusion_constant is None and velocity is not None:  # only linear drift
        position_deltas = frames_[:, np.newaxis] * velocity
        position_deltas = position_deltas.T

    elif diffusion_constant is not None:  # random walk plus linear drift
        velocity_ = (np.zeros(len(diffusion_constant)) if velocity is None else velocity)
        cumsteps = _random_walk_drift(frames_.max() + 1, diffusion_constant, velocity_, seed)
        position_deltas = cumsteps[:, frames_]

    else:  # no drift
        position_deltas = None

    return position_deltas


def add_drift(locdata, diffusion_constant=None, velocity=None, seed=None):
    """
    Compute position deltas as function of frame number.

    Parameters
    ----------
    locdata : LocData object
        Original localization data
    diffusion_constant : tuple of float with shape (point_dimension,), None
        Diffusion constant for each dimension specifying the drift velocity.
        The diffusion constant has the unit of square of localization coordinate unit per frame unit.
    velocity : tuple of float with shape (point_dimension,)
        Drift velocity in units of localization coordinate unit per frame unit
    seed : int
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    local_parameter = locals()

    if len(locdata) == 0:
        return locdata

    points = locdata.coordinates
    frames = locdata.data['frame']

    if diffusion_constant is None and velocity is None:
        transformed_points = points
    else:
        transformed_points = points + _drift(frames, diffusion_constant, velocity, seed).T

    # new LocData object
    new_dataframe = locdata.data.copy()
    new_dataframe.update(pd.DataFrame(transformed_points, columns=locdata.coordinate_labels,
                                      index=locdata.data.index))
    new_locdata = LocData.from_dataframe(new_dataframe)

    # update metadata
    meta_ = _modify_meta(locdata, new_locdata, function_name=sys._getframe().f_code.co_name,
                         parameter=local_parameter,
                         meta=None)
    new_locdata.meta = meta_

    return new_locdata
