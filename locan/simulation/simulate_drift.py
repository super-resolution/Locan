"""
Simulate drift and apply it to localization data.

Drift can be linear in time or resemble a random walk.
"""
from __future__ import annotations

import sys

import numpy as np
import numpy.typing as npt
import pandas as pd

from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta
from locan.locan_types import RandomGeneratorSeed

__all__: list[str] = ["add_drift"]


def _random_walk_drift(
    n_steps: int,
    diffusion_constant: npt.ArrayLike,
    velocity: npt.ArrayLike,
    seed: RandomGeneratorSeed = None,
) -> npt.NDArray[np.float_]:
    """
    Transform locdata coordinates according to a simulated drift.

    Position deltas are computed as function of frame number.
    Within a single time unit delta_t the probability for a diffusion step of
    length delta_x is:

    p(delta_x, delta_t) = Norm(-velocity*delta_t, sigma**2) where the standard
    deviation sigma**2 = 2 * D * delta_t

    Parameters
    ----------
    n_steps
        The number of time steps for the random walk.
    diffusion_constant
        Diffusion constant for each dimension specifying the drift velocity
        with shape (point_dimension,).
        The diffusion constant has the unit of square of localization
        coordinate unit per frame unit.
    velocity
        Drift velocity in units of localization coordinate unit per frame unit
        with shape (point_dimension,)
    seed
        random number generation seed

    Returns
    -------
    npt.NDArray[np.float_]
        Position deltas with shape (n_points, diffusion_constant.shape)
        that have to be added to the original localizations.
    """
    rng = np.random.default_rng(seed)
    diffusion_constant = np.asarray(diffusion_constant)
    velocity = np.asarray(velocity)

    sigmas = np.sqrt(2 * diffusion_constant)  # * delta_t which is 1 frame
    steps = np.array(
        [
            rng.normal(loc=-vel, scale=sigma, size=n_steps)
            for vel, sigma in zip(velocity, sigmas)
        ]
    )
    return np.cumsum(steps, axis=1)


def _drift(
    frames: npt.ArrayLike,
    diffusion_constant: tuple[float, ...] | None = None,
    velocity: tuple[float, ...] | None = None,
    seed: RandomGeneratorSeed = None,
) -> npt.NDArray[np.float_]:
    """
    Compute position deltas as function of frame number.

     Parameters
    ----------
    frames
        array with frame numbers
    diffusion_constant
        Diffusion constant for each dimension specifying the drift velocity
        with shape (point_dimension,).
        The diffusion constant has the unit of square of localization
        coordinate unit per frame unit.
        If None only linear drift is computed.
    velocity
        Drift velocity in units of localization coordinate unit per frame
        unit with shape (point_dimension,)
    seed
        random number generation seed

    Returns
    -------
    npt.NDArray[np.float_]
        Position deltas with shape (n_points, diffusion_constant.shape)
        that have to be added to the original localizations.
    """
    frames_ = np.asarray(frames)

    if diffusion_constant is None and velocity is not None:  # only linear drift
        position_deltas = frames_[:, np.newaxis] * velocity
        position_deltas = position_deltas.T

    elif diffusion_constant is not None:  # random walk plus linear drift
        velocity_ = np.zeros(len(diffusion_constant)) if velocity is None else velocity
        cumsteps = _random_walk_drift(
            frames_.max() + 1, diffusion_constant, velocity_, seed  # type: ignore
        )
        position_deltas = cumsteps[:, frames_]

    else:  # no drift
        position_deltas = None

    return position_deltas


def add_drift(
    locdata: LocData,
    diffusion_constant: tuple[float, ...] | None = None,
    velocity: tuple[float, ...] | None = None,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Compute position deltas as function of frame number.

    Parameters
    ----------
    locdata
        Original localization data
    diffusion_constant
        Diffusion constant for each dimension specifying the drift velocity
        with shape (point_dimension,).
        The diffusion constant has the unit of square of
        localization coordinate unit per frame unit.
    velocity
        Drift velocity in units of localization coordinate unit per frame
        unit with shape (point_dimension,)
    seed
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
    frames = locdata.data["frame"]

    if diffusion_constant is None and velocity is None:
        transformed_points = points
    else:
        transformed_points = (
            points + _drift(frames, diffusion_constant, velocity, seed).T
        )

    # new LocData object
    new_dataframe = locdata.data.copy()
    new_dataframe.update(
        pd.DataFrame(
            transformed_points,
            columns=locdata.coordinate_keys,
            index=locdata.data.index,
        )
    )
    new_locdata = LocData.from_dataframe(new_dataframe)

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata
