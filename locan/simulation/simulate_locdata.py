"""

Simulate localization data.

This module provides functions to simulate localization data and return
LocData objects.
Localizations are often distributed either by a spatial process of
complete-spatial randomness or following a Neyman-Scott process [1]_.
For a Neyman-Scott process parent events (representing single emitters) yield
a random number of cluster_mu events (representing localizations due to
repeated blinking). Related spatial point processes include Matérn and Thomas
processes.

Functions that are named as make_* provide point data arrays.
Functions that are named as simulate_* provide locdata.


Parts of this code is adapted from
scikit-learn/sklearn/datasets/_samples_generator.py .
(BSD 3-Clause License, Copyright (c) 2007-2020 The scikit-learn developers.)

References
----------
.. [1] Neyman, J. & Scott, E. L.,
   A Theory of the Spatial Distribution of Galaxies.
   Astrophysical Journal 1952, vol. 116, p.144.

"""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.data.locdata_utils import (
    _bump_property_key,
    _get_loc_property_key_per_dimension,
)
from locan.data.metadata_utils import _modify_meta
from locan.data.region import (
    AxisOrientedCuboid,
    AxisOrientedHypercuboid,
    Ellipse,
    EmptyRegion,
    Interval,
    Rectangle,
    Region,
)
from locan.data.region_utils import expand_region
from locan.locan_types import RandomGeneratorSeed

__all__: list[str] = [
    "make_uniform",
    "make_Poisson",
    "make_cluster",
    "make_NeymanScott",
    "make_Matern",
    "make_Thomas",
    "make_dstorm",
    "simulate_uniform",
    "simulate_Poisson",
    "simulate_cluster",
    "simulate_NeymanScott",
    "simulate_Matern",
    "simulate_Thomas",
    "simulate_dstorm",
    "simulate_tracks",
    "resample",
    "simulate_frame_numbers",
    "randomize",
]

logger = logging.getLogger(__name__)


def make_uniform(
    n_samples: int,
    region: Region | npt.ArrayLike = (0, 1),
    seed: RandomGeneratorSeed = None,
) -> npt.NDArray[np.float_]:
    """
    Provide points that are distributed by a uniform
    (complete spatial randomness) point process
    within the boundaries given by `region`.

    Parameters
    ----------
    n_samples
        The total number of localizations of the point process
    region
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    seed
        Random number generation seed

    Returns
    -------
    npt.NDArray[np.float_]
        The generated samples of shape (n_samples, n_features).
    """
    rng = np.random.default_rng(seed)

    if not isinstance(region, Region):
        region = Region.from_intervals(region)

    if isinstance(region, EmptyRegion):
        samples = np.array([])
    elif isinstance(
        region, (Interval, AxisOrientedCuboid, AxisOrientedHypercuboid)
    ) or (isinstance(region, Rectangle) and region.angle == 0):
        samples = rng.uniform(
            region.bounds[: region.dimension],
            region.bounds[region.dimension :],
            size=(n_samples, region.dimension),
        )
    elif isinstance(region, Ellipse) and region.width == region.height:
        radius = region.width / 2
        # angular and radial coordinates of Poisson points
        theta = rng.random(n_samples) * 2 * np.pi
        rho = radius * np.sqrt(rng.random(n_samples))
        # Convert from polar to Cartesian coordinates
        xx = rho * np.cos(theta)
        yy = rho * np.sin(theta)
        samples = np.array((xx, yy)).T
    else:
        sampling_ratio = region.region_measure / region.bounding_box.region_measure
        n_samples_updated = int(n_samples / sampling_ratio * 2)

        samples_ = []
        n_remaining = n_samples
        while n_remaining > 0:
            new_samples = rng.random(size=(n_samples_updated, region.dimension))  # type: ignore
            new_samples = region.extent * new_samples + region.bounding_box.corner  # type: ignore
            new_samples = new_samples[region.contains(new_samples)]
            samples_.append(new_samples)
            n_remaining = n_remaining - len(new_samples)

        samples = np.concatenate(samples_)
        samples = samples[0:n_samples]

    return samples


def simulate_uniform(
    n_samples: int,
    region: Region | npt.ArrayLike = (0, 1),
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Provide points that are distributed by a uniform Poisson point process
    within the boundaries given by `region`.

    Parameters
    ----------
    n_samples
        The total number of localizations of the point process
    region
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    seed
        random number generation seed

    Returns
    -------
    LocData
        The generated samples.
    """
    parameter = locals()
    samples = make_uniform(n_samples=n_samples, region=region, seed=seed)
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)
    assert region_.dimension is not None  # type narrowing # noqa: S101
    locdata = LocData.from_coordinates(coordinates=samples)
    locdata.dimension = region_.dimension
    locdata.region = region_

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=make_uniform.__name__, parameter=str(parameter))

    return locdata


def make_Poisson(
    intensity: int | float,
    region: Region | npt.ArrayLike = (0, 1),
    seed: RandomGeneratorSeed = None,
) -> npt.NDArray[np.float_]:
    """
    Provide points that are distributed by a uniform Poisson point process
    within the boundaries given by `region`.

    Parameters
    ----------
    intensity
        The intensity (points per unit region measure) of the point process
    region
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    seed
        random number generation seed

    Returns
    -------
    npt.NDArray[np.float_]
        The generated samples of shape (n_samples, n_features).
    """
    rng = np.random.default_rng(seed)

    if not isinstance(region, Region):
        region = Region.from_intervals(region)

    n_samples = rng.poisson(lam=intensity * region.region_measure)

    if isinstance(region, EmptyRegion):
        samples = np.array([])
    elif n_samples == 0:
        samples = np.array([])
        assert region.dimension is not None  # type narrowing # noqa: S101
        for _i in range(region.dimension):
            samples = samples[..., np.newaxis]
    elif isinstance(
        region, (Interval, Rectangle, AxisOrientedCuboid, AxisOrientedHypercuboid)
    ):
        samples = rng.uniform(
            region.bounds[: region.dimension],
            region.bounds[region.dimension :],
            size=(n_samples, region.dimension),
        )
    elif isinstance(region, Ellipse) and region.width == region.height:
        radius = region.width / 2
        # angular and radial coordinates of Poisson points
        theta = rng.random(n_samples) * 2 * np.pi
        rho = radius * np.sqrt(rng.random(n_samples))
        # Convert from polar to Cartesian coordinates
        xx = rho * np.cos(theta)
        yy = rho * np.sin(theta)
        samples = np.array((xx, yy)).T
    else:
        sampling_ratio = region.region_measure / region.bounding_box.region_measure
        n_samples_updated = int(n_samples / sampling_ratio * 2)

        samples_ = []
        n_remaining = n_samples
        while n_remaining > 0:
            assert region.dimension is not None  # type narrowing # noqa: S101
            new_samples = rng.random(size=(n_samples_updated, region.dimension))
            new_samples = region.extent * new_samples + region.bounding_box.corner  # type: ignore
            new_samples = new_samples[region.contains(new_samples)]
            samples_.append(new_samples)
            n_remaining = n_remaining - len(new_samples)

        samples = np.concatenate(samples_)
        samples = samples[0:n_samples]

    return samples


def simulate_Poisson(
    intensity: int | float,
    region: Region | npt.ArrayLike = (0, 1),
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Provide points that are distributed by a uniform Poisson point process
    within the boundaries given by `region`.

    Parameters
    ----------
    intensity
        The intensity (points per unit region measure) of the point process
    region
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    seed
        random number generation seed

    Returns
    -------
    LocData
        The generated samples.
    """
    parameter = locals()
    samples = make_Poisson(intensity=intensity, region=region, seed=seed)
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)
    assert region_.dimension is not None  # type narrowing # noqa: S101
    locdata = LocData.from_coordinates(coordinates=samples)
    locdata.dimension = region_.dimension
    locdata.region = region_

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=make_Poisson.__name__, parameter=str(parameter))

    return locdata


def make_cluster(
    centers: int | npt.ArrayLike = 3,
    region: Region | npt.ArrayLike = (0, 1.0),
    expansion_distance: float = 0,
    offspring: npt.ArrayLike | Callable[..., Any] | None = None,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region
]:
    """
    Parent positions are taken from `centers`
    or are distributed according to a homogeneous Poisson process with
    exactly `centers`
    within the boundaries given by `region` expanded by the expansion_distance.
    Each parent position is then replaced by cluster_mu offspring points as
    passed or generated by a given function.
    Offspring from parent events that are located outside the region are
    included.

    Parameters
    ----------
    centers
        The number of parents or coordinates for parent events,
        where each parent represents a cluster center.
    region
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_distance
        The distance by which region is expanded on all boundaries.
    offspring
        Points or function for point process to provide cluster.
        Callable must take single parent point as parameter and return an iterable.
        If array-like it must have the same length as parent events.
    clip
        If True the result will be clipped to 'region'.
        If False the extended region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region]
       The generated samples, labels, parent_samples
       of shape (n_samples, n_features) and region
    """
    rng = np.random.default_rng(seed)

    if isinstance(region, EmptyRegion):
        samples, labels, parent_samples, region = (
            np.array([]),
            np.array([]),
            np.array([]),
            EmptyRegion(),
        )
        return samples, labels, parent_samples, region
    elif not isinstance(region, Region):
        region = Region.from_intervals(region)

    expanded_region = expand_region(region, expansion_distance)

    if isinstance(centers, (int, np.integer)):
        n_centers = int(centers)
        parent_samples = make_uniform(
            n_samples=n_centers, region=expanded_region, seed=rng
        )
    else:  # if centers is array
        parent_samples = np.array(centers)
        centers_shape = np.shape(parent_samples)
        centers_dimension = 1 if len(centers_shape) == 1 else centers_shape[1]
        if region.dimension != centers_dimension:
            raise ValueError(
                f"Region dimensions must be the same as the dimensions for each center. "
                f"Got region dimension: {region.dimension} and "
                f"center dimensions: {centers_dimension} instead."
            )
        n_centers = centers_shape[0]

    # replace parents by offspring samples
    if offspring is None:
        samples = parent_samples
        labels = np.arange(0, len(parent_samples))

    elif callable(offspring):
        try:
            offspring_samples = offspring(parent_samples)
            labels_ = [[i] * len(os) for i, os in enumerate(offspring_samples)]
        except TypeError:
            offspring_samples = []
            labels_ = []
            for i, parent in enumerate(parent_samples):
                offspring_samples_ = offspring(parent)
                offspring_samples.append(offspring_samples_)
                labels_.append([i] * len(offspring_samples_))
        samples = np.array(list(chain(*offspring_samples)))
        labels = np.array(list(chain(*labels_)))

    elif len(offspring) >= len(parent_samples):  # type: ignore
        offspring_samples = []
        labels_ = []
        for i, (os, parent) in enumerate(zip(offspring[:n_centers], parent_samples)):  # type: ignore
            if len(os) > 0:
                offspring_samples_ = np.asarray(os) + parent
                offspring_samples.append(offspring_samples_)
                labels_.append([i] * len(offspring_samples_))
        samples = np.array(list(chain(*offspring_samples)))
        labels = np.array(list(chain(*labels_)))

    else:
        raise TypeError(
            f"offspring must be callable or array-like with length >= than n_centers {n_centers}."
        )

    if (
        samples.ndim == 1
    ):  # this is to convert 1-dimensional arrays into arrays with shape (n_samples, 1).
        samples.shape = (len(samples), 1)

    if clip is True:
        if len(samples) != 0:
            inside_indices = region.contains(samples)
            samples = samples[inside_indices]
            labels = labels[inside_indices]
        region_ = region
    else:
        region_ = expanded_region

    if shuffle:
        shuffled_indices = rng.permutation(len(samples))
        samples = samples[shuffled_indices]
        labels = labels[shuffled_indices]

    if (
        len(samples) == 0
    ):  # this is to convert empty arrays into arrays with shape (n_samples, n_features).
        samples = np.array([])
        samples = samples[:, np.newaxis]

    return samples, labels, parent_samples, region_


def simulate_cluster(
    centers: int | npt.ArrayLike = 3,
    region: Region | npt.ArrayLike = (0, 1.0),
    expansion_distance: float = 0,
    offspring: npt.ArrayLike | Callable[..., Any] | None = None,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Generate clustered point data.
    Parent positions are taken from `centers`
    or are distributed according to a homogeneous Poisson process with exactly `centers`
    within the boundaries given by `region` expanded by the expansion_distance.
    Each parent position is then replaced by offspring points as passed or generated by a given function.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    centers
        The number of parents or coordinates for parent events,
        where each parent represents a cluster center.
    region
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_distance
        The distance by which region is expanded on all boundaries.
    offspring
        Points or function for point process to provide cluster.
        Callable must take single parent point as parameter and return an iterable.
        If array-like it must have the same length as parent events.
    clip
        If True the result will be clipped to 'region'.
        If False the extended region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    LocData
        The generated samples.
    """
    parameter = locals()
    samples, labels, _, region = make_cluster(
        centers, region, expansion_distance, offspring, clip, shuffle, seed
    )
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)  # type: ignore
    assert region_.dimension is not None  # type narrowing # noqa: S101
    locdata = LocData.from_coordinates(coordinates=samples)
    locdata.dimension = region_.dimension
    locdata.region = region_
    locdata.dataframe = locdata.dataframe.assign(cluster_label=labels)

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=make_cluster.__name__, parameter=str(parameter))

    return locdata


def make_NeymanScott(
    parent_intensity: int | float = 100,
    region: Region | npt.ArrayLike = (0, 1.0),
    expansion_distance: float = 0,
    offspring: npt.ArrayLike | Callable[..., Any] | None = None,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region
]:
    """
    Generate clustered point data following a Neyman-Scott random point
    process.
    Parent positions are distributed according to a homogeneous Poisson
    process with `parent_intensity`
    within the boundaries given by `region` expanded by the expansion_distance.
    Each parent position is then replaced by offspring points as passed or
    generated by a given function.
    Offspring from parent events that are located outside the region are
    included.

    Parameters
    ----------
    parent_intensity
        The intensity (points per unit region measure) of the Poisson point
        process for parent events.
    region
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_distance
        The distance by which region is expanded on all boundaries.
    offspring
        Points or function for point process to provide offspring points.
        Callable must take single parent point as parameter.
        If array-like it must have enough elements to fit the randomly
        generated number of parent events.
    clip
        If True the result will be clipped to 'region'.
        If False the extended region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region]
       The generated samples, labels, parent_samples of shape
       (n_samples, n_features) and region
    """
    rng = np.random.default_rng(seed)

    if isinstance(region, EmptyRegion):
        samples, labels, parent_samples, region = (
            np.array([]),
            np.array([]),
            np.array([]),
            EmptyRegion(),
        )
        return samples, labels, parent_samples, region
    elif not isinstance(region, Region):
        region = Region.from_intervals(region)

    expanded_region = expand_region(region, expansion_distance)

    parent_samples = make_Poisson(
        intensity=parent_intensity, region=expanded_region, seed=rng
    )

    # replace parents by offspring samples
    if offspring is None:
        samples = parent_samples
        labels = np.arange(0, len(parent_samples))

    elif callable(offspring):
        try:
            offspring_samples = offspring(parent_samples)
            labels_ = [[i] * len(os) for i, os in enumerate(offspring_samples)]
        except TypeError:
            offspring_samples = []
            labels_ = []
            for i, parent in enumerate(parent_samples):
                offspring_samples_ = offspring(parent)
                offspring_samples.append(offspring_samples_)
                labels_.append([i] * len(offspring_samples_))
        samples = np.array(list(chain(*offspring_samples)))
        labels = np.array(list(chain(*labels_)))

    elif len(offspring) >= len(parent_samples):  # type: ignore
        offspring_samples = []
        labels_ = []
        if isinstance(offspring, np.ndarray):
            offspring_samples = (
                np.asarray(offspring[: len(parent_samples)]) + parent_samples
            )
            labels_ = [[i] * len(os) for i, os in enumerate(offspring_samples)]
        else:
            for i, (os, parent) in enumerate(
                zip(offspring[: len(parent_samples)], parent_samples)  # type: ignore
            ):
                if len(os) > 0:
                    offspring_samples_ = np.asarray(os) + parent
                    offspring_samples.append(offspring_samples_)
                    labels_.append([i] * len(offspring_samples_))
        samples = np.array(list(chain(*offspring_samples)))
        labels = np.array(list(chain(*labels_)))

    else:
        raise TypeError(
            f"offspring must be callable or array-like with "
            f"length >= n_centers {len(parent_samples)}."
        )

    if (
        samples.ndim == 1
    ):  # this is to convert 1-dimensional arrays into arrays with shape (n_samples, 1).
        samples.shape = (len(samples), 1)

    if clip is True:
        if len(samples) != 0:
            inside_indices = region.contains(samples)
            samples = samples[inside_indices]
            labels = labels[inside_indices]
        region_ = region
    else:
        region_ = expanded_region

    if shuffle:
        shuffled_indices = rng.permutation(len(samples))
        samples = samples[shuffled_indices]
        labels = labels[shuffled_indices]

    if (
        len(samples) == 0
    ):  # this is to convert empty arrays into arrays with shape (n_samples, n_features).
        samples = np.array([])
        samples = samples[:, np.newaxis]

    return samples, labels, parent_samples, region_


def simulate_NeymanScott(
    parent_intensity: int | float = 100,
    region: Region | npt.ArrayLike = (0, 1.0),
    expansion_distance: float = 0,
    offspring: npt.ArrayLike | Callable[..., Any] | None = None,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Generate clustered point data following a Neyman-Scott random point
    process.
    Parent positions are distributed according to a homogeneous Poisson
    process with `parent_intensity`
    within the boundaries given by `region` expanded by the expansion_distance.
    Each parent position is then replaced by offspring points as passed or
    generated by a given function.
    Offspring from parent events that are located outside the region are
    included.

    Parameters
    ----------
    parent_intensity
        The intensity (points per unit region measure) of the Poisson point
        process for parent events.
    region
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_distance
        The distance by which region is expanded on all boundaries.
    offspring
        Points or function for point process to provide offspring points.
        Callable must take single parent point as parameter.
        If array-like it must have enough elements to fit the randomly
        generated number of parent events.
    clip
        If True the result will be clipped to 'region'.
        If False the extended region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    LocData
        The generated samples.
    """
    parameter = locals()
    samples, labels, _, region = make_NeymanScott(
        parent_intensity, region, expansion_distance, offspring, clip, shuffle, seed
    )
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)  # type: ignore
    assert region_.dimension is not None  # type narrowing # noqa: S101
    locdata = LocData.from_coordinates(coordinates=samples)
    locdata.dimension = region_.dimension
    locdata.region = region_
    locdata.dataframe = locdata.dataframe.assign(cluster_label=labels)

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=make_NeymanScott.__name__, parameter=str(parameter))

    return locdata


def make_Matern(
    parent_intensity: int | float = 1,
    region: Region | npt.ArrayLike = (0, 1.0),
    cluster_mu: int | float = 1,
    radius: float | Sequence[float] = 1.0,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region
]:
    """
    Generate clustered point data following a Matern cluster random point
    process.
    Parent positions are distributed according to a homogeneous Poisson
    process with `parent_intensity`
    within the boundaries given by `region` expanded by the maximum radius.
    Each parent position is then replaced by spots of size `radius` with
    Poisson distributed points inside.
    Offspring from parent events that are located outside the region are
    included.

    Parameters
    ----------
    parent_intensity
        The intensity (points per unit region measure) of the Poisson point
        process for parent events.
    region
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    cluster_mu
        The mean number of points of the Poisson point process for
        cluster(cluster_mu) events.
    radius
        The radius for the spots. If tuple, the number of elements must be
        larger than the expected number of parents.
    clip
        If True the result will be clipped to 'region'. If False the extended
        region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region]
       The generated samples, labels, parent_samples
       of shape (n_samples, n_features) and region
    """
    rng = np.random.default_rng(seed)

    if isinstance(region, EmptyRegion):
        samples, labels, parent_samples, region = (
            np.array([]),
            np.array([]),
            np.array([]),
            EmptyRegion(),
        )
        return samples, labels, parent_samples, region
    elif not isinstance(region, Region):
        region = Region.from_intervals(region)

    expansion_distance = np.max(radius)
    expanded_region = expand_region(region, expansion_distance)

    parent_samples = make_Poisson(
        intensity=parent_intensity, region=expanded_region, seed=rng
    )

    # radius: if radius is given as list, it must be consistent with the n_parent_samples
    if hasattr(radius, "__len__"):
        if len(radius) < len(parent_samples):
            raise ValueError(
                f"Length of `radius` {len(radius)} is less than "
                f"the generated n_parent_samples {len(parent_samples)}."
            )
        else:
            radii = radius[: len(parent_samples)]  # type: ignore
    else:  # if isinstance(radius, float):
        radii = np.full(len(parent_samples), radius)

    # replace parents by offspring samples
    samples_ = []
    labels_ = []
    for i, (parent, radius_) in enumerate(zip(parent_samples, radii)):
        if region.dimension == 1:
            offspring_region: Region = Interval(-radius_, radius_)
            offspring_intensity = cluster_mu / offspring_region.region_measure
            offspring_samples = make_Poisson(
                intensity=offspring_intensity, region=offspring_region, seed=rng
            )
        elif region.dimension == 2:
            offspring_region = Ellipse((0, 0), 2 * radius_, 2 * radius_, 0)
            offspring_intensity = cluster_mu / offspring_region.region_measure
            offspring_samples = make_Poisson(
                intensity=offspring_intensity, region=offspring_region, seed=rng
            )
        elif region.dimension == 3:
            raise NotImplementedError
        else:
            raise ValueError("region dimension must be 1, 2, or 3.")
        if len(offspring_samples) != 0:
            offspring_samples = offspring_samples + parent
            samples_.append(offspring_samples)
        labels_ += [i] * len(offspring_samples)

    samples = np.array(list(chain(*samples_)))
    labels = np.array(labels_)

    if clip is True:
        if len(samples) != 0:
            inside_indices = region.contains(samples)
            samples = samples[inside_indices]
            labels = labels[inside_indices]
        region_ = region
    else:
        region_ = expanded_region

    if shuffle:
        shuffled_indices = rng.permutation(len(samples))
        samples = samples[shuffled_indices]
        labels = labels[shuffled_indices]

    if (
        len(samples) == 0
    ):  # this is to convert empty arrays into arrays with shape (n_samples, n_features).
        samples = np.array([])
        samples = samples[:, np.newaxis]

    return samples, labels, parent_samples, region_


def simulate_Matern(
    parent_intensity: int | float = 1,
    region: Region | npt.ArrayLike = (0, 1.0),
    cluster_mu: int | float = 1,
    radius: float | Sequence[float] = 1.0,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Generate clustered point data following a Matern cluster random point
    process.
    Parent positions are distributed according to a homogeneous Poisson
    process with `parent_intensity`
    within the boundaries given by `region` expanded by the maximum radius.
    Each parent position is then replaced by spots of size `radius` with
    Poisson distributed points inside.
    Offspring from parent events that are located outside the region are
    included.

    Parameters
    ----------
    parent_intensity
        The intensity (points per unit region measure) of the Poisson point
        process for parent events.
    region
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    cluster_mu
        The mean number of points of the Poisson point process for
        cluster(cluster_mu) events.
    radius
        The radius for the spots. If tuple, the number of elements must be
        larger than the expected number of parents.
    clip
        If True the result will be clipped to 'region'.
        If False the extended region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    LocData
        The generated samples.
    """
    parameter = locals()
    samples, labels, _, region = make_Matern(
        parent_intensity, region, cluster_mu, radius, clip, shuffle, seed
    )
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)  # type: ignore
    assert region_.dimension is not None  # type narrowing # noqa: S101
    locdata = LocData.from_coordinates(coordinates=samples)
    locdata.dimension = region_.dimension
    locdata.region = region_
    locdata.dataframe = locdata.dataframe.assign(cluster_label=labels)

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=make_Matern.__name__, parameter=str(parameter))

    return locdata


def make_Thomas(
    parent_intensity: int | float = 1,
    region: Region | npt.ArrayLike = (0, 1.0),
    expansion_factor: int | float = 6,
    cluster_mu: int | float | Sequence[float] = 1,
    cluster_std: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region
]:
    """
    Generate clustered point data following a Thomas random point process.
    Parent positions are distributed according to a homogeneous Poisson
    process with `parent_intensity`
    within the boundaries given by `region` expanded by an expansion distance
    that equals
    expansion_factor * max(cluster_std).
    Each parent position is then replaced by n offspring points
    where n is Poisson-distributed with mean number `cluster_mu`
    and point coordinates are normal-distributed around the parent point with
    standard deviation `cluster_std`.
    Offspring from parent events that are located outside the region are
    included.

    Parameters
    ----------
    parent_intensity
        The intensity (points per unit region measure) of the Poisson point
        process for parent events.
    region
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_factor
        Factor by which the cluster_std is multiplied to set the region
        expansion distance.
    cluster_mu
        The mean number of points for normal-distributed offspring points.
    cluster_std
        The standard deviation for normal-distributed offspring points.
    clip
        If True the result will be clipped to 'region'.
        If False the extended region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region]
       The generated samples, labels, parent_samples
       of shape (n_samples, n_features) and region
    """
    rng = np.random.default_rng(seed)

    if not isinstance(region, Region):
        region = Region.from_intervals(region)

    if (
        parent_intensity == 0
        or (np.size(cluster_mu) == 1 and cluster_mu == 0)
        or isinstance(region, EmptyRegion)
    ):
        samples, labels, parent_samples, region = (
            np.array([]),
            np.array([]),
            np.array([]),
            region,
        )
        if region.dimension and region.dimension > 0:
            samples = samples[:, np.newaxis]
        return samples, labels, parent_samples, region

    # expand region
    expansion_distance = expansion_factor * np.max(cluster_std)
    expanded_region = expand_region(region, expansion_distance)

    parent_samples = make_Poisson(
        intensity=parent_intensity, region=expanded_region, seed=rng
    )
    n_cluster = len(parent_samples)

    # check cluster_std consistent with n_centers or n_features
    if len(np.shape(cluster_std)) == 0:
        cluster_std_ = np.full(
            shape=(n_cluster, region.dimension), fill_value=cluster_std  # type: ignore
        )
    elif len(np.shape(cluster_std)) == 1:  # iterate over cluster_std for each feature
        if region.dimension == 1 or len(cluster_std) != region.dimension:  # type: ignore
            raise TypeError(
                f"The shape of cluster_std {np.shape(cluster_std)} is incompatible "
                f"with n_features {region.dimension}."
            )
        else:
            cluster_std_ = np.empty(shape=(n_cluster, region.dimension))
            for i, element in enumerate(cluster_std):  # type: ignore
                cluster_std_[:, i] = np.full((n_cluster,), element)
    elif len(np.shape(cluster_std)) == 2:  # iterate over cluster_std for each center
        if np.shape(cluster_std) < (n_cluster, region.dimension):
            raise TypeError(
                f"The shape of cluster_std {np.shape(cluster_std)} is incompatible with "
                f"n_cluster {n_cluster} or n_features {region.dimension}."
            )
        else:
            cluster_std_ = cluster_std  # type: ignore
    else:
        raise TypeError(
            f"The shape of cluster_std {np.shape(cluster_std)} is incompatible."
        )

    # replace parents by normal-distributed offspring samples
    try:
        n_offspring_list = rng.poisson(lam=cluster_mu[:n_cluster], size=n_cluster)  # type: ignore
    except ValueError as e:
        e.args += (f"Too few offspring events for n_cluster: {n_cluster}",)
        raise
    except (TypeError, IndexError):
        n_offspring_list = rng.poisson(lam=cluster_mu, size=n_cluster)
    samples_ = []
    labels_ = []
    for i, (parent, std, n_offspring) in enumerate(
        zip(parent_samples, cluster_std_, n_offspring_list)
    ):
        offspring_samples = rng.normal(
            loc=parent, scale=std, size=(n_offspring, region.dimension)  # type: ignore
        )
        samples_.append(offspring_samples)
        labels_ += [i] * len(offspring_samples)

    samples = np.concatenate(samples_) if len(samples_) != 0 else np.array([])
    labels = np.array(labels_)

    if clip is True:
        if len(samples) != 0:
            inside_indices = region.contains(samples)
            samples = samples[inside_indices]
            labels = labels[inside_indices]
        region_ = region
    else:
        region_ = expanded_region

    if shuffle:
        shuffled_indices = rng.permutation(len(samples))
        samples = samples[shuffled_indices]
        labels = labels[shuffled_indices]

    if (
        len(samples) == 0
    ):  # this is to convert empty arrays into arrays with shape (n_samples, n_features).
        samples = np.array([])
        samples = samples[:, np.newaxis]

    return samples, labels, parent_samples, region_


def simulate_Thomas(
    parent_intensity: int | float = 1,
    region: Region | npt.ArrayLike = (0, 1.0),
    expansion_factor: int | float = 6,
    cluster_mu: int | float | Sequence[float] = 1,
    cluster_std: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Generate clustered point data following a Thomas random point process.
    Parent positions are distributed according to a homogeneous Poisson
    process with `parent_intensity`
    within the boundaries given by `region` expanded by an expansion distance
    that equals
    expansion_factor * max(cluster_std).
    Each parent position is then replaced by n offspring points
    where n is Poisson-distributed with mean number `cluster_mu`
    and point coordinates are normal-distributed around the parent point with
    standard deviation `cluster_std`.
    Offspring from parent events that are located outside the region are
    included.

    Parameters
    ----------
    parent_intensity
        The intensity (points per unit region measure) of the Poisson point
        process for parent events.
    region
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_factor
        Factor by which the cluster_std is multiplied to set the region
        expansion distance.
    cluster_mu
        The mean number of points for normal-distributed offspring points.
    cluster_std
        The standard deviation for normal-distributed offspring points.
    clip
        If True the result will be clipped to 'region'.
        If False the extended region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    LocData
        The generated samples.
    """
    parameter = locals()
    samples, labels, _, region = make_Thomas(
        parent_intensity,
        region,
        expansion_factor,
        cluster_mu,
        cluster_std,
        clip,
        shuffle,
        seed,
    )
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)  # type: ignore
    assert region_.dimension is not None  # type narrowing # noqa: S101
    locdata = LocData.from_coordinates(coordinates=samples)
    locdata.dimension = region_.dimension
    locdata.region = region_
    locdata.dataframe = locdata.dataframe.assign(cluster_label=labels)

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=make_Thomas.__name__, parameter=str(parameter))

    return locdata


def make_dstorm(
    parent_intensity: int | float = 1,
    region: Region | npt.ArrayLike = (0, 1.0),
    expansion_factor: int | float = 6,
    cluster_mu: int | float | Sequence[float] = 1,
    min_points: int = 0,
    cluster_std: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region
]:
    """
    Generate clustered point data following a Thomas-like random point process.
    Parent positions are distributed according to a homogeneous Poisson
    process with `parent_intensity`
    within the boundaries given by `region` expanded by an expansion distance
    that equals
    expansion_factor * max(cluster_std).
    Each parent position is then replaced by n offspring points
    where n is geometrically-distributed with mean number `cluster_mu`
    and point coordinates are normal-distributed around the parent point with
    standard deviation `cluster_std`.
    Offspring from parent events that are located outside the region are
    included.

    Parameters
    ----------
    parent_intensity
        The intensity (points per unit region measure) of the Poisson point
        process for parent events.
    region
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_factor
        Factor by which the cluster_std is multiplied to set the region
        expansion distance.
    cluster_mu
        The mean number of points for normal-distributed offspring points.
    min_points
        The minimum number of points per cluster.
    cluster_std
        The standard deviation for normal-distributed offspring points.
    clip
        If True the result will be clipped to 'region'. If False the extended
        region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_], Region]
       The generated samples, labels, parent_samples
        of shape (n_samples, n_features) and region
    """
    rng = np.random.default_rng(seed)

    if not isinstance(region, Region):
        region = Region.from_intervals(region)

    if (
        parent_intensity == 0
        or (np.size(cluster_mu) == 1 and cluster_mu == 0)
        or isinstance(region, EmptyRegion)
    ):
        samples, labels, parent_samples, region = (
            np.array([]),
            np.array([]),
            np.array([]),
            region,
        )
        if region.dimension and region.dimension > 0:
            samples = samples[:, np.newaxis]
        return samples, labels, parent_samples, region

    # expand region
    expansion_distance = expansion_factor * np.max(cluster_std)
    expanded_region = expand_region(region, expansion_distance)

    parent_samples = make_Poisson(
        intensity=parent_intensity, region=expanded_region, seed=rng
    )
    n_cluster = len(parent_samples)

    # check cluster_std consistent with n_centers or n_features
    if len(np.shape(cluster_std)) == 0:
        cluster_std_ = np.full(
            shape=(n_cluster, region.dimension), fill_value=cluster_std  # type: ignore
        )
    elif len(np.shape(cluster_std)) == 1:  # iterate over cluster_std for each feature
        if region.dimension == 1 or len(cluster_std) != region.dimension:  # type: ignore
            raise TypeError(
                f"The shape of cluster_std {np.shape(cluster_std)} is incompatible "
                f"with n_features {region.dimension}."
            )
        else:
            cluster_std_ = np.empty(shape=(n_cluster, region.dimension))
            for i, element in enumerate(cluster_std):  # type: ignore
                cluster_std_[:, i] = np.full((n_cluster,), element)
    elif len(np.shape(cluster_std)) == 2:  # iterate over cluster_std for each center
        if np.shape(cluster_std) < (n_cluster, region.dimension):
            raise TypeError(
                f"The shape of cluster_std {np.shape(cluster_std)} is incompatible with "
                f"n_cluster {n_cluster} or n_features {region.dimension}."
            )
        else:
            cluster_std_ = cluster_std  # type: ignore
    else:
        raise TypeError(
            f"The shape of cluster_std {np.shape(cluster_std)} is incompatible."
        )

    # replace parents by normal-distributed offspring samples
    try:
        p_values = [1 / (mu + 1 - min_points) for mu in cluster_mu[:n_cluster]]  # type: ignore
        # p for a geometric distribution sampling points from min_points to inf is 1 / (mean + 1 - min_points)
        n_offspring_list = rng.geometric(p=p_values, size=n_cluster) - 1 + min_points
        # rng.geometric samples points from 1 to inf.
    except ValueError as e:
        e.args += (f"Too few offspring events for n_cluster: {n_cluster}",)
        raise
    except (TypeError, IndexError):
        n_offspring_list = (
            rng.geometric(p=1 / (cluster_mu + 1 - min_points), size=n_cluster)  # type: ignore
            - 1
            + min_points
        )

    samples_ = []
    labels_ = []
    for i, (parent, std, n_offspring) in enumerate(
        zip(parent_samples, cluster_std_, n_offspring_list)
    ):
        offspring_samples = rng.normal(
            loc=parent, scale=std, size=(n_offspring, region.dimension)  # type: ignore
        )
        samples_.append(offspring_samples)
        labels_ += [i] * len(offspring_samples)

    samples = np.concatenate(samples_) if len(samples_) != 0 else np.array([])
    labels = np.array(labels_)

    if clip is True:
        if len(samples) != 0:
            inside_indices = region.contains(samples)
            samples = samples[inside_indices]
            labels = labels[inside_indices]
        region_ = region
    else:
        region_ = expanded_region

    if shuffle:
        shuffled_indices = rng.permutation(len(samples))
        samples = samples[shuffled_indices]
        labels = labels[shuffled_indices]

    if (
        len(samples) == 0
    ):  # this is to convert empty arrays into arrays with shape (n_samples, n_features).
        samples = np.array([])
        samples = samples[:, np.newaxis]

    return samples, labels, parent_samples, region_


def simulate_dstorm(
    parent_intensity: int | float = 1,
    region: Region | npt.ArrayLike = (0, 1.0),
    expansion_factor: int | float = 6,
    cluster_mu: int | float | Sequence[float] = 1,
    min_points: int = 0,
    cluster_std: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
    clip: bool = True,
    shuffle: bool = True,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Generate clustered point data following a Thomas-like random point process.
    Parent positions are distributed according to a homogeneous Poisson
    process with `parent_intensity`
    within the boundaries given by `region` expanded by an expansion distance
    that equals
    expansion_factor * max(cluster_std).
    Each parent position is then replaced by n offspring points
    where n is geometrically-distributed with mean number `cluster_mu`
    and point coordinates are normal-distributed around the parent point with
    standard deviation `cluster_std`.
    Offspring from parent events that are located outside the region are
    included.

    Parameters
    ----------
    parent_intensity
        The intensity (points per unit region measure) of the Poisson point
        process for parent events.
    region
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_factor
        Factor by which the cluster_std is multiplied to set the region
        expansion distance.
    cluster_mu
        The mean number of points for normal-distributed offspring points.
    min_points
        The minimum number of points per cluster.
    cluster_std
        The standard deviation for normal-distributed offspring points.
    clip
        If True the result will be clipped to 'region'.
        If False the extended region will be kept.
    shuffle
        If True shuffle the samples.
    seed
        random number generation seed

    Returns
    -------
    LocData
        The generated samples.
    """
    parameter = locals()
    samples, labels, _, region = make_dstorm(
        parent_intensity,
        region,
        expansion_factor,
        cluster_mu,
        min_points,
        cluster_std,
        clip,
        shuffle,
        seed,
    )
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)  # type: ignore
    assert region_.dimension is not None  # type narrowing # noqa: S101
    locdata = LocData.from_coordinates(coordinates=samples)
    locdata.dimension = region_.dimension
    locdata.region = region_
    locdata.dataframe = locdata.dataframe.assign(cluster_label=labels)

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=make_dstorm.__name__, parameter=str(parameter))

    return locdata


def _random_walk(
    n_walks: int = 1,
    n_steps: int = 10,
    dimensions: int = 2,
    diffusion_constant: int | float = 1,
    time_step: float = 10,
    seed: RandomGeneratorSeed = None,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Random walk simulation

    Parameters
    ----------
    n_walks
        Number of walks
    n_steps
        Number of time steps (i.e. frames)
    dimensions
        spatial dimensions to simulate
    diffusion_constant
        Diffusion constant in units length per seconds^2
    time_step
        Time per frame (or simulation step) in seconds.
    seed
        random number generation seed

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        (times, positions), where shape(times) is 1 and shape of positions
        is (n_walks, n_steps, dimensions)
    """
    rng = np.random.default_rng(seed)
    # equally spaced time steps
    times = np.arange(n_steps) * time_step

    # random step sizes according to the diffusion constant
    random_numbers = rng.integers(
        0, 2, size=(n_walks, n_steps, dimensions)
    )  # np.random.randint(0, 2, size=(n_walks, n_steps, dimensions))
    step_size = np.sqrt(2 * dimensions * diffusion_constant * time_step)
    steps = np.where(random_numbers == 0, -step_size, +step_size)

    # walker positions
    positions = np.cumsum(steps, axis=1)

    return times, positions


def simulate_tracks(
    n_walks: int = 1,
    n_steps: int = 10,
    ranges: tuple[tuple[int, int], tuple[int, int]] = ((0, 10000), (0, 10000)),
    diffusion_constant: int | float = 1,
    time_step: float = 10,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Provide a dataset of localizations representing random walks with
    starting points being spatially-distributed
    on a rectangular shape or cubic volume by complete spatial randomness.

    Parameters
    ----------
    n_walks
        Number of walks
    n_steps
        Number of time steps (i.e. frames)
    ranges
        the range for valid x[, y, z]-coordinates
    diffusion_constant
        Diffusion constant with unit length per seconds^2
    time_step
        Time per frame (or simulation step) in seconds.
    seed
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    parameter = locals()

    rng = np.random.default_rng(seed)

    start_positions = np.array(
        [rng.uniform(*_range, size=n_walks) for _range in ranges]
    ).T

    times, positions = _random_walk(
        n_walks=n_walks,
        n_steps=n_steps,
        dimensions=len(ranges),
        diffusion_constant=diffusion_constant,
        time_step=time_step,
        seed=rng,
    )

    new_positions = np.concatenate(
        [
            start_position + position
            for start_position, position in zip(start_positions, positions)
        ]
    )

    locdata_dict = {
        "position_" + label: position_values
        for _, position_values, label in zip(ranges, new_positions.T, ("x", "y", "z"))
    }

    locdata_dict.update(frame=np.tile(range(len(times)), n_walks))

    locdata = LocData.from_dataframe(dataframe=pd.DataFrame(locdata_dict))

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return locdata


def resample(
    locdata: LocData, n_samples: int = 10, seed: RandomGeneratorSeed = None
) -> LocData:
    """
    Resample locdata according to localization uncertainty.
    Per localization `n_samples` new localizations
    are simulated normally distributed around the localization coordinates
    with a standard deviation set to the uncertainty in each dimension.
    Uncertainties are taken from "uncertainty_c" or "uncertainty".
    The resulting LocData object carries new localizations with the following
    new properties: position coordinates, 'original_index'.

    Parameters
    ----------
    locdata
        Localization data to be resampled
    n_samples
        The number of localizations generated for each original localization.
    seed
        random number generation seed

    Returns
    -------
    LocData
        New localization data with simulated coordinates.
    """
    rng = np.random.default_rng(seed)

    available_coordinate_keys = _get_loc_property_key_per_dimension(
        locdata=locdata.data, property_key="position"
    )
    available_uncertainty_keys = _get_loc_property_key_per_dimension(
        locdata=locdata.data, property_key="uncertainty"
    )

    original_index_label = _bump_property_key(
        loc_property="original_index", loc_properties=locdata.data.columns
    )
    new_df = locdata.data.loc[locdata.data.index.repeat(n_samples)].reset_index(  # type: ignore
        names=original_index_label
    )

    for c_label, u_label in zip(available_coordinate_keys, available_uncertainty_keys):
        if c_label is not None:
            if u_label is None:
                logger.warning(f"No uncertainties available for {c_label}.")
            else:
                new_df[c_label] = rng.normal(loc=new_df[c_label], scale=new_df[u_label])

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

    meta_.modification_time.GetCurrentTime()
    meta_.state = metadata_pb2.MODIFIED
    meta_.ancestor_identifiers.append(locdata.meta.identifier)
    meta_.history.add(
        name="resample", parameter=f"locdata={locdata}, n_samples={n_samples}"
    )

    # instantiate
    new_locdata = LocData.from_dataframe(dataframe=new_df, meta=meta_)

    return new_locdata


def _random_poisson_repetitions(
    n_samples: int, lam: float, seed: RandomGeneratorSeed = None
) -> npt.NDArray[np.int_]:
    """
    Return numpy.ndarray of sorted integers with each integer i being
    repeated n(i) times
    where n(i) is drawn from a Poisson distribution with mean `lam`.

    Parameters
    ----------
    n_samples
        number of elements to be returned
    lam
        mean of the Poisson distribution (lambda)
    seed
        random number generation seed

    Returns
    -------
    npt.NDArray[np.int_]
        The generated sequence of integers with shape (n_samples,)
    """
    rng = np.random.default_rng(seed)

    frames = np.ones(n_samples, dtype=int)
    n_random_numbers = n_samples if lam > 0 else int(n_samples / lam)
    position = 0
    current_number = 0
    while position < n_samples:
        repeats = rng.poisson(lam=lam, size=n_random_numbers)
        for repeat in repeats:
            try:
                frames[position : position + repeat] = current_number
            except ValueError:
                frames[position:] = current_number
                break
            position += repeat
            current_number += 1
    return frames


def simulate_frame_numbers(
    n_samples: int, lam: float, seed: RandomGeneratorSeed = None
) -> npt.NDArray[np.int_]:
    """
    Simulate Poisson-distributed frame numbers for a list of localizations.

    Return numpy.ndarray of sorted integers with each integer i being
    repeated n(i) times
    where n(i) is drawn from a Poisson distribution with mean `lam`.

    Use the following to add frame numbers to a given LocData object::

        frames = simulate_frame_numbers(n_samples=len(locdata), lam=5)
        locdata.dataframe = locdata.dataframe.assign(frame = frames)

    Parameters
    ----------
    n_samples
        number of elements to be returned
    lam
        mean of the Poisson distribution (lambda)
    seed
        random number generation seed

    Returns
    -------
    npt.NDArray[np.int_]
        The generated sequence of integers with shape (n_samples,)
    """
    return _random_poisson_repetitions(n_samples, lam, seed=seed)


def randomize(
    locdata: LocData,
    hull_region: Region | Literal["bb", "ch", "as", "obb"] = "bb",
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Transform locdata coordinates into randomized coordinates that follow
    complete spatial randomness on the same region as the input locdata.

    Parameters
    ----------
    locdata
        Localization data to be randomized
    hull_region
        Region of interest.
        String identifier can refer to the corresponding hull.
    seed
        random number generation seed

    Returns
    -------
    LocData
        New localization data with randomized coordinates.
    """
    # todo: fix treatment of empty locdata
    local_parameter = locals()

    rng = np.random.default_rng(seed)

    try:
        if hull_region == "bb":
            region_ = locdata.bounding_box.hull.T  # type: ignore[union-attr]
        elif hull_region == "ch":
            region_ = locdata.convex_hull.region  # type: ignore[union-attr]
        elif hull_region == "as":
            region_ = locdata.alpha_shape.region  # type: ignore[union-attr]
        elif hull_region == "obb":
            region_ = locdata.oriented_bounding_box.region  # type: ignore[union-attr]
        elif isinstance(hull_region, Region):
            region_ = hull_region
        else:
            raise ValueError

        new_locdata = simulate_uniform(n_samples=len(locdata), region=region_, seed=rng)

    except (AttributeError, ValueError, TypeError) as exception:
        raise AttributeError(f"Region {hull_region} is not available.") from exception

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
