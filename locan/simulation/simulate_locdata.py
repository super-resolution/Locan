"""

Simulate localization data.

This module provides functions to simulate localization data and return LocData objects.
Localizations are often distributed either by a spatial process of complete-spatial randomness or following a
Neyman-Scott process [1]_. For a Neyman-Scott process parent events (representing single emitters) yield a random number
of cluster_mu events (representing localizations due to repeated blinking). Related spatial point processes include
Matérn and Thomas processes.

Functions that are named as make_* provide point data arrays. Functions that are named as simulate_* provide
locdata.


Parts of this code is adapted from scikit-learn/sklearn/datasets/_samples_generator.py .
(BSD 3-Clause License, Copyright (c) 2007-2020 The scikit-learn developers.)

References
----------
.. [1] Neyman, J. & Scott, E. L.,
   A Theory of the Spatial Distribution of Galaxies.
   Astrophysical Journal 1952, vol. 116, p.144.

"""
import sys
import time
from itertools import chain

import numpy as np
import pandas as pd

from locan.data import metadata_pb2
from locan.data.locdata import LocData
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

__all__ = [
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
]


def make_uniform(n_samples, region=(0, 1), seed=None):
    """
    Provide points that are distributed by a uniform (complete spatial randomness) point process
    within the boundaries given by `region`.

    Parameters
    ----------
    n_samples : int
        The total number of localizations of the point process
    region : Region, array-like
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        Random number generation seed

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
        The generated samples.
    """
    rng = np.random.default_rng(seed)

    if not isinstance(region, Region):
        region = Region.from_intervals(region)

    if isinstance(region, EmptyRegion):
        samples = np.array([])
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

        samples = []
        n_remaining = n_samples
        while n_remaining > 0:
            new_samples = rng.random(size=(n_samples_updated, region.dimension))
            new_samples = region.extent * new_samples + region.bounding_box.corner
            new_samples = new_samples[region.contains(new_samples)]
            samples.append(new_samples)
            n_remaining = n_remaining - len(new_samples)

        samples = np.concatenate(samples)
        samples = samples[0:n_samples]

    return samples


def simulate_uniform(n_samples, region=(0, 1), seed=None):
    """
    Provide points that are distributed by a uniform Poisson point process within the boundaries given by `region`.

    Parameters
    ----------
    n_samples : int
        The total number of localizations of the point process
    region : Region, array-like
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    LocData
        The generated samples.
    """
    parameter = locals()
    samples = make_uniform(n_samples=n_samples, region=region, seed=seed)
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)
    locdata = LocData.from_coordinates(coordinates=samples)
    locdata.dimension = region_.dimension
    locdata.region = region_

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=make_uniform.__name__, parameter=str(parameter))

    return locdata


def make_Poisson(intensity, region=(0, 1), seed=None):
    """
    Provide points that are distributed by a uniform Poisson point process within the boundaries given by `region`.

    Parameters
    ----------
    intensity : int, float
        The intensity (points per unit region measure) of the point process
    region : Region, array-like
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
        The generated samples.
    """
    rng = np.random.default_rng(seed)

    if not isinstance(region, Region):
        region = Region.from_intervals(region)

    n_samples = rng.poisson(lam=intensity * region.region_measure)

    if isinstance(region, EmptyRegion):
        samples = np.array([])
    elif n_samples == 0:
        samples = np.array([])
        for i in range(region.dimension):
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

        samples = []
        n_remaining = n_samples
        while n_remaining > 0:
            new_samples = rng.random(size=(n_samples_updated, region.dimension))
            new_samples = region.extent * new_samples + region.bounding_box.corner
            new_samples = new_samples[region.contains(new_samples)]
            samples.append(new_samples)
            n_remaining = n_remaining - len(new_samples)

        samples = np.concatenate(samples)
        samples = samples[0:n_samples]

    return samples


def simulate_Poisson(intensity, region=(0, 1), seed=None):
    """
    Provide points that are distributed by a uniform Poisson point process within the boundaries given by `region`.

    Parameters
    ----------
    intensity : int, float
        The intensity (points per unit region measure) of the point process
    region : Region, array-like
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    LocData
        The generated samples.
    """
    parameter = locals()
    samples = make_Poisson(intensity=intensity, region=region, seed=seed)
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)
    locdata = LocData.from_coordinates(coordinates=samples)
    locdata.dimension = region_.dimension
    locdata.region = region_

    # metadata
    locdata.meta.source = metadata_pb2.SIMULATION
    del locdata.meta.history[:]
    locdata.meta.history.add(name=make_Poisson.__name__, parameter=str(parameter))

    return locdata


def make_cluster(
    centers=3,
    region=(0, 1.0),
    expansion_distance=0,
    offspring=None,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Parent positions are taken from `centers`
    or are distributed according to a homogeneous Poisson process with exactly `centers`
    within the boundaries given by `region` expanded by the expansion_distance.
    Each parent position is then replaced by cluster_mu offspring points as passed or generated by a given function.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    centers : int, array-like
        The number of parents or coordinates for parent events, where each parent represents a cluster center.
    region : Region, array-like
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_distance : float
        The distance by which region is expanded on all boundaries.
    offspring : array-like, callable, None
        Points or function for point process to provide cluster.
        Callable must take single parent point as parameter and return an iterable.
        If array-like it must have the same length as parent events.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    tuple of numpy.ndarray of shape (n_samples, n_features) and region
       The generated samples, labels, parent_samples, region
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
        n_centers = centers
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
            labels = [[i] * len(os) for i, os in enumerate(offspring_samples)]
        except TypeError:
            offspring_samples = []
            labels = []
            for i, parent in enumerate(parent_samples):
                offspring_samples_ = offspring(parent)
                offspring_samples.append(offspring_samples_)
                labels.append([i] * len(offspring_samples_))
        samples = np.array(list(chain(*offspring_samples)))
        labels = np.array(list(chain(*labels)))

    elif len(offspring) >= len(parent_samples):
        offspring_samples = []
        labels = []
        for i, (os, parent) in enumerate(zip(offspring[:n_centers], parent_samples)):
            if len(os) > 0:
                offspring_samples_ = np.asarray(os) + parent
                offspring_samples.append(offspring_samples_)
                labels.append([i] * len(offspring_samples_))
        samples = np.array(list(chain(*offspring_samples)))
        labels = np.array(list(chain(*labels)))

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
    centers=3,
    region=(0, 1.0),
    expansion_distance=0,
    offspring=None,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Generate clustered point data.
    Parent positions are taken from `centers`
    or are distributed according to a homogeneous Poisson process with exactly `centers`
    within the boundaries given by `region` expanded by the expansion_distance.
    Each parent position is then replaced by offspring points as passed or generated by a given function.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    centers : int, array-like
        The number of parents or coordinates for parent events, where each parent represents a cluster center.
    region : Region, array-like
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_distance : float
        The distance by which region is expanded on all boundaries.
    offspring : array-like, callable, None
        Points or function for point process to provide cluster.
        Callable must take single parent point as parameter and return an iterable.
        If array-like it must have the same length as parent events.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
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
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)
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
    parent_intensity=100,
    region=(0, 1.0),
    expansion_distance=0,
    offspring=None,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Generate clustered point data following a Neyman-Scott random point process.
    Parent positions are distributed according to a homogeneous Poisson process with `parent_intensity`
    within the boundaries given by `region` expanded by the expansion_distance.
    Each parent position is then replaced by offspring points as passed or generated by a given function.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    parent_intensity : int, float
        The intensity (points per unit region measure) of the Poisson point process for parent events.
    region : Region, array-like
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_distance : float
        The distance by which region is expanded on all boundaries.
    offspring : array-like, callable, None
        Points or function for point process to provide offspring points.
        Callable must take single parent point as parameter.
        If array-like it must have enough elements to fit the randomly generated number of parent events.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    tuple of numpy.ndarray of shape (n_samples, n_features)
       The generated samples, labels, parent_samples
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
            labels = [[i] * len(os) for i, os in enumerate(offspring_samples)]
        except TypeError:
            offspring_samples = []
            labels = []
            for i, parent in enumerate(parent_samples):
                offspring_samples_ = offspring(parent)
                offspring_samples.append(offspring_samples_)
                labels.append([i] * len(offspring_samples_))
        samples = np.array(list(chain(*offspring_samples)))
        labels = np.array(list(chain(*labels)))

    elif len(offspring) >= len(parent_samples):
        offspring_samples = []
        labels = []
        if isinstance(offspring, np.ndarray):
            offspring_samples = (
                np.asarray(offspring[: len(parent_samples)]) + parent_samples
            )
            labels = [[i] * len(os) for i, os in enumerate(offspring_samples)]
        else:
            for i, (os, parent) in enumerate(
                zip(offspring[: len(parent_samples)], parent_samples)
            ):
                if len(os) > 0:
                    offspring_samples_ = np.asarray(os) + parent
                    offspring_samples.append(offspring_samples_)
                    labels.append([i] * len(offspring_samples_))
        samples = np.array(list(chain(*offspring_samples)))
        labels = np.array(list(chain(*labels)))

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
    parent_intensity=100,
    region=(0, 1.0),
    expansion_distance=0,
    offspring=None,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Generate clustered point data following a Neyman-Scott random point process.
    Parent positions are distributed according to a homogeneous Poisson process with `parent_intensity`
    within the boundaries given by `region` expanded by the expansion_distance.
    Each parent position is then replaced by offspring points as passed or generated by a given function.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    parent_intensity : int, float
        The intensity (points per unit region measure) of the Poisson point process for parent events.
    region : Region, array-like
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_distance : float
        The distance by which region is expanded on all boundaries.
    offspring : array-like, callable, None
        Points or function for point process to provide offspring points.
        Callable must take single parent point as parameter.
        If array-like it must have enough elements to fit the randomly generated number of parent events.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
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
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)
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
    parent_intensity=1,
    region=(0, 1.0),
    cluster_mu=1,
    radius=1.0,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Generate clustered point data following a Matern cluster random point process.
    Parent positions are distributed according to a homogeneous Poisson process with `parent_intensity`
    within the boundaries given by `region` expanded by the maximum radius.
    Each parent position is then replaced by spots of size `radius` with Poisson distributed points inside.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    parent_intensity : int, float
        The intensity (points per unit region measure) of the Poisson point process for parent events.
    region : Region, array-like
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    cluster_mu : int, float
        The mean number of points of the Poisson point process for cluster(cluster_mu) events.
    radius : float or sequence of floats
        The radius for the spots. If tuple, the number of elements must be larger than the expected number of parents.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    tuple of numpy.ndarray of shape (n_samples, n_features)
       The generated samples, labels, parent_samples
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
            radii = radius[: len(parent_samples)]
    else:  # if isinstance(radius, float):
        radii = np.full(len(parent_samples), radius)

    # replace parents by offspring samples
    samples = []
    labels = []
    for i, (parent, radius_) in enumerate(zip(parent_samples, radii)):
        if region.dimension == 1:
            offspring_region = Interval(-radius_, radius_)
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
            samples.append(offspring_samples)
        labels += [i] * len(offspring_samples)

    samples = np.array(list(chain(*samples)))
    labels = np.array(labels)

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
    parent_intensity=1,
    region=(0, 1.0),
    cluster_mu=1,
    radius=1.0,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Generate clustered point data following a Matern cluster random point process.
    Parent positions are distributed according to a homogeneous Poisson process with `parent_intensity`
    within the boundaries given by `region` expanded by the maximum radius.
    Each parent position is then replaced by spots of size `radius` with Poisson distributed points inside.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    parent_intensity : int, float
        The intensity (points per unit region measure) of the Poisson point process for parent events.
    region : Region, array-like
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    cluster_mu : int, float
        The mean number of points of the Poisson point process for cluster(cluster_mu) events.
    radius : float or sequence of floats
        The radius for the spots. If tuple, the number of elements must be larger than the expected number of parents.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
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
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)
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
    parent_intensity=1,
    region=(0, 1.0),
    expansion_factor=6,
    cluster_mu=1,
    cluster_std=1.0,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Generate clustered point data following a Thomas random point process.
    Parent positions are distributed according to a homogeneous Poisson process with `parent_intensity`
    within the boundaries given by `region` expanded by an expansion distance that equals
    expansion_factor * max(cluster_std).
    Each parent position is then replaced by n offspring points
    where n is Poisson-distributed with mean number `cluster_mu`
    and point coordinates are normal-distributed around the parent point with standard deviation `cluster_std`.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    parent_intensity : int, float
        The intensity (points per unit region measure) of the Poisson point process for parent events.
    region : Region, array-like
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_factor : int, float
        Factor by which the cluster_std is multiplied to set the region expansion distance.
    cluster_mu : int, float, sequence of floats
        The mean number of points for normal-distributed offspring points.
    cluster_std : float, sequence of floats, sequence of sequence of floats
        The standard deviation for normal-distributed offspring points.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    tuple of numpy.ndarray of shape (n_samples, n_features)
       The generated samples, labels, parent_samples
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
            shape=(n_cluster, region.dimension), fill_value=cluster_std
        )
    elif len(np.shape(cluster_std)) == 1:  # iterate over cluster_std for each feature
        if region.dimension == 1 or len(cluster_std) != region.dimension:
            raise TypeError(
                f"The shape of cluster_std {np.shape(cluster_std)} is incompatible "
                f"with n_features {region.dimension}."
            )
        else:
            cluster_std_ = np.empty(shape=(n_cluster, region.dimension))
            for i, element in enumerate(cluster_std):
                cluster_std_[:, i] = np.full((n_cluster,), element)
    elif len(np.shape(cluster_std)) == 2:  # iterate over cluster_std for each center
        if np.shape(cluster_std) < (n_cluster, region.dimension):
            raise TypeError(
                f"The shape of cluster_std {np.shape(cluster_std)} is incompatible with "
                f"n_cluster {n_cluster} or n_features {region.dimension}."
            )
        else:
            cluster_std_ = cluster_std
    else:
        raise TypeError(
            f"The shape of cluster_std {np.shape(cluster_std)} is incompatible."
        )

    # replace parents by normal-distributed offspring samples
    try:
        n_offspring_list = rng.poisson(lam=cluster_mu[:n_cluster], size=n_cluster)
    except ValueError as e:
        e.args += (f"Too few offspring events for n_cluster: {n_cluster}",)
        raise
    except (TypeError, IndexError):
        n_offspring_list = rng.poisson(lam=cluster_mu, size=n_cluster)
    samples = []
    labels = []
    for i, (parent, std, n_offspring) in enumerate(
        zip(parent_samples, cluster_std_, n_offspring_list)
    ):
        offspring_samples = rng.normal(
            loc=parent, scale=std, size=(n_offspring, region.dimension)
        )
        samples.append(offspring_samples)
        labels += [i] * len(offspring_samples)

    samples = np.concatenate(samples) if len(samples) != 0 else np.array([])
    labels = np.array(labels)

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
    parent_intensity=1,
    region=(0, 1.0),
    expansion_factor=6,
    cluster_mu=1,
    cluster_std=1.0,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Generate clustered point data following a Thomas random point process.
    Parent positions are distributed according to a homogeneous Poisson process with `parent_intensity`
    within the boundaries given by `region` expanded by an expansion distance that equals
    expansion_factor * max(cluster_std).
    Each parent position is then replaced by n offspring points
    where n is Poisson-distributed with mean number `cluster_mu`
    and point coordinates are normal-distributed around the parent point with standard deviation `cluster_std`.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    parent_intensity : int, float
        The intensity (points per unit region measure) of the Poisson point process for parent events.
    region : Region, array-like
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_factor : int, float
        Factor by which the cluster_std is multiplied to set the region expansion distance.
    cluster_mu : int, float, sequence of floats
        The mean number of points for normal-distributed offspring points.
    cluster_std : float, sequence of floats, sequence of sequence of floats
        The standard deviation for normal-distributed offspring points.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
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
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)
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
    parent_intensity=1,
    region=(0, 1.0),
    expansion_factor=6,
    cluster_mu=1,
    min_points=0,
    cluster_std=1.0,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Generate clustered point data following a Thomas-like random point process.
    Parent positions are distributed according to a homogeneous Poisson process with `parent_intensity`
    within the boundaries given by `region` expanded by an expansion distance that equals
    expansion_factor * max(cluster_std).
    Each parent position is then replaced by n offspring points
    where n is geometrically-distributed with mean number `cluster_mu`
    and point coordinates are normal-distributed around the parent point with standard deviation `cluster_std`.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    parent_intensity : int, float
        The intensity (points per unit region measure) of the Poisson point process for parent events.
    region : Region, array-like
        The region (or support) for all features.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_factor : int, float
        Factor by which the cluster_std is multiplied to set the region expansion distance.
    cluster_mu : int, float, sequence of floats
        The mean number of points for normal-distributed offspring points.
    min_points : int
        The minimum number of points per cluster.
    cluster_std : float, sequence of floats, sequence of sequence of floats
        The standard deviation for normal-distributed offspring points.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    tuple of numpy.ndarray of shape (n_samples, n_features)
       The generated samples, labels, parent_samples
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
            shape=(n_cluster, region.dimension), fill_value=cluster_std
        )
    elif len(np.shape(cluster_std)) == 1:  # iterate over cluster_std for each feature
        if region.dimension == 1 or len(cluster_std) != region.dimension:
            raise TypeError(
                f"The shape of cluster_std {np.shape(cluster_std)} is incompatible "
                f"with n_features {region.dimension}."
            )
        else:
            cluster_std_ = np.empty(shape=(n_cluster, region.dimension))
            for i, element in enumerate(cluster_std):
                cluster_std_[:, i] = np.full((n_cluster,), element)
    elif len(np.shape(cluster_std)) == 2:  # iterate over cluster_std for each center
        if np.shape(cluster_std) < (n_cluster, region.dimension):
            raise TypeError(
                f"The shape of cluster_std {np.shape(cluster_std)} is incompatible with "
                f"n_cluster {n_cluster} or n_features {region.dimension}."
            )
        else:
            cluster_std_ = cluster_std
    else:
        raise TypeError(
            f"The shape of cluster_std {np.shape(cluster_std)} is incompatible."
        )

    # replace parents by normal-distributed offspring samples
    try:
        p_values = [1 / (mu + 1 - min_points) for mu in cluster_mu[:n_cluster]]
        # p for a geometric distribution sampling points from min_points to inf is 1 / (mean + 1 - min_points)
        n_offspring_list = rng.geometric(p=p_values, size=n_cluster) - 1 + min_points
        # rng.geometric samples points from 1 to inf.
    except ValueError as e:
        e.args += (f"Too few offspring events for n_cluster: {n_cluster}",)
        raise
    except (TypeError, IndexError):
        n_offspring_list = (
            rng.geometric(p=1 / (cluster_mu + 1 - min_points), size=n_cluster)
            - 1
            + min_points
        )

    samples = []
    labels = []
    for i, (parent, std, n_offspring) in enumerate(
        zip(parent_samples, cluster_std_, n_offspring_list)
    ):
        offspring_samples = rng.normal(
            loc=parent, scale=std, size=(n_offspring, region.dimension)
        )
        samples.append(offspring_samples)
        labels += [i] * len(offspring_samples)

    samples = np.concatenate(samples) if len(samples) != 0 else np.array([])
    labels = np.array(labels)

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
    parent_intensity=1,
    region=(0, 1.0),
    expansion_factor=6,
    cluster_mu=1,
    min_points=0,
    cluster_std=1.0,
    clip=True,
    shuffle=True,
    seed=None,
):
    """
    Generate clustered point data following a Thomas-like random point process.
    Parent positions are distributed according to a homogeneous Poisson process with `parent_intensity`
    within the boundaries given by `region` expanded by an expansion distance that equals
    expansion_factor * max(cluster_std).
    Each parent position is then replaced by n offspring points
    where n is geometrically-distributed with mean number `cluster_mu`
    and point coordinates are normal-distributed around the parent point with standard deviation `cluster_std`.
    Offspring from parent events that are located outside the region are included.

    Parameters
    ----------
    parent_intensity : int, float
        The intensity (points per unit region measure) of the Poisson point process for parent events.
    region : Region, array-like
        The region (or support) for each feature.
        If array-like it must provide upper and lower bounds for each feature.
    expansion_factor : int, float
        Factor by which the cluster_std is multiplied to set the region expansion distance.
    cluster_mu : int, float, sequence of floats
        The mean number of points for normal-distributed offspring points.
    min_points : int
        The minimum number of points per cluster.
    cluster_std : float, sequence of floats, sequence of sequence of floats
        The standard deviation for normal-distributed offspring points.
    clip : bool
        If True the result will be clipped to 'region'. If False the extended region will be kept.
    shuffle : boolean
        Shuffle the samples.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
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
    region_ = region if isinstance(region, Region) else Region.from_intervals(region)
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
    n_walks=1, n_steps=10, dimensions=2, diffusion_constant=1, time_step=10, seed=None
):
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
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    tuple of arrays
        (times, positions), where shape(times) is 1 and shape of positions is (n_walks, n_steps, dimensions)
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
    n_walks=1,
    n_steps=10,
    ranges=((0, 10000), (0, 10000)),
    diffusion_constant=1,
    time_step=10,
    seed=None,
):
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
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    LocData
        A new LocData instance with localization data.
    """
    parameter = locals()

    start_positions = np.array(
        [np.random.uniform(*_range, size=n_walks) for _range in ranges]
    ).T

    times, positions = _random_walk(
        n_walks=n_walks,
        n_steps=n_steps,
        dimensions=len(ranges),
        diffusion_constant=diffusion_constant,
        time_step=time_step,
        seed=seed,
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


def resample(locdata, n_samples=10, seed=None):
    """
    Resample locdata according to localization uncertainty. Per localization *n_samples* new localizations
    are simulated normally distributed around the localization coordinates with a standard deviation set to the
    uncertainty in each dimension.
    The resulting LocData object carries new localizations with the following properties: 'position_x',
    'position_y'[, 'position_z'], 'origin_index'

    Parameters
    ----------
    locdata : LocData
        Localization data to be resampled
    n_samples : int
        The number of localizations generated for each original localization.
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    locdata : LocData
        New localization data with simulated coordinates.
    """
    rng = np.random.default_rng(seed)

    # generate dataframe
    list_ = []
    for i in range(len(locdata)):
        new_d = {}
        new_d.update({"origin_index": np.full(n_samples, i)})
        x_values = rng.normal(
            loc=locdata.data.iloc[i]["position_x"],
            scale=locdata.data.iloc[i]["uncertainty_x"],
            size=n_samples,
        )
        new_d.update({"position_x": x_values})

        y_values = rng.normal(
            loc=locdata.data.iloc[i]["position_y"],
            scale=locdata.data.iloc[i]["uncertainty_y"],
            size=n_samples,
        )
        new_d.update({"position_y": y_values})

        try:
            z_values = rng.normal(
                loc=locdata.data.iloc[i]["position_z"],
                scale=locdata.data.iloc[i]["uncertainty_z"],
                size=n_samples,
            )
            new_d.update({"position_z": z_values})
        except KeyError:
            pass

        list_.append(pd.DataFrame(new_d))

    dataframe = pd.concat(list_, ignore_index=True)

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
        name="resample", parameter="locdata={}, n_samples={}".format(locdata, n_samples)
    )

    # instantiate
    new_locdata = LocData.from_dataframe(dataframe=dataframe, meta=meta_)

    return new_locdata


def _random_poisson_repetitions(n_samples, lam, seed=None):
    """
    Return numpy.ndarray of sorted integers with each integer i being repeated n(i) times
    where n(i) is drawn from a Poisson distribution with mean `lam`.

    Parameters
    ----------
    n_samples : int
        number of elements to be returned
    lam : float
        mean of the Poisson distribution (lambda)
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    numpy.ndarray with shape (n_samples,)
        The generated sequence of integers.
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


def simulate_frame_numbers(n_samples, lam, seed=None):
    """
    Simulate Poisson-distributed frame numbers for a list of localizations.

    Return numpy.ndarray of sorted integers with each integer i being repeated n(i) times
    where n(i) is drawn from a Poisson distribution with mean `lam`.

    Use the following to add frame numbers to a given LocData object::

        frames = simulate_frame_numbers(n_samples=len(locdata), lam=5)
        locdata.dataframe = locdata.dataframe.assign(frame = frames)

    Parameters
    ----------
    n_samples : int
        number of elements to be returned
    lam : float
        mean of the Poisson distribution (lambda)
    seed : None, int, array_like[ints], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    numpy.ndarray with shape (n_samples,)
        The generated sequence of integers.
    """
    return _random_poisson_repetitions(n_samples, lam, seed=seed)
