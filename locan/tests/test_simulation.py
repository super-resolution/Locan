import warnings
from copy import deepcopy

import matplotlib.pyplot as plt  # needed for visual inspection
import numpy as np
import pandas as pd
import pytest

from locan import (
    Ellipse,
    EmptyRegion,
    Interval,
    LocData,
    Polygon,
    Rectangle,
    add_drift,
    resample,
    simulate_frame_numbers,
    simulate_tracks,
)
from locan.simulation import (
    make_cluster,
    make_dstorm,
    make_Matern,
    make_NeymanScott,
    make_Poisson,
    make_Thomas,
    make_uniform,
    randomize,
    simulate_cluster,
    simulate_dstorm,
    simulate_Matern,
    simulate_NeymanScott,
    simulate_Poisson,
    simulate_Thomas,
    simulate_uniform,
)
from locan.simulation.simulate_drift import _drift, _random_walk_drift


def test_make_uniform():
    rng = np.random.default_rng(seed=1)
    samples = make_uniform(n_samples=10, region=EmptyRegion(), seed=rng)
    assert np.size(samples) == 0
    samples = make_uniform(n_samples=10, region=(0, 1), seed=rng)
    assert len(samples) == 10
    assert samples.shape[1] == 1
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    samples = make_uniform(n_samples=10, region=((0, 1), (10, 11)), seed=rng)
    assert len(samples) == 10
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples = make_uniform(
        n_samples=10, region=((0, 1), (10, 11), (100, 101)), seed=rng
    )
    assert len(samples) == 10
    assert samples.shape[1] == 3
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)
    assert np.all(100 <= samples[:, 2])
    assert np.all(samples[:, 2] < 101)

    samples = make_uniform(n_samples=10, region=Interval(0, 1), seed=rng)
    assert len(samples) == 10
    assert samples.shape[1] == 1
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    samples = make_uniform(n_samples=10, region=Rectangle((0, 10), 1, 1, 0), seed=rng)
    assert len(samples) == 10
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples = make_uniform(n_samples=10, region=Rectangle((0, 10), 1, 1, 45), seed=rng)
    assert len(samples) == 10
    assert samples.shape[1] == 2
    assert np.all(-0.71 <= samples[:, 0])  # 0.71 ~ (np.sqrt(2) / 2)
    assert np.all(samples[:, 0] < 0.71)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < (10 + 1.42))

    samples = make_uniform(n_samples=10, region=Ellipse((0, 0), 1, 1, 0), seed=rng)
    assert len(samples) == 10
    assert samples.shape[1] == 2
    assert np.all(-0.5 <= samples[:, 0])
    assert np.all(samples[:, 0] < 0.5)
    assert np.all(-0.5 <= samples[:, 1])
    assert np.all(samples[:, 1] < 0.5)

    samples = make_uniform(n_samples=10, region=Ellipse((0, 0), 1, 2, 0), seed=rng)
    assert len(samples) == 10
    assert samples.shape[1] == 2
    assert np.all(-0.5 <= samples[:, 0])
    assert np.all(samples[:, 0] < 0.5)
    assert np.all(-1 <= samples[:, 1])
    assert np.all(samples[:, 1] < 1)

    samples = make_uniform(
        n_samples=10, region=Polygon(((0, 10), (1, 11), (1, 10))), seed=rng
    )
    assert len(samples) == 10
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)


def test_simulate_uniform():
    rng = np.random.default_rng(seed=1)
    locdata = simulate_uniform(n_samples=10, region=(0, 1), seed=rng)
    assert len(locdata) == 10
    assert locdata.dimension == 1
    assert repr(locdata.region) == "Interval(0, 1)"


def test_make_Poisson():
    rng = np.random.default_rng(seed=1)

    samples = make_Poisson(intensity=10, region=EmptyRegion(), seed=rng)
    assert np.size(samples) == 0
    assert samples.ndim == 1

    samples = make_Poisson(intensity=1e-10, region=(0, 1), seed=rng)
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples = make_Poisson(intensity=1e-10, region=((0, 1), (10, 11)), seed=rng)
    assert np.size(samples) == 0
    assert samples.ndim == 3

    samples = make_Poisson(intensity=10, region=(0, 1), seed=rng)
    assert samples.shape[1] == 1
    assert samples.ndim == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    samples = make_Poisson(intensity=10, region=((0, 1), (10, 11)), seed=rng)
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples = make_Poisson(
        intensity=10, region=((0, 1), (10, 11), (100, 101)), seed=rng
    )
    assert samples.shape[1] == 3
    assert samples.ndim == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)
    assert np.all(100 <= samples[:, 2])
    assert np.all(samples[:, 2] < 101)

    samples = make_Poisson(intensity=10, region=Interval(0, 1), seed=rng)
    assert samples.shape[1] == 1
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    samples = make_Poisson(intensity=10, region=Rectangle((0, 10), 1, 1, 0), seed=rng)
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples = make_Poisson(intensity=10, region=Ellipse((0, 0), 1, 1, 0), seed=rng)
    assert samples.shape[1] == 2
    assert np.all(-0.5 <= samples[:, 0])
    assert np.all(samples[:, 0] < 0.5)
    assert np.all(-0.5 <= samples[:, 1])
    assert np.all(samples[:, 1] < 0.5)

    samples = make_Poisson(intensity=10, region=Ellipse((0, 0), 1, 2, 0), seed=rng)
    assert samples.shape[1] == 2
    assert np.all(-0.5 <= samples[:, 0])
    assert np.all(samples[:, 0] < 0.5)
    assert np.all(-1 <= samples[:, 1])
    assert np.all(samples[:, 1] < 1)

    samples = make_Poisson(
        intensity=10, region=Polygon(((0, 10), (1, 11), (1, 10))), seed=rng
    )
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)


@pytest.mark.visual
def test_make_Poisson_visual():
    rng = np.random.default_rng(seed=1)

    samples = make_Poisson(
        intensity=1000, region=Polygon(((0, 10), (1, 11), (1, 10))), seed=rng
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples[:, 0], samples[:, 1])
    ax.axis("equal")
    plt.show()


def test_simulate_Poisson():
    rng = np.random.default_rng(seed=1)
    locdata = simulate_Poisson(intensity=10, region=(0, 1), seed=rng)
    assert locdata.dimension == 1
    assert repr(locdata.region) == "Interval(0, 1)"


def test_make_cluster():
    rng = np.random.default_rng(seed=1)

    samples, labels, parent_samples, region = make_cluster(
        centers=10, region=EmptyRegion(), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 1

    samples, labels, parent_samples, region = make_cluster(
        centers=0, region=(0, 1), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=(0, 1),
        expansion_distance=1,
        offspring=None,
        clip=False,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert len(parent_samples) == 10
    assert np.all(-1 <= samples[:, 0])
    assert np.all(samples[:, 0] < 2)

    samples, labels, parent_samples, region = make_cluster(
        centers=(-1, 0, 0.5, 1, 2),
        region=(0, 1),
        expansion_distance=1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert len(parent_samples) == 5
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    offspring = [np.linspace(-0.1, 0.1, 11)] * 20
    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=(0, 1),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert len(parent_samples) == 10
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    with pytest.raises(TypeError):
        offspring = [np.linspace(-0.1, 0.1, 11)] * 2
        make_cluster(
            centers=10,
            region=(0, 1),
            expansion_distance=1,
            offspring=offspring,
            clip=True,
            shuffle=True,
            seed=rng,
        )

    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=((0, 1), (10, 11), (100, 101)),
        expansion_distance=0.1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert len(parent_samples) == 10
    assert samples.ndim == 2
    assert samples.shape[1] == 3
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)
    assert np.all(100 <= samples[:, 2])
    assert np.all(samples[:, 2] < 101)

    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=Interval(0, 1),
        expansion_distance=1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert len(parent_samples) == 10
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=Rectangle((0, 10), 1, 1, 0),
        expansion_distance=1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert len(parent_samples) == 10
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples, labels, parent_samples, region = make_cluster(
        centers=((-0.5, 10), (0.5, 11), (1, 0)),
        region=Rectangle((0, 10), 1, 1, 0),
        expansion_distance=1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert len(parent_samples) == 3
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] <= 11)

    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_distance=1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert len(parent_samples) == 10
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    def offspring(parent):
        angles = np.linspace(0, 360, 36)
        for angle in angles:
            circle = Ellipse(parent, 1, 1, angle)
        return circle.points

    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    def offspring(parent):
        from math import sin

        if np.ndim(parent) == 0:
            return [sin(parent)]
        elif np.ndim(parent) == 1:
            return [sin(parent[0])]
        else:
            raise TypeError

    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=(0, 1),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    offspring = [
        make_uniform(n_samples=20, region=Ellipse((0, 0), 0.1, 0.1), seed=rng)
    ] * 10
    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_distance=0.1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    offspring = [
        np.array([[1, 2], [1, 3], [2, 3]]),
        np.array([]),
        np.array([[1, 2], [1, 3], [2, 3]]),
    ]
    samples, labels, parent_samples, region = make_cluster(
        centers=3,
        region=((0, 10), (0, 10)),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 10)
    assert np.all(0 <= samples[:, 1])
    assert np.all(samples[:, 1] < 10)

    offspring = [np.array([])] * 20
    samples, labels, parent_samples, region = make_cluster(
        centers=3,
        region=((0, 10), (0, 10)),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == 0
    assert samples.ndim == 2


@pytest.mark.visual
def test_make_cluster_visual():
    rng = np.random.default_rng(seed=1)

    samples, labels, parent_samples, region = make_cluster(
        centers=10,
        region=(0, 1),
        expansion_distance=1,
        offspring=[np.linspace(-0.1, 0.1, 11)] * 10,
        clip=False,
        shuffle=True,
        seed=rng,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples, [0] * len(samples), alpha=0.1)
    plt.scatter(parent_samples, [0] * len(parent_samples), color="Red", alpha=0.1)
    ax.axis("equal")
    plt.show()

    def offspring_points(parent):
        angles = np.linspace(0, 360, 36)
        for angle in angles:
            circle = Ellipse(parent, 1, 1, angle)
        return circle.points

    samples, labels, parent_samples, region = make_cluster(
        centers=5,
        region=Polygon(((0, 10), (10, 20), (10, 10))),
        offspring=offspring_points,
        clip=True,
        shuffle=True,
        seed=rng,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples[:, 0], samples[:, 1])
    ax.axis("equal")
    plt.show()


def test_simulate_cluster():
    rng = np.random.default_rng(seed=1)
    locdata = simulate_cluster(
        centers=10,
        region=(0, 1),
        expansion_distance=1,
        offspring=None,
        clip=False,
        shuffle=False,
        seed=rng,
    )
    assert locdata.dimension == 1
    assert repr(locdata.region) == "Interval(-1, 2)"
    assert "cluster_label" in locdata.data.columns


def test_make_NeymanScott():
    rng = np.random.default_rng(seed=1)

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10, region=EmptyRegion(), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 1

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=1e-10, region=(0, 1), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=(0, 1),
        expansion_distance=1,
        offspring=None,
        clip=False,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-1 <= samples[:, 0])
    assert np.all(samples[:, 0] < 2)

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=(0, 1),
        expansion_distance=0,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    offspring = [np.linspace(-0.1, 0.1, 11)] * 200
    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=(0, 1),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-1 <= samples[:, 0])
    assert np.all(samples[:, 0] < 2)

    with pytest.raises(TypeError):
        offspring = [np.linspace(-0.1, 0.1, 11)] * 2
        make_NeymanScott(
            parent_intensity=10,
            region=(0, 1),
            expansion_distance=1,
            offspring=offspring,
            clip=True,
            shuffle=True,
            seed=rng,
        )

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=((0, 1), (10, 11), (100, 101)),
        expansion_distance=1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 3
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)
    assert np.all(100 <= samples[:, 2])
    assert np.all(samples[:, 2] < 101)

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=Interval(0, 1),
        expansion_distance=1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=Rectangle((0, 10), 1, 1, 0),
        expansion_distance=1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_distance=1,
        offspring=None,
        clip=False,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.shape[1] == 2
    assert samples.ndim == 2
    assert np.all(-1 <= samples[:, 0])
    assert np.all(samples[:, 0] < 2)
    assert np.all(9 <= samples[:, 1])
    assert np.all(samples[:, 1] < 12)

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_distance=1,
        offspring=None,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.shape[1] == 2
    assert samples.ndim == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    def offspring(parent):
        angles = np.linspace(0, 360, 36)
        for angle in angles:
            circle = Ellipse(parent, 1, 1, angle)
        return circle.points

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.shape[1] == 2
    assert samples.ndim == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    def offspring(parent):
        from math import sin

        if np.ndim(parent) == 0:
            return [sin(parent)]
        elif np.ndim(parent) == 1:
            return [sin(parent[0])]
        else:
            raise TypeError

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=10,
        region=(0, 1),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.shape[1] == 1
    assert samples.ndim == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)

    offspring = [
        np.array([[1, 2], [1, 3], [2, 3]]),
        np.array([]),
        np.array([[1, 2], [1, 3], [2, 3]]),
    ] * 20
    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=0.1,
        region=((0, 10), (0, 10)),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.shape[1] == 2
    assert samples.ndim == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 10)

    offspring = [np.array([])] * 20
    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=0.1,
        region=((0, 10), (0, 10)),
        expansion_distance=1,
        offspring=offspring,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == 0
    assert samples.ndim == 2


@pytest.mark.visual
def test_make_NeymanScott_visual():
    rng = np.random.default_rng(seed=1)

    p_intensity = 0.5
    region = Polygon(((0, 10), (10, 20), (10, 10)))

    def offspring_points(parent):
        angles = np.linspace(0, 360, 36)
        for angle in angles:
            circle = Ellipse(parent, 1, 1, angle)
        return circle.points

    samples, labels, parent_samples, region = make_NeymanScott(
        parent_intensity=p_intensity,
        region=region,
        expansion_distance=1,
        offspring=offspring_points,
        clip=True,
        shuffle=True,
        seed=rng,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples[:, 0], samples[:, 1])
    ax.axis("equal")
    plt.show()


def test_simulate_NeymanScott():
    rng = np.random.default_rng(seed=1)
    locdata = simulate_NeymanScott(
        parent_intensity=10,
        region=(0, 1),
        expansion_distance=0,
        offspring=None,
        clip=False,
        shuffle=True,
        seed=rng,
    )
    assert locdata.dimension == 1
    assert repr(locdata.region) == "Interval(0, 1)"
    assert "cluster_label" in locdata.data.columns


def test_make_Matern():
    rng = np.random.default_rng(seed=1)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=10, region=EmptyRegion(), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 1

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=1e-10, region=(0, 10), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=1e-10, region=((0, 1), (10, 11)), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=10,
        region=(0, 10),
        cluster_mu=10,
        radius=1.0,
        clip=False,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 12)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=10,
        region=(0, 10),
        cluster_mu=1,
        radius=1.0,
        clip=False,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 12)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=10,
        region=(0, 10),
        cluster_mu=10,
        radius=np.linspace(1, 2, 200),
        clip=False,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-4 <= samples[:, 0])
    assert np.all(samples[:, 0] < 14)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=10,
        region=(0, 10),
        cluster_mu=10,
        radius=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 10)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=10,
        region=((0, 1), (10, 11)),
        cluster_mu=10,
        radius=1,
        clip=False,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)
    assert np.all(8 <= samples[:, 1])
    assert np.all(samples[:, 1] < 13)

    with pytest.raises(NotImplementedError):
        samples, labels, parent_samples, region = make_Matern(
            parent_intensity=10,
            region=((0, 1), (10, 11), (100, 101)),
            cluster_mu=10,
            radius=1,
            clip=False,
            shuffle=True,
            seed=rng,
        )
        assert len(samples) == len(labels)
        assert samples.ndim == 2
        assert samples.shape[1] == 3
        assert np.all(-1 <= samples[:, 0])
        assert np.all(samples[:, 0] < 2)
        assert np.all(9 <= samples[:, 1])
        assert np.all(samples[:, 1] < 12)
        assert np.all(99 <= samples[:, 1])
        assert np.all(samples[:, 1] < 102)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=10,
        region=Interval(0, 1),
        cluster_mu=10,
        radius=1,
        clip=False,
        shuffle=True,
        seed=rng,
    )
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=10,
        region=Rectangle((0, 10), 1, 1, 0),
        cluster_mu=10,
        radius=1,
        clip=False,
        shuffle=True,
        seed=rng,
    )
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)
    assert np.all(8 <= samples[:, 1])
    assert np.all(samples[:, 1] < 13)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        cluster_mu=10,
        radius=1,
        clip=False,
        shuffle=True,
        seed=rng,
    )
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)
    assert np.all(8 <= samples[:, 1])
    assert np.all(samples[:, 1] < 13)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=100,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        cluster_mu=10,
        radius=1,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)


@pytest.mark.visual
def test_make_Matern_visual():
    rng = np.random.default_rng(seed=1)

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=1,
        region=(0, 10),
        cluster_mu=10,
        radius=1.0,
        clip=False,
        shuffle=False,
        seed=rng,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples, [0] * len(samples), alpha=0.1)
    ax.axis("equal")
    plt.show()

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=0.1,
        region=Polygon(((0, 10), (10, 20), (10, 10))),
        cluster_mu=200,
        radius=np.linspace(0.1, 3, 100),
        clip=False,
        shuffle=False,
        seed=rng,
    )
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples[:, 0], samples[:, 1])
    ax.axis("equal")
    plt.show()

    samples, labels, parent_samples, region = make_Matern(
        parent_intensity=0.1,
        region=Polygon(((0, 10), (10, 20), (10, 10))),
        cluster_mu=200,
        radius=1,
        clip=True,
        shuffle=True,
        seed=rng,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples[:, 0], samples[:, 1])
    ax.axis("equal")
    plt.show()


def test_simulate_Matern():
    rng = np.random.default_rng(seed=1)
    locdata = simulate_Matern(
        parent_intensity=10,
        region=(0, 10),
        cluster_mu=10,
        radius=1.0,
        clip=False,
        shuffle=False,
        seed=rng,
    )
    assert locdata.dimension == 1
    assert repr(locdata.region) == "Interval(-1.0, 11.0)"
    assert "cluster_label" in locdata.data.columns


def test_make_Thomas():
    rng = np.random.default_rng(seed=1)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10, region=EmptyRegion(), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 1

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=0, region=(0, 10), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10, region=(0, 10), cluster_mu=0, seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=1e-10, region=(0, 10), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=(0, 10),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 16)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 10)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=(0, 10),
        expansion_factor=0,
        cluster_mu=0.1,
        cluster_std=1.0,
        clip=True,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 16)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 10)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=np.float64(10),
        region=(0, 10),
        expansion_factor=np.float64(1),
        cluster_mu=np.float64(10),
        cluster_std=np.float64(1),
        clip=True,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) > 0

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=(0, 10),
        expansion_factor=1,
        cluster_mu=np.linspace(1, 10, 1000),
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 16)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 10)

    with pytest.raises(ValueError):
        samples, labels, parent_samples, region = make_Thomas(
            parent_intensity=10,
            region=(0, 10),
            expansion_factor=1,
            cluster_mu=np.linspace(1, 10, 10),
            cluster_std=1.0,
            clip=True,
            shuffle=True,
            seed=rng,
        )

    with pytest.raises(TypeError):
        samples, labels, parent_samples, region = make_Thomas(
            parent_intensity=10,
            region=(0, 10),
            expansion_factor=6,
            cluster_mu=10,
            cluster_std=(1, 2),
            clip=True,
            shuffle=True,
            seed=rng,
        )

    with pytest.raises(TypeError):
        samples, labels, parent_samples, region = make_Thomas(
            parent_intensity=10,
            region=(0, 10),
            expansion_factor=6,
            cluster_mu=10,
            cluster_std=np.linspace(0.2, 2, 200),
            clip=True,
            shuffle=True,
            seed=rng,
        )

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=((0, 1), (10, 11)),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)
    assert np.all(8 <= samples[:, 1])
    assert np.all(samples[:, 1] < 13)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=((0, 1), (10, 11), (100, 101)),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 3
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(94 <= parent_samples[:, 2])
    assert np.all(parent_samples[:, 2] < 107)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)
    assert np.all(100 <= samples[:, 2])
    assert np.all(samples[:, 2] < 101)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=Interval(0, 1),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=Rectangle((0, 10), 1, 1, 0),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)
    assert np.all(8 <= samples[:, 1])
    assert np.all(samples[:, 1] < 13)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)
    assert np.all(8 <= samples[:, 1])
    assert np.all(samples[:, 1] < 13)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=0,
        cluster_mu=np.linspace(10, 20, 200),
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=0,
        cluster_mu=10,
        cluster_std=(0.1, 1),
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 1)
    assert np.all(10 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 11)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=0,
        cluster_mu=10,
        cluster_std=np.linspace((0.01, 0.02), (0.1, 0.2), 100),
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 1)
    assert np.all(10 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 11)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)


@pytest.mark.visual
def test_make_Thomas_visual():
    rng = np.random.default_rng(seed=1)

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=0.2,
        region=(0, 10),
        expansion_factor=6,
        cluster_mu=100,
        cluster_std=1.0,
        clip=True,
        shuffle=False,
        seed=rng,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples, [0] * len(samples), alpha=0.1)
    plt.scatter(parent_samples, [0] * len(parent_samples), c="Red", alpha=0.2)
    ax.axis("equal")
    plt.show()

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=50,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=0,
        cluster_mu=100,
        cluster_std=np.linspace((0.001, 0.002), (0.02, 0.04), 100),
        clip=True,
        shuffle=False,
        seed=rng,
    )
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.scatter(parent_samples[:, 0], parent_samples[:, 1], c="Red")
    ax.axis("equal")
    plt.show()

    samples, labels, parent_samples, region = make_Thomas(
        parent_intensity=20,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=2,
        cluster_mu=500,
        cluster_std=0.02,
        clip=True,
        shuffle=True,
        seed=rng,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.scatter(parent_samples[:, 0], parent_samples[:, 1], c="Red")
    ax.axis("equal")
    plt.show()


def test_simulate_Thomas():
    rng = np.random.default_rng(seed=1)
    locdata = simulate_Thomas(
        parent_intensity=10,
        region=(0, 10),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=False,
        seed=rng,
    )
    assert locdata.dimension == 1
    assert repr(locdata.region) == "Interval(0, 10)"
    assert "cluster_label" in locdata.data.columns


def test_make_dstorm():
    rng = np.random.default_rng(seed=1)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10, region=EmptyRegion(), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 1

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=0, region=(0, 10), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=1e-10, region=(0, 10), seed=rng
    )
    assert np.size(samples) == 0
    assert samples.ndim == 2

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=(0, 10),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 16)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 10)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=np.float64(10),
        region=(0, 10),
        expansion_factor=np.float64(1),
        cluster_mu=np.float64(10),
        cluster_std=np.float64(1),
        clip=True,
        shuffle=False,
        seed=rng,
    )
    assert len(samples) > 0

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=(0, 10),
        expansion_factor=1,
        cluster_mu=np.linspace(1, 10, 1000),
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 16)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 10)

    with pytest.raises(ValueError):
        samples, labels, parent_samples, region = make_dstorm(
            parent_intensity=10,
            region=(0, 10),
            expansion_factor=1,
            cluster_mu=np.linspace(1, 10, 10),
            cluster_std=1.0,
            clip=True,
            shuffle=True,
            seed=rng,
        )

    with pytest.raises(TypeError):
        samples, labels, parent_samples, region = make_dstorm(
            parent_intensity=10,
            region=(0, 10),
            expansion_factor=6,
            cluster_mu=10,
            cluster_std=(1, 2),
            clip=True,
            shuffle=True,
            seed=rng,
        )

    with pytest.raises(TypeError):
        samples, labels, parent_samples, region = make_dstorm(
            parent_intensity=10,
            region=(0, 10),
            expansion_factor=6,
            cluster_mu=10,
            cluster_std=np.linspace(0.2, 2, 200),
            clip=True,
            shuffle=True,
            seed=rng,
        )

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=((0, 1), (10, 11)),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)
    assert np.all(8 <= samples[:, 1])
    assert np.all(samples[:, 1] < 13)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=((0, 1), (10, 11), (100, 101)),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 3
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(94 <= parent_samples[:, 2])
    assert np.all(parent_samples[:, 2] < 107)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)
    assert np.all(100 <= samples[:, 2])
    assert np.all(samples[:, 2] < 101)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=Interval(0, 1),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=Rectangle((0, 10), 1, 1, 0),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)
    assert np.all(8 <= samples[:, 1])
    assert np.all(samples[:, 1] < 13)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(-2 <= samples[:, 0])
    assert np.all(samples[:, 0] < 3)
    assert np.all(8 <= samples[:, 1])
    assert np.all(samples[:, 1] < 13)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=0,
        cluster_mu=np.linspace(10, 20, 200),
        cluster_std=1.0,
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(-6 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 7)
    assert np.all(4 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 17)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=0,
        cluster_mu=10,
        cluster_std=(0.1, 1),
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 1)
    assert np.all(10 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 11)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=Polygon(((0, 10), (1, 11), (1, 10))),
        expansion_factor=0,
        cluster_mu=10,
        cluster_std=np.linspace((0.01, 0.02), (0.1, 0.2), 100),
        clip=True,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.all(0 <= parent_samples[:, 0])
    assert np.all(parent_samples[:, 0] < 1)
    assert np.all(10 <= parent_samples[:, 1])
    assert np.all(parent_samples[:, 1] < 11)
    assert np.all(0 <= samples[:, 0])
    assert np.all(samples[:, 0] < 1)
    assert np.all(10 <= samples[:, 1])
    assert np.all(samples[:, 1] < 11)

    samples, labels, parent_samples, region = make_dstorm(
        parent_intensity=10,
        region=(0, 10),
        expansion_factor=0,
        cluster_mu=10,
        min_points=5,
        cluster_std=0.1,
        clip=False,
        shuffle=True,
        seed=rng,
    )
    assert len(samples) > 0
    assert len(samples) == len(labels)
    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert np.bincount(labels).mean() == pytest.approx(10, rel=0.2)


def test_simulate_dstorm():
    rng = np.random.default_rng(seed=1)
    locdata = simulate_dstorm(
        parent_intensity=10,
        region=(0, 10),
        expansion_factor=6,
        cluster_mu=10,
        cluster_std=1.0,
        clip=True,
        shuffle=False,
        seed=rng,
    )
    assert locdata.dimension == 1
    assert repr(locdata.region) == "Interval(0, 10)"
    assert "cluster_label" in locdata.data.columns


@pytest.fixture()
def locdata_simple():
    localization_dict = {
        "position_x": [0, 0, 1, 4, 5],
        "position_y": [0, 1, 3, 4, 1],
        "position_z": [0, 1, 3, 4, 1],
        "intensity": [0, 1, 3, 4, 1],
        "uncertainty_y": [10, 30, 100, 300, 10],
        "uncertainty_z": [10, 30, 100, 300, 10],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(localization_dict))


def test_resample(locdata_simple, caplog):
    dat = resample(locdata=locdata_simple, n_samples=3)
    assert len(dat) == 15
    assert len(locdata_simple) == 5
    assert all(column_ in dat.data.columns for column_ in locdata_simple.data.columns)
    assert caplog.record_tuples == [
        (
            "locan.simulation.simulate_locdata",
            30,
            "No uncertainties available for position_x.",
        )
    ]

    dat = resample(locdata=locdata_simple, n_samples=3)
    assert len(dat) == 15
    assert all(column_ in dat.data.columns for column_ in locdata_simple.data.columns)


def test_simulate_tracks():
    dat = simulate_tracks(n_walks=2, n_steps=3)
    # print(dat.data)
    # print(dat.meta)
    assert len(dat) == 6
    assert len(dat.coordinate_keys) == 2


def test__random_walk_drift():
    cumsteps = _random_walk_drift(
        n_steps=10, diffusion_constant=(1, 10), velocity=(0, 0), seed=1
    )
    assert cumsteps.shape == (2, 10)
    cumsteps = _random_walk_drift(
        n_steps=10, diffusion_constant=(0, 0), velocity=(1, 10), seed=1
    )
    assert cumsteps.shape == (2, 10)


def test__drift():
    frames = np.arange(2, 10, 2)
    n_frames = len(frames)
    position_deltas = _drift(frames, diffusion_constant=(1, 10), velocity=None, seed=1)
    assert position_deltas.shape == (2, n_frames)
    position_deltas = _drift(frames, diffusion_constant=None, velocity=(1, 2), seed=1)
    assert position_deltas.shape == (2, n_frames)
    position_deltas = _drift(frames, diffusion_constant=None, velocity=None, seed=1)
    assert position_deltas is None


def test_simulate_drift(locdata_2d):
    new_locdata = add_drift(locdata_2d, diffusion_constant=None, velocity=None, seed=1)
    assert len(new_locdata) == len(locdata_2d)
    new_locdata = add_drift(
        locdata_2d, diffusion_constant=(1, 10), velocity=(10, 10), seed=1
    )
    assert len(new_locdata) == len(locdata_2d)
    # print(new_locdata.meta)


@pytest.mark.visual
def test_visual__drift():
    frames = np.arange(0, 1_000_000, dtype=int)
    print(frames.shape)

    # position_deltas = _drift(frames=frames, velocity=(1, 2), seed=1)
    position_deltas = _drift(frames=frames, diffusion_constant=(1, 2), seed=1)
    print(position_deltas.shape)
    print(position_deltas[0, :10])
    for pd_ in position_deltas:
        plt.plot(frames, pd_)
        plt.plot(frames, pd_)
    plt.show()


@pytest.mark.visual
def test_visual_add_drift(locdata_2d):
    # new_locdata = add_drift(locdata_2d, diffusion_constant=(1, 10), velocity=(10, 10), seed=1)
    new_locdata = add_drift(
        locdata_2d, diffusion_constant=None, velocity=(1, 1), seed=1
    )
    ax = locdata_2d.data.plot(*locdata_2d.coordinate_properties, kind="scatter")
    new_locdata.data.plot(*new_locdata.coordinate_keys, kind="scatter", ax=ax, c="r")
    plt.show()


def test_simulate_frame_numbers(locdata_2d):
    frames = simulate_frame_numbers(n_samples=(len(locdata_2d)), lam=2)
    assert len(frames) == len(locdata_2d)


def test_randomize_2d(locdata_2d):
    locdata = LocData()
    with pytest.raises(AttributeError):
        randomize(locdata, hull_region="bb")
    with pytest.raises(AttributeError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            randomize(locdata, hull_region="ch")
    with pytest.raises(AttributeError):
        randomize(locdata, hull_region="as")
    with pytest.raises(AttributeError):
        randomize(locdata, hull_region="something")

    locdata_2d = deepcopy(locdata_2d)

    locdata_randomized = randomize(locdata_2d, hull_region="bb")
    # locdata_randomized.print_meta()
    assert len(locdata_randomized) == 6
    # print(locdata_randomized.meta)
    assert locdata_randomized.meta.history[-1].name == "randomize"

    locdata_randomized = randomize(locdata_2d, hull_region="ch")
    assert len(locdata_randomized) == 6

    locdata_randomized = randomize(locdata_2d, hull_region="obb")
    assert len(locdata_randomized) == 6

    locdata_2d.update_alpha_shape(alpha=10)
    locdata_randomized = randomize(locdata_2d, hull_region="as")
    assert len(locdata_randomized) == 6

    region = Polygon(((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    locdata_randomized = randomize(locdata_2d, hull_region=region)
    assert len(locdata_randomized) == 6


def test_randomize_3d(locdata_3d):
    locdata_randomized = randomize(locdata_3d, hull_region="bb")
    assert len(locdata_randomized) == 6
    assert locdata_randomized.meta.history[-1].name == "randomize"

    # todo: implement make_csr in 3d
    with pytest.raises(NotImplementedError):
        locdata_randomized = randomize(locdata_3d, hull_region="ch")
        assert len(locdata_randomized) == 6

    # region_dict = dict(region='polygon', region_specs=((0, 0, 0), (0, 5, 0), (4, 3, 2), (2, 0.5, 2), (0, 0, 0)))
    # locdata_randomized = randomize(locdata_3d, hull_region=region_dict)
    # assert len(locdata_randomized) == 6


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        # ('locdata_empty', 0),
        # ('locdata_single_localization', 1),
        ("locdata_2d", 6),
        ("locdata_3d", 6),
        ("locdata_non_standard_index", 6),
    ],
)
def test_randomize_locdata_objects(
    locdata_empty,
    locdata_single_localization,
    locdata_2d,
    locdata_3d,
    locdata_non_standard_index,
    fixture_name,
    expected,
):
    locdata = eval(fixture_name)
    locdata_randomized = randomize(locdata, hull_region="bb")
    assert len(locdata_randomized) == expected
