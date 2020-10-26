import pytest
import numpy as np
import pandas as pd

from surepy import LocData
from surepy.simulation import make_csr, simulate_csr, make_csr_on_disc, make_csr_on_region
from surepy.simulation import make_Matern, make_Thomas, make_Thomas_on_region
from surepy.simulation import simulate_csr_on_disc, simulate_csr_on_region
from surepy.simulation import simulate_Matern, simulate_Thomas, simulate_Thomas_on_region
from surepy.simulation import resample, simulate_tracks, add_drift, simulate_frame_numbers
from surepy.simulation.simulate_drift import _random_walk_drift, _drift


def test_make_csr():
    points = make_csr(n_samples=5, n_features=1, feature_range=(0, 10), seed=1)
    assert points.shape == (5, 1)
    points = make_csr(n_samples=5, n_features=2, feature_range=(0, 10), seed=1)
    assert points.shape == (5, 2)
    points = make_csr(n_samples=5, n_features=2, feature_range=((-10, -5), (5, 10)), seed=1)
    assert points.shape == (5, 2)
    points = make_csr(n_samples=5, n_features=3, feature_range=((-10, -5), (5, 10), (100, 200)), seed=1)
    assert points.shape == (5, 3)


def test_simulate_csr():
    dat = simulate_csr(n_samples=5, n_features=2, feature_range=(0, 1.), seed=1)
    assert all(dat.data.columns == ['position_x', 'position_y'])
    dat = simulate_csr(n_samples=5, n_features=4, feature_range=(0, 1.), seed=1)
    assert all(dat.data.columns == ['position_x', 'position_y', 'position_z', 'feature_0'])


def test_make_csr_on_disc():
    samples = make_csr_on_disc(n_samples=10, radius=2.0, seed=1)
    assert len(samples) == 10
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_simulate_csr_on_disc():
    dat = simulate_csr_on_disc(n_samples=100, radius=1.0, seed=1)
    assert all(dat.data.columns == ['position_x', 'position_y'])


def test_make_Matern():
    samples, labels = make_Matern()
    assert len(samples) == 100
    assert max(labels) + 1 == 3
    samples, labels = make_Matern(n_samples=100, n_features=2, centers=1,
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    assert len(samples) == 100
    assert max(labels) + 1 == 1
    samples, labels = make_Matern(n_samples=100, n_features=2, centers=5,
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    samples, labels = make_Matern(n_samples=100, n_features=2, centers=((0, 0), (0, 1)),
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    samples, labels = make_Matern(n_samples=(1, 2), n_features=2, centers=None,
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    samples, labels = make_Matern(n_samples=(1, 2, 3, 4, 5), n_features=2, centers=5,
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    samples, labels = make_Matern(n_samples=(1, 2), n_features=2, centers=((0, 0), (0, 1)),
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    samples, labels = make_Matern(n_samples=10, n_features=2, centers=None,
                                  radius=1.0, feature_range=((0, 2), (-1, 0)), shuffle=True, seed=1)
    samples, labels = make_Matern(n_samples=10, n_features=2, centers=((0, 1), (0, 1)),
                                  radius=(1, 2), feature_range=(0, 1), shuffle=True, seed=1)
    samples, labels = make_Matern(n_samples=10, n_features=2, centers=2,
                                  radius=(1, 2), feature_range=(0, 1), shuffle=True, seed=1)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=10, n_features=3, centers=None,
                                      radius=1.0, feature_range=((0, 2), (-1, 0)), shuffle=True, seed=1)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=100, n_features=3, centers=((0, 1), (0, 1)), radius=1.0, feature_range=(0, 1),
                                      shuffle=True, seed=1)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=(1, 2, 3), n_features=2, centers=((0, 1), (0, 1)),
                                      radius=1.0, feature_range=(0, 1), shuffle=True, seed=1)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=10, n_features=2, centers=((0, 1), (0, 1)),
                                      radius=(1, 2, 3), feature_range=(0, 1), shuffle=True, seed=1)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=10, n_features=2, centers=2,
                                      radius=(1, 2, 3), feature_range=(0, 1), shuffle=True, seed=1)

    samples, labels = make_Matern(n_samples=1000, n_features=2, centers=5,
                                  radius=(1, 2, 3, 4, 5), feature_range=(-10.0, 10.0), shuffle=False, seed=1)
    assert len(samples) == len(labels)
    # print(labels)

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_simulate_Matern():
    dat = simulate_Matern(n_samples=100, n_features=2, centers=None, radius=1.0, feature_range=(-10.0, 10.0),
                          shuffle=True, seed=1)
    assert len(dat) == 100
    assert dat.meta.element_count == 100


def test_make_Thomas():
    samples, labels = make_Thomas(n_samples=10, n_features=2, centers=None, cluster_std=1.0,
                                  feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    assert len(samples) == 10
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas(n_samples=10, n_features=2, centers=5, cluster_std=10,
                                  feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    assert len(samples) == 10
    assert max(labels) + 1 == 5
    samples, labels = make_Thomas(n_samples=(1, 2, 3), n_features=2, centers=None, cluster_std=10,
                                  feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    assert len(samples) == 6
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas(n_samples=(1, 2, 3), n_features=2, centers=((1, 1), (10, 10), (100, 100)),
                                  cluster_std=(1, 10, 100),
                                  feature_range=(-10.0, 10.0), shuffle=True, seed=1)
    assert len(samples) == 6
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas(n_samples=10, shuffle=False)
    assert len(samples) == 10


def test_simulate_Thomas():
    dat = simulate_Thomas(n_samples=100, n_features=2, centers=None, cluster_std=1.0, feature_range=(-10.0, 10.0),
                          shuffle=True, seed=1)
    assert len(dat) == 100
    assert dat.meta.element_count == 100


def test_make_csr_on_region():
    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    samples = make_csr_on_region(region_dict, n_samples=1000, seed=1)
    assert len(samples) == 1000
    samples = make_csr_on_region(region_dict, n_samples=2, seed=1)
    assert len(samples) == 2
    # todo add tests for other regions and 3D case
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_simulate_csr_on_region():
    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    dat = simulate_csr_on_region(region_dict, n_samples=100, seed=1)
    assert all(dat.data.columns == ['position_x', 'position_y'])


def test_make_Thomas_on_region():
    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    samples, labels = make_Thomas_on_region(region=region_dict)
    assert len(samples) == 100
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas_on_region(region=region_dict, n_samples=10, centers=None, cluster_std=1.0,
                                            shuffle=True, seed=1)
    assert len(samples) == 10
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas_on_region(region=region_dict, n_samples=10, centers=5, cluster_std=10,
                                            shuffle=True, seed=1)
    assert len(samples) == 10
    assert max(labels) + 1 == 5
    samples, labels = make_Thomas_on_region(region=region_dict, n_samples=(1, 2, 3), centers=None, cluster_std=10,
                                            shuffle=True, seed=1)
    assert len(samples) == 6
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas_on_region(region=region_dict, n_samples=(1, 2, 3),
                                            centers=((1, 1), (10, 10), (100, 100)),
                                            cluster_std=(1, 10, 100),
                                            shuffle=True, seed=1)
    assert len(samples) == 6
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas_on_region(region=region_dict, n_samples=10, shuffle=False)
    assert len(samples) == 10

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_simulate_Thomas_on_region():
    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    dat = simulate_Thomas_on_region(region_dict, n_samples=1000, centers=5, cluster_std=0.1, seed=1)
    assert all(dat.data.columns == ['position_x', 'position_y'])

    # dat.data.plot(x='Position_x', y='Position_y', kind='scatter')
    # plt.show()


@pytest.fixture()
def locdata_simple():
    localization_dict = {
        'position_x': [0, 0, 1, 4, 5],
        'position_y': [0, 1, 3, 4, 1],
        'position_z': [0, 1, 3, 4, 1],
        'intensity': [0, 1, 3, 4, 1],
        'uncertainty_x': [10, 30, 100, 300, 0],
        'uncertainty_y': [10, 30, 100, 300, 10],
        'uncertainty_z': [10, 30, 100, 300, 10],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(localization_dict))


def test_resample(locdata_simple):
    dat = resample(locdata=locdata_simple, n_samples=3)
    # print(dat.data)
    assert len(dat) == 15

    # print(locdata_simple.meta)
    # print(dat.meta)


def test_simulate_tracks():
    dat = simulate_tracks(n_walks=2, n_steps=3)
    # print(dat.data)
    # print(dat.meta)
    assert len(dat) == 6
    assert len(dat.coordinate_labels) == 2


def test__random_walk_drift():
    cumsteps = _random_walk_drift(n_steps=10, diffusion_constant=(1, 10), velocity=(0, 0), seed=1)
    assert cumsteps.shape == (2, 10)
    cumsteps = _random_walk_drift(n_steps=10, diffusion_constant=(0, 0), velocity=(1, 10), seed=1)
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
    new_locdata = add_drift(locdata_2d, diffusion_constant=(1, 10), velocity=(10, 10), seed=1)
    assert len(new_locdata) == len(locdata_2d)
    # print(new_locdata.meta)


@pytest.mark.skip('Test needs visual inspection.')
def test_visual__drift():
    import matplotlib.pyplot as plt

    frames = np.arange(0, 1_000_000, dtype=int)
    print(frames.shape)

    # position_deltas = _drift(frames=frames, velocity=(1, 2), seed=1)
    position_deltas = _drift(frames=frames, diffusion_constant=(1, 2), seed=1)
    print(position_deltas.shape)
    print(position_deltas[0, :10])
    for pd in position_deltas:
        plt.plot(frames, pd)
        plt.plot(frames, pd)
    plt.show()


@pytest.mark.skip('Test needs visual inspection.')
def test_visual_add_drift(locdata_2d):
    import matplotlib.pyplot as plt
    # new_locdata = add_drift(locdata_2d, diffusion_constant=(1, 10), velocity=(10, 10), seed=1)
    new_locdata = add_drift(locdata_2d, diffusion_constant=None, velocity=(1, 1), seed=1)
    ax = locdata_2d.data.plot(*locdata_2d.coordinate_labels, kind='scatter')
    new_locdata.data.plot(*new_locdata.coordinate_labels, kind='scatter', ax=ax, c='r')
    plt.show()

def test_simulate_frame_numbers(locdata_2d):
    frames = simulate_frame_numbers(n_samples=(len(locdata_2d)), lam=2)
    assert len(frames) == len(locdata_2d)
    locdata_2d.dataframe = locdata_2d.dataframe.assign(frame=frames)
    assert np.array_equal(locdata_2d.data.frame, frames)
