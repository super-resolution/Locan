import pytest
import pandas as pd

from surepy import LocData
from surepy.simulation import make_csr, simulate_csr, make_csr_on_disc, make_csr_on_region
from surepy.simulation import make_Matern, make_Thomas, make_Thomas_on_region
from surepy.simulation import simulate_csr_on_disc, simulate_csr_on_region
from surepy.simulation import simulate_Matern, simulate_Thomas, simulate_Thomas_on_region
from surepy.simulation import resample, simulate_tracks


def test_make_csr():
    points = make_csr(n_samples=5, n_features=1, feature_range=(0, 10), seed=None)
    assert points.shape == (5, 1)
    points = make_csr(n_samples=5, n_features=2, feature_range=(0, 10), seed=None)
    assert points.shape == (5, 2)
    points = make_csr(n_samples=5, n_features=2, feature_range=((-10, -5), (5, 10)), seed=None)
    assert points.shape == (5, 2)
    points = make_csr(n_samples=5, n_features=3, feature_range=((-10, -5), (5, 10), (100, 200)), seed=None)
    assert points.shape == (5, 3)


def test_simulate_csr():
    dat = simulate_csr(n_samples=5, n_features=2, feature_range=(0, 1.), seed=None)
    assert all(dat.data.columns == ['position_x', 'position_y'])
    dat = simulate_csr(n_samples=5, n_features=4, feature_range=(0, 1.), seed=None)
    assert all(dat.data.columns == ['position_x', 'position_y', 'position_z', 'feature_0'])


def test_make_csr_on_disc():
    samples = make_csr_on_disc(n_samples=10, radius=2.0, seed=None)
    assert len(samples) == 10
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_simulate_csr_on_disc():
    dat = simulate_csr_on_disc(n_samples=100, radius=1.0, seed=None)
    assert all(dat.data.columns == ['position_x', 'position_y'])


def test_make_Matern():
    samples, labels = make_Matern()
    assert len(samples) == 100
    assert max(labels) + 1 == 3
    samples, labels = make_Matern(n_samples=100, n_features=2, centers=1,
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    assert len(samples) == 100
    assert max(labels) + 1 == 1
    samples, labels = make_Matern(n_samples=100, n_features=2, centers=5,
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_Matern(n_samples=100, n_features=2, centers=((0, 0), (0, 1)),
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_Matern(n_samples=(1, 2), n_features=2, centers=None,
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_Matern(n_samples=(1, 2, 3, 4, 5), n_features=2, centers=5,
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_Matern(n_samples=(1, 2), n_features=2, centers=((0, 0), (0, 1)),
                                  radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_Matern(n_samples=10, n_features=2, centers=None,
                                  radius=1.0, feature_range=((0, 2), (-1, 0)), shuffle=True, seed=None)
    samples, labels = make_Matern(n_samples=10, n_features=2, centers=((0, 1), (0, 1)),
                                  radius=(1, 2), feature_range=(0, 1), shuffle=True, seed=None)
    samples, labels = make_Matern(n_samples=10, n_features=2, centers=2,
                                  radius=(1, 2), feature_range=(0, 1), shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=10, n_features=3, centers=None,
                                      radius=1.0, feature_range=((0, 2), (-1, 0)), shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=100, n_features=3, centers=((0, 1), (0, 1)), radius=1.0, feature_range=(0, 1),
                                      shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=(1, 2, 3), n_features=2, centers=((0, 1), (0, 1)),
                                      radius=1.0, feature_range=(0, 1), shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=10, n_features=2, centers=((0, 1), (0, 1)),
                                      radius=(1, 2, 3), feature_range=(0, 1), shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_Matern(n_samples=10, n_features=2, centers=2,
                                      radius=(1, 2, 3), feature_range=(0, 1), shuffle=True, seed=None)

    samples, labels = make_Matern(n_samples=1000, n_features=2, centers=5,
                                  radius=(1, 2, 3, 4, 5), feature_range=(-10.0, 10.0), shuffle=False, seed=None)
    assert len(samples) == len(labels)
    # print(labels)

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_simulate_Matern():
    dat = simulate_Matern(n_samples=100, n_features=2, centers=None, radius=1.0, feature_range=(-10.0, 10.0),
                          shuffle=True, seed=None)
    assert len(dat) == 100
    assert dat.meta.element_count == 100


def test_make_Thomas():
    samples, labels = make_Thomas(n_samples=10, n_features=2, centers=None, cluster_std=1.0,
                                  feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    assert len(samples) == 10
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas(n_samples=10, n_features=2, centers=5, cluster_std=10,
                                  feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    assert len(samples) == 10
    assert max(labels) + 1 == 5
    samples, labels = make_Thomas(n_samples=(1, 2, 3), n_features=2, centers=None, cluster_std=10,
                                  feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    assert len(samples) == 6
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas(n_samples=(1, 2, 3), n_features=2, centers=((1, 1), (10, 10), (100, 100)),
                                  cluster_std=(1, 10, 100),
                                  feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    assert len(samples) == 6
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas(n_samples=10, shuffle=False)
    assert len(samples) == 10


def test_simulate_Thomas():
    dat = simulate_Thomas(n_samples=100, n_features=2, centers=None, cluster_std=1.0, feature_range=(-10.0, 10.0),
                          shuffle=True, seed=None)
    assert len(dat) == 100
    assert dat.meta.element_count == 100


def test_make_csr_on_region():
    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    samples = make_csr_on_region(region_dict, n_samples=1000, seed=None)
    assert len(samples) == 1000
    samples = make_csr_on_region(region_dict, n_samples=2, seed=None)
    assert len(samples) == 2
    # todo add tests for other regions and 3D case
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_simulate_csr_on_region():
    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    dat = simulate_csr_on_region(region_dict, n_samples=100, seed=None)
    assert all(dat.data.columns == ['position_x', 'position_y'])


def test_make_Thomas_on_region():
    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    samples, labels = make_Thomas_on_region(region=region_dict)
    assert len(samples) == 100
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas_on_region(region=region_dict, n_samples=10, centers=None, cluster_std=1.0,
                                            shuffle=True, seed=None)
    assert len(samples) == 10
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas_on_region(region=region_dict, n_samples=10, centers=5, cluster_std=10,
                                            shuffle=True, seed=None)
    assert len(samples) == 10
    assert max(labels) + 1 == 5
    samples, labels = make_Thomas_on_region(region=region_dict, n_samples=(1, 2, 3), centers=None, cluster_std=10,
                                            shuffle=True, seed=None)
    assert len(samples) == 6
    assert max(labels) + 1 == 3
    samples, labels = make_Thomas_on_region(region=region_dict, n_samples=(1, 2, 3),
                                            centers=((1, 1), (10, 10), (100, 100)),
                                            cluster_std=(1, 10, 100),
                                            shuffle=True, seed=None)
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
    dat = simulate_Thomas_on_region(region_dict, n_samples=1000, centers=5, cluster_std=0.1, seed=None)
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
    dat = resample(locdata=locdata_simple, number_samples=3)
    # print(dat.data)
    assert len(dat) == 15

    # print(locdata_simple.meta)
    # print(dat.meta)


def test_simulate_tracks():
    dat = simulate_tracks(number_walks=2, number_steps=3)
    # print(dat.data)
    # print(dat.meta)
    assert len(dat) == 6
    assert len(dat.coordinate_labels) == 2
