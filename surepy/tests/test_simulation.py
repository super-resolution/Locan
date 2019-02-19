import pytest
import pandas as pd
import matplotlib.pyplot as plt
from surepy import LocData
from surepy.simulation import make_csr, simulate_csr, make_csr_on_disc, make_csr_on_region, make_spots, \
    simulate_csr_on_disc, simulate_csr_on_region, simulate_blobs, resample, simulate_tracks


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
    assert all(dat.data.columns == ['Position_x', 'Position_y'])
    dat = simulate_csr(n_samples=5, n_features=4, feature_range=(0, 1.), seed=None)
    assert all(dat.data.columns == ['Position_x', 'Position_y', 'Position_z', 'Feature_0'])


def test_make_csr_on_disc():
    samples = make_csr_on_disc(n_samples=10, radius=2.0, seed=None)
    assert len(samples) == 10
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_simulate_csr_on_disc():
    dat = simulate_csr_on_disc(n_samples=100, radius=1.0, seed=None)
    assert all(dat.data.columns == ['Position_x', 'Position_y'])


def test_make_spots():
    samples, labels = make_spots()
    assert len(samples) == 100
    assert max(labels) + 1 == 3
    samples, labels = make_spots(n_samples=100, n_features=2, centers=1,
                         radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    assert len(samples) == 100
    assert max(labels) + 1 == 1
    samples, labels = make_spots(n_samples=100, n_features=2, centers=5,
                         radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_spots(n_samples=100, n_features=2, centers=((0, 0), (0, 1)),
                         radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_spots(n_samples=(1, 2), n_features=2, centers=None,
                         radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_spots(n_samples=(1, 2, 3, 4, 5), n_features=2, centers=5,
                         radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_spots(n_samples=(1, 2), n_features=2, centers=((0, 0), (0, 1)),
                         radius=1.0, feature_range=(-10.0, 10.0), shuffle=True, seed=None)
    samples, labels = make_spots(n_samples=10, n_features=2, centers=None,
                         radius=1.0, feature_range=((0, 2), (-1, 0)), shuffle=True, seed=None)
    samples, labels = make_spots(n_samples=10, n_features=2, centers=((0, 1), (0, 1)),
                         radius=(1, 2), feature_range=(0, 1), shuffle=True, seed=None)
    samples, labels = make_spots(n_samples=10, n_features=2, centers=2,
                         radius=(1, 2), feature_range=(0, 1), shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_spots(n_samples=10, n_features=3, centers=None,
                         radius=1.0, feature_range=((0, 2), (-1, 0)), shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_spots(n_samples=100, n_features=3, centers=((0, 1), (0, 1)), radius=1.0, feature_range=(0, 1),
                         shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_spots(n_samples=(1,2,3), n_features=2, centers=((0, 1), (0, 1)),
                             radius=1.0, feature_range=(0, 1), shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_spots(n_samples=10, n_features=2, centers=((0, 1), (0, 1)),
                             radius=(1, 2, 3), feature_range=(0, 1), shuffle=True, seed=None)
    with pytest.raises(ValueError):
        samples, labels = make_spots(n_samples=10, n_features=2, centers=2,
                             radius=(1, 2, 3), feature_range=(0, 1), shuffle=True, seed=None)

    samples, labels = make_spots(n_samples=1000, n_features=2, centers=5,
                         radius=(1, 2, 3, 4, 5), feature_range=(-10.0, 10.0), shuffle=False, seed=None)
    assert len(samples) == len(labels)
    # print(labels)

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_make_csr_on_region():
    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    samples = make_csr_on_region(region_dict, n_samples=1000, seed=None)
    assert len(samples) == 1000
    # todo add tests for other regions and 3D case
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plt.scatter(samples[:,0], samples[:,1])
    # ax.axis('equal')
    # plt.show()


def test_simulate_csr_on_region():
    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    dat = simulate_csr_on_region(region_dict, n_samples=100, seed=None)
    assert all(dat.data.columns == ['Position_x', 'Position_y'])


##########################
def test_simulate_csr_():
    dat = simulate_csr(n_samples = 10, x_range=(0,10000), y_range=None, z_range=None, seed=None)
    assert(len(dat) == 10)
    assert(len(dat.coordinate_labels)==1)
    # dat.print_meta()

    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = (0,10000), z_range=None, seed=None)
    assert(len(dat) == 10)
    assert(len(dat.coordinate_labels)==2)

    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = (0,10000), z_range = (0,10000), seed=None)
    assert(len(dat) == 10)
    assert(len(dat.coordinate_labels)==3)


def test_simulate_blobs_1D():
    dat = simulate_blobs(n_centers=10, n_samples=100, n_features=1, center_box=(0, 10000), cluster_std=10, seed=None)
    assert (len(dat) == 100)
    assert(len(dat.coordinate_labels)==1)
    assert ('Position_x' in dat.data.columns)
    # dat.print_meta()

def test_simulate_blobs_2D():
    dat = simulate_blobs(n_centers=10, n_samples=100, n_features=2, center_box=(0, 10000), cluster_std=10, seed=None)
    assert (len(dat) == 100)
    assert(len(dat.coordinate_labels)==2)
    assert ('Position_y' in dat.data.columns)

def test_simulate_blobs_3D():
    dat = simulate_blobs(n_centers=10, n_samples=100, n_features=3, center_box=(0, 10000), cluster_std=10, seed=None)
    assert (len(dat) == 100)
    assert(len(dat.coordinate_labels)==3)
    assert ('Position_z' in dat.data.columns)


@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1],
        'Position_z': [0, 1, 3, 4, 1],
        'Intensity': [0, 1, 3, 4, 1],
        'Uncertainty_x': [10, 30, 100, 300, 0],
        'Uncertainty_y': [10, 30, 100, 300, 10],
        'Uncertainty_z': [10, 30, 100, 300, 10],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


def test_resample(locdata_simple):
    dat = resample(locdata=locdata_simple, number_samples=3)
    #print(dat.data)
    assert (len(dat)==15)


def test_simulate_tracks():
    dat = simulate_tracks(number_walks=2, number_steps=3)
    #print(dat.data)
    #print(dat.meta)
    assert (len(dat) == 6)
    assert(len(dat.coordinate_labels)==2)

