import pytest
import pandas as pd
from surepy import LocData
from surepy.simulation import simulate_csr, simulate_blobs, resample, simulate_tracks


def test_simulate_csr():
    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = None, z_range = None, seed=None)
    assert(len(dat) == 10)
    assert(len(dat.coordinate_labels)==1)
    # dat.print_meta()

    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = (0,10000), z_range = None, seed=None)
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
    print(dat.data)
    print(dat.meta)
    assert (len(dat) == 6)
    assert(len(dat.coordinate_labels)==2)

