import pytest
import matplotlib.pyplot as plt

from surepy.simulation import simulate_Thomas
from surepy.render import render_2d
import surepy.tests.test_data


# fixtures

@pytest.fixture()
def locdata():
    # dat = io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt',
    #                               nrows=10)
    dat = simulate_Thomas(n_samples=1000, n_features=2, centers=10, feature_range=(0, 1000), cluster_std=10, seed=0)
    return dat


# tests

def test_simple_rendering_2D(locdata):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    cax = render_2d(locdata, ax=ax, bin_size=100, range=[[500, 1000], [500, 1000]], show=False)
    cax = render_2d(locdata, ax=ax, bin_size=100, range='auto', rescale=(0, 100), show=False)
    cax = render_2d(locdata, ax=ax, bin_size=100, range='zero', rescale=(0, 100), show=False)
    cax = render_2d(locdata, ax=ax, bin_size=100, range='auto', rescale='equal', show=False)
    cax = render_2d(locdata, ax=ax, bin_size=100, range='auto', rescale=None, show=False)
    plt.colorbar(cax)
    # plt.show()
