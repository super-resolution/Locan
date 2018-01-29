import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import surepy.constants
import surepy.io.io_locdata as io
from surepy.render import render2D
import surepy.tests.test_data

# fixtures

@pytest.fixture()
def locdata():
    dat = io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/someData.txt', nrows=10)
    return dat

# tests

def test_simple_rendering_2D (locdata):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    cax = render2D(locdata, ax=ax, bin_size=100, range=[[100,20000],[100,20000]], rescale=(0, 100))
    cax = render2D(locdata, ax=ax, bin_size=100, range='auto', rescale=(0, 100))
    cax = render2D(locdata, ax=ax, bin_size=100, range='zero', rescale=(0, 100))
    cax = render2D(locdata, ax=ax, bin_size=100, range='auto', rescale='equal')
    plt.colorbar(cax)
    # plt.show()

