from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file
from surepy.data.transform import transform_affine
from surepy.data.properties import coordinate_based_colocalization

# fixtures

@pytest.fixture()
def locdata():
    path = Path(ROOT_DIR + '/tests/Test_data/five_blobs.txt')
    dat = load_txt_file(path)
    return dat

@pytest.fixture()
def locdata_3D():
    path = Path(ROOT_DIR + '/tests/Test_data/five_blobs_3D.txt')
    dat = load_txt_file(path)
    return dat

def test_cbc(locdata):
    points = locdata.coordinates
    points_trans = transform_affine(locdata.coordinates)
    res = coordinate_based_colocalization(points_1=points, points_2=points_trans, radius=100, n_steps=10)
    assert(res[0]==0.6162215398589203)
    #print(res[0:5])

    # # plot data
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # axes[0].scatter(x=points[:,0], y=points[:,1], marker='.', c=res, cmap='jet', label='points')
    # axes[0].set_title('CBC')
    #
    # axes[1].scatter(x=points[:,0], y=points[:,1], marker='.', color='Blue', label='points')
    # axes[1].scatter(x=points_trans[:,0], y=points_trans[:,1], marker='o', color='Red', label='transformed points')
    # axes[1].set_title('Transformed points')
    #
    # plt.show()
