from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file
from surepy.data.transform import transform_affine
from surepy.analysis.cbc import _coordinate_based_colocalization
from surepy.analysis import CoordinateBasedColocalization


# fixtures

@pytest.fixture()
def locdata():
    path = Path(ROOT_DIR + '/tests/Test_data/five_blobs.txt')
    dat = load_txt_file(path)
    return dat


@pytest.fixture()
def locdata_3d():
    path = Path(ROOT_DIR + '/tests/Test_data/five_blobs_3D.txt')
    dat = load_txt_file(path)
    return dat


@pytest.fixture()
def locdata_line():
    locdata_dict = {
        'position_x': [0, 1, 3, 4, 50, 98, 99, 100],
        'position_y': np.zeros(8)
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


def test_cbc(locdata):
    points = locdata.coordinates
    points_trans = transform_affine(locdata.coordinates)
    res = _coordinate_based_colocalization(points=points, other_points=points_trans, radius=100, n_steps=10)
    assert(res[0] == 0.6162215398589203)
    # print(res[0:5])

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


def test_cbc_nan(locdata_line):
    res = _coordinate_based_colocalization(points=locdata_line.coordinates,
                                           other_points=locdata_line.coordinates+(0.5, 0),
                                           radius=10, n_steps=10)
    assert(len(res) == 8)
    assert(np.isnan(res[4]))


def test_Coordinate_based_colocalization(locdata_line):
    other_locdata = transform_affine(locdata_line)
    cbc = CoordinateBasedColocalization(radius=100, n_steps=10).compute(locdata=locdata_line,
                                                                        other_locdata=other_locdata)
    # print(cbc.results)
    assert(cbc.results.columns == f'colocalization_cbc_{other_locdata.meta.identifier}')

    cbc = CoordinateBasedColocalization(radius=100, n_steps=10).compute(locdata=locdata_line,
                                                                        other_locdata=other_locdata)
    print(other_locdata.meta)
    # assert(cbc.results.columns == 'colocalization_cbc_self')
