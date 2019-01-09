import pytest
import numpy as np
import pandas as pd
from surepy import LocData
import surepy.constants
from surepy.io.io_locdata import load_rapidSTORM_file
from surepy.data.transform import randomize
from surepy.data.transform.bunwarpj import _read_matrix, bunwarp

from surepy.render import render2D


@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))

def test_randomize(locdata_simple):
    locdata_randomized = randomize(locdata_simple, hull_region='bb')
    #locdata_randomized.print_meta()
    assert (len(locdata_randomized) == len(locdata_simple))

# todo: test colocalization of bead data
def test_bunwarp_raw_transformation():
    matrix_path = surepy.constants.ROOT_DIR + '/tests/test_data/transform/BunwarpJ_transformation_raw_green.txt'
    dat_green = load_rapidSTORM_file(path=surepy.constants.ROOT_DIR +
                                          '/tests/test_data/transform/rapidSTORM_beads_green.txt')
    dat_red = load_rapidSTORM_file(path=surepy.constants.ROOT_DIR +
                                        '/tests/test_data/transform/rapidSTORM_beads_red.txt')

    image_height_width, x_transformation_array, y_transformation_array = _read_matrix(path=matrix_path)
    assert(all(image_height_width==[130, 130]))

    dat_green_transformed = bunwarp(locdata=dat_green, matrix_path=matrix_path)

    # render2D(dat_green)
