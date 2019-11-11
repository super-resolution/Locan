import pytest
import numpy as np
import pandas as pd
from surepy import LocData
import surepy.constants
from surepy.io.io_locdata import load_rapidSTORM_file
from surepy.data.transform import randomize, transform_affine
from surepy.data.transform.transformation import _homogeneous_matrix
from surepy.data.transform.bunwarpj import _read_matrix, bunwarp

from surepy.render import render_2d


@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        'position_x': [0, 0, 1, 4, 5],
        'position_y': [0, 1, 3, 4, 1]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


def test_randomize(locdata_simple):
    locdata_randomized = randomize(locdata_simple, hull_region='bb')
    # locdata_randomized.print_meta()
    assert len(locdata_randomized) == len(locdata_simple)
    # print(locdata_randomized.meta)
    assert locdata_randomized.meta.history[-1].name == 'randomize'

    locdata_randomized = randomize(locdata_simple, hull_region='ch')
    assert len(locdata_randomized) == len(locdata_simple)

    region_dict = dict(region_type='polygon', region_specs=((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    locdata_randomized = randomize(locdata_simple, hull_region=region_dict)
    assert len(locdata_randomized) == 5


# todo: test colocalization of bead data
def test_bunwarp_raw_transformation():
    matrix_path = surepy.constants.ROOT_DIR / 'tests/test_data/transform/BunwarpJ_transformation_raw_green.txt'
    dat_green = load_rapidSTORM_file(path=surepy.constants.ROOT_DIR /
                                     'tests/test_data/transform/rapidSTORM_beads_green.txt')

    image_height_width, x_transformation_array, y_transformation_array = _read_matrix(path=matrix_path)
    assert all(image_height_width == [130, 130])

    dat_green_transformed = bunwarp(locdata=dat_green, matrix_path=matrix_path)
    # print(dat_green_transformed.meta)
    assert dat_green_transformed.meta.history[-1].name == 'bunwarp'

    # dat_red = load_rapidSTORM_file(path=surepy.constants.ROOT_DIR +
    #                                     '/tests/test_data/transform/rapidSTORM_beads_red.txt')
    # render_2d(dat_green)


def test_transform_affine(locdata_simple):
    # with points as input
    new_points = transform_affine(locdata_simple.coordinates, matrix=((0, 1), (-1, 0)), offset=(10, 10))
    assert np.all(new_points == [[10, 10], [11, 10], [13, 9], [14, 6], [11, 5]])

    # with locdata as input
    new_locdata = transform_affine(locdata_simple, matrix=((0, 1), (-1, 0)), offset=(10, 10))
    assert np.all(new_locdata.coordinates == [[10, 10], [11, 10], [13, 9], [14, 6], [11, 5]])
    # print(new_locdata.data)
    # print(new_locdata.meta)
    assert new_locdata.meta.history[-1].name == 'transform_affine'


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', 0),
    ('locdata_single_localization', 4),
    ('locdata_fix', 4),
    ('locdata_non_standard_index', 4)
])
def test_standard_locdata_objects(
        locdata_empty, locdata_single_localization, locdata_fix, locdata_non_standard_index,
        fixture_name, expected):
    locdata = eval(fixture_name)
    new_locdata = transform_affine(locdata, matrix=((0, 1), (-1, 0)), offset=(10, 10))
    assert len(new_locdata.data.columns) == expected


def test_homogeneous_matrix():
    matrix_out = _homogeneous_matrix()
    result = np.identity(4)
    assert np.array_equal(matrix_out, result)

    matrix = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    offset = (10, 20, 30)
    matrix_out = _homogeneous_matrix(matrix, offset)
    result = np.array(((1, 2, 3, 10), (4, 5, 6, 20), (7, 8, 9, 30), (0, 0, 0, 1)))
    assert np.array_equal(matrix_out, result)

    matrix = ((1, 2), (3, 4))
    offset = (10, 20)
    matrix_out = _homogeneous_matrix(matrix, offset)
    result = np.array(((1, 2, 10), (3, 4, 20), (0, 0, 1)))
    assert np.array_equal(matrix_out, result)
