import pytest
import numpy as np
import matplotlib.pyplot as plt

import locan.constants
from locan.constants import _has_open3d
from locan.data.region import Polygon
from locan.io.locdata.io_locdata import load_rapidSTORM_file
from locan.data.transform import randomize, transform_affine
from locan.data.transform.transformation import _homogeneous_matrix
from locan.data.transform.bunwarpj import _read_matrix, _unwarp, bunwarp


def test_randomize_2d(locdata_2d):
    locdata_randomized = randomize(locdata_2d, hull_region='bb')
    # locdata_randomized.print_meta()
    assert len(locdata_randomized) == 6
    # print(locdata_randomized.meta)
    assert locdata_randomized.meta.history[-1].name == 'randomize'

    locdata_randomized = randomize(locdata_2d, hull_region='ch')
    assert len(locdata_randomized) == 6

    region = Polygon(((0, 0), (0, 5), (4, 3), (2, 0.5), (0, 0)))
    locdata_randomized = randomize(locdata_2d, hull_region=region)
    assert len(locdata_randomized) == 6


def test_randomize_3d(locdata_3d):
    locdata_randomized = randomize(locdata_3d, hull_region='bb')
    assert len(locdata_randomized) == 6
    assert locdata_randomized.meta.history[-1].name == 'randomize'

    # todo: implement make_csr in 3d
    with pytest.raises(NotImplementedError):
        locdata_randomized = randomize(locdata_3d, hull_region='ch')
        assert len(locdata_randomized) == 6

    # region_dict = dict(region='polygon', region_specs=((0, 0, 0), (0, 5, 0), (4, 3, 2), (2, 0.5, 2), (0, 0, 0)))
    # locdata_randomized = randomize(locdata_3d, hull_region=region_dict)
    # assert len(locdata_randomized) == 6


@pytest.mark.parametrize('fixture_name, expected', [
    # ('locdata_empty', 0),
    # ('locdata_single_localization', 1),
    ('locdata_2d', 6),
    ('locdata_3d', 6),
    ('locdata_non_standard_index', 6)
])
def test_randomize_locdata_objects(
        locdata_empty, locdata_single_localization, locdata_2d, locdata_3d, locdata_non_standard_index,
        fixture_name, expected):
    locdata = eval(fixture_name)
    locdata_randomized = randomize(locdata, hull_region='bb')
    assert len(locdata_randomized) == expected


def test_bunwarp_raw_transformation():
    matrix_path = locan.constants.ROOT_DIR / 'tests/test_data/transform/BunwarpJ_transformation_raw_green.txt'
    dat_green = load_rapidSTORM_file(path=locan.constants.ROOT_DIR /
                                     'tests/test_data/transform/rapidSTORM_beads_green.txt')

    matrix_size, matrix_x, matrix_y = _read_matrix(path=matrix_path)
    assert np.array_equal(matrix_size, [130, 130])

    new_points = _unwarp(dat_green.coordinates, matrix_x, matrix_y, pixel_size=(10, 10), matrix_size=matrix_size)
    assert len(new_points) == len(dat_green)

    dat_green_transformed = bunwarp(locdata=dat_green, matrix_path=matrix_path, pixel_size=(10, 10))
    assert len(dat_green_transformed) == len(dat_green)
    assert dat_green_transformed.meta.history[-1].name == 'bunwarp'

    # for visual inspection
    # dat_red = load_rapidSTORM_file(path=locan.constants.ROOT_DIR /
    #                                     'tests/test_data/transform/rapidSTORM_beads_red.txt')
    # fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    # render_2d(dat_red, ax=ax, bin_size=500, rescale=True, cmap='Reds')
    # render_2d(dat_green, ax=ax, bin_size=500, rescale=True, cmap='Greens', alpha=0.5)
    # render_2d(dat_green_transformed, ax=ax, bin_size=500, rescale=True, cmap='Blues', alpha=0.5)
    # plt.show()

    plt.close('all')


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


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', 0),
    ('locdata_single_localization', 4),
    ('locdata_2d', 4),
    ('locdata_3d', 5),
    ('locdata_non_standard_index', 4)
])
def test_standard_locdata_objects(
        locdata_empty, locdata_single_localization, locdata_2d, locdata_3d, locdata_non_standard_index,
        fixture_name, expected):
    locdata = eval(fixture_name)
    new_locdata = transform_affine(locdata)
    assert len(new_locdata.data.columns) == expected


@pytest.mark.skipif(not _has_open3d, reason="Test requires open3d.")
@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', 0),
    ('locdata_single_localization', 4),
    ('locdata_2d', 4),
    ('locdata_3d', 5),
    ('locdata_non_standard_index', 4)
])
def test_standard_locdata_objects_open3d(
        locdata_empty, locdata_single_localization, locdata_2d, locdata_3d, locdata_non_standard_index,
        fixture_name, expected):
    locdata = eval(fixture_name)
    new_locdata = transform_affine(locdata, method='open3d')
    assert len(new_locdata.data.columns) == expected


def test_transformation_affine_2d(locdata_2d):
    new_locdata = transform_affine(locdata_2d)
    assert np.array_equal(new_locdata.coordinates, locdata_2d.coordinates)
    assert len(new_locdata.data.columns) == 4
    assert new_locdata.meta.history[-1].name == 'transform_affine'
    assert "'matrix': None, 'offset': None, 'pre_translation': None, 'method': 'numpy'" \
           in new_locdata.meta.history[-1].parameter

    matrix = ((-1, 0), (0, -1))
    offset = (10, 10)
    pre_translation = (100, 100)

    new_locdata = transform_affine(locdata_2d, matrix, offset)
    points_target = ((9, 9), (9, 5), (8, 7), (7, 4), (6, 8), (5, 5))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 4

    new_locdata = transform_affine(locdata_2d, offset=offset, pre_translation=pre_translation)
    points_target = ((11, 11), (11, 15), (12, 13), (13, 16), (14, 12), (15, 15))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 4


@pytest.mark.skipif(not _has_open3d, reason="Test requires open3d.")
def test_transformation_affine_2d_open3d(locdata_2d):
    new_locdata = transform_affine(locdata_2d, method='open3d')
    assert np.array_equal(new_locdata.coordinates, locdata_2d.coordinates)
    assert len(new_locdata.data.columns) == 4

    matrix = ((-1, 0), (0, -1))
    offset = (10, 10)
    pre_translation = (100, 100)

    new_locdata = transform_affine(locdata_2d, matrix, offset,  method='open3d')
    points_target = ((9, 9), (9, 5), (8, 7), (7, 4), (6, 8), (5, 5))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 4

    new_locdata = transform_affine(locdata_2d,  offset=offset, pre_translation=pre_translation,  method='open3d')
    points_target = ((11, 11), (11, 15), (12, 13), (13, 16), (14, 12), (15, 15))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 4


def test_transformation_affine_3d(locdata_3d):
    new_locdata = transform_affine(locdata_3d)
    assert np.array_equal(new_locdata.coordinates, locdata_3d.coordinates)
    assert len(new_locdata.data.columns) == 5
    assert new_locdata.meta.history[-1].name == 'transform_affine'
    assert "'matrix': None, 'offset': None, 'pre_translation': None, 'method': 'numpy'" \
           in new_locdata.meta.history[-1].parameter

    matrix = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
    offset = (10, 10, 10)
    pre_translation = (100, 100, 100)

    new_locdata = transform_affine(locdata_3d, matrix, offset)
    points_target = ((9, 9, 9), (9, 5, 8), (8, 7, 5), (7, 4, 6), (6, 8, 7), (5, 5, 8))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 5

    new_locdata = transform_affine(locdata_3d, offset=offset, pre_translation=pre_translation)
    points_target = ((11, 11, 11), (11, 15, 12), (12, 13, 15), (13, 16, 14), (14, 12, 13), (15, 15, 12))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 5


@pytest.mark.skipif(not _has_open3d, reason="Test requires open3d.")
def test_transformation_affine_3d_open3d(locdata_3d):
    new_locdata = transform_affine(locdata_3d, method='open3d')
    assert np.array_equal(new_locdata.coordinates, locdata_3d.coordinates)
    assert len(new_locdata.data.columns) == 5

    matrix = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
    offset = (10, 10, 10)
    pre_translation = (100, 100, 100)

    new_locdata = transform_affine(locdata_3d, matrix, offset,  method='open3d')
    points_target = ((9, 9, 9), (9, 5, 8), (8, 7, 5), (7, 4, 6), (6, 8, 7), (5, 5, 8))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 5

    new_locdata = transform_affine(locdata_3d,  offset=offset, pre_translation=pre_translation,  method='open3d')
    points_target = ((11, 11, 11), (11, 15, 12), (12, 13, 15), (13, 16, 14), (14, 12, 13), (15, 15, 12))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 5
