import warnings
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data


def test_get_correct_column_names_from_rapidSTORM_header():
    columns = io.load_rapidSTORM_header(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt')
    assert (columns == ['Position_x', 'Position_y', 'Frame', 'Intensity', 'Chi_square', 'Local_background'])

def test_loading_rapidSTORM_file():
    dat = io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt', nrows=10)
    #print(dat.data.head())
    #dat.print_meta()
    assert (len(dat) == 10)

def test_get_correct_column_names_from_Elyra_header():
    columns = io.load_Elyra_header(path=surepy.constants.ROOT_DIR + '/tests/test_data/Elyra_dstorm_data.txt')
    assert (columns == ['Original_index', 'Frame', 'Frames_number', 'Frames_missing', 'Position_x', 'Position_y',
                        'Precision', 'Intensity', 'Local_background', 'Chi_square', 'Psf_half_width', 'Channel',
                        'Slice_z'])

def test_loading_Elyra_file():
    dat = io.load_Elyra_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/Elyra_dstorm_data.txt', nrows=10)
    assert (len(dat) == 10)


def test_get_correct_column_names_from_Thunderstorm_header():
    columns = io.load_thunderstorm_header(path=surepy.constants.ROOT_DIR + '/tests/test_data/Thunderstorm_dstorm_data.csv')
    assert (columns == ['Original_index', 'Frame', 'Position_x', 'Position_y', 'Psf_sigma_x', 'Intensity', 'Local_background',
     'bkgstd [photon]', 'Chi_square', 'uncertainty [nm]'])

def test_loading_Thunderstorm_file():
    dat = io.load_thunderstorm_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/Thunderstorm_dstorm_data.csv',
                                    nrows=10)
    #print(dat.data.columns)
    assert (len(dat) == 10)

def test_loading_txt_file():
    dat = io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs.txt', nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)

    dat = io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs.txt',
                           columns=['Index', 'Position_x', 'Position_y', 'Cluster_label'], nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)

    with pytest.warns(UserWarning):
        dat = io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs.txt', columns=['c1'], nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)

def test_save_asdf(locdata_fix):
    io.save_asdf(locdata_fix, path=surepy.constants.ROOT_DIR + '/tests/test_data/locdata.asdf')

def test_load_asdf_file(locdata_fix):
    locdata = io.load_asdf_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/locdata.asdf')
    # print(locdata.data)
    assert_frame_equal(locdata.data, locdata_fix.data)
    assert(locdata.meta.identifier == locdata_fix.meta.identifier)
    assert(locdata.properties == locdata_fix.properties)

    dat = io.load_asdf_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/locdata.asdf', nrows=5)
    assert (len(dat) == 5)


def test__map_file_type_to_load_function():
    file_type = io._map_file_type_to_load_function(type=1)
    # print(file_type)


def test_load_locdata():
    dat = io.load_locdata(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt',
                          type='RAPIDSTORM',
                          nrows=10)
    assert (len(dat) == 10)

    dat = io.load_locdata(path=surepy.constants.ROOT_DIR + '/tests/test_data/Elyra_dstorm_data.txt',
                          type='ELYRA',
                          nrows=10)
    assert (len(dat) == 10)

    dat = io.load_locdata(path=surepy.constants.ROOT_DIR + '/tests/test_data/Elyra_dstorm_data.txt',
                          type='elyra',
                          nrows=10)
    assert (len(dat) == 10)

    dat = io.load_locdata(path=surepy.constants.ROOT_DIR + '/tests/test_data/Elyra_dstorm_data.txt',
                          type=3,
                          nrows=10)
    assert (len(dat) == 10)

