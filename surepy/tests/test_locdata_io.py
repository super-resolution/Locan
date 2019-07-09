import pytest
from pandas.testing import assert_frame_equal
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.data import metadata_pb2


def test_get_correct_column_names_from_rapidSTORM_header():
    columns = io.load_rapidSTORM_header(path=surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt')
    assert columns == ['position_x', 'position_y', 'frame', 'intensity', 'chi_square', 'local_background']


def test_loading_rapidSTORM_file():
    dat = io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
                                  nrows=10)
    #print(dat.data.head())
    #dat.print_meta()
    assert (len(dat) == 10)


def test_get_correct_column_names_from_Elyra_header():
    columns = io.load_Elyra_header(path=surepy.constants.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt')
    assert (columns == ['original_index', 'frame', 'frames_number', 'frames_missing', 'position_x', 'position_y',
                        'precision', 'intensity', 'local_background', 'chi_square', 'psf_half_width', 'channel',
                        'slice_z'])


def test_loading_Elyra_file():
    dat = io.load_Elyra_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt', nrows=10)
    assert (len(dat) == 10)


def test_get_correct_column_names_from_Thunderstorm_header():
    with pytest.warns(UserWarning):
        columns = io.load_thunderstorm_header(
            path=surepy.constants.ROOT_DIR / 'tests/test_data/Thunderstorm_dstorm_data.csv')
    assert (columns == ['original_index', 'frame', 'position_x', 'position_y', 'psf_sigma_x', 'intensity',
                        'local_background', 'bkgstd [photon]', 'chi_square', 'uncertainty [nm]'])


def test_loading_Thunderstorm_file():
    with pytest.warns(UserWarning):
        dat = io.load_thunderstorm_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/Thunderstorm_dstorm_data.csv',
                                    nrows=10)
    #print(dat.data.columns)
    assert (len(dat) == 10)


def test_loading_txt_file():
    dat = io.load_txt_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/five_blobs.txt', nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)

    dat = io.load_txt_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/five_blobs.txt',
                           columns=['index', 'position_x', 'position_y', 'cluster_label'], nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)

    with pytest.warns(UserWarning):
        dat = io.load_txt_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/five_blobs.txt',
                               columns=['c1'], nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)


def test_save_asdf(locdata_fix):
    io.save_asdf(locdata_fix, path=surepy.constants.ROOT_DIR / 'tests/test_data/locdata.asdf')


def test_load_asdf_file(locdata_fix):
    locdata = io.load_asdf_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/locdata.asdf')
    # print(locdata.data)
    assert_frame_equal(locdata.data, locdata_fix.data)
    assert(locdata.meta.identifier == locdata_fix.meta.identifier)
    assert(locdata.properties == locdata_fix.properties)

    dat = io.load_asdf_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/locdata.asdf', nrows=5)
    assert (len(dat) == 5)


def test__map_file_type_to_load_function():
    file_type = io._map_file_type_to_load_function(file_type=1)
    assert callable(file_type)
    file_type = io._map_file_type_to_load_function(file_type='RAPIDSTORM')
    assert callable(file_type)
    file_type = io._map_file_type_to_load_function(file_type=surepy.constants.File_type.RAPIDSTORM)
    assert callable(file_type)
    file_type = io._map_file_type_to_load_function(file_type=metadata_pb2.RAPIDSTORM)
    assert callable(file_type)


def test_load_locdata():
    dat = io.load_locdata(path=surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
                          file_type='RAPIDSTORM',
                          nrows=10)
    assert (len(dat) == 10)

    dat = io.load_locdata(path=surepy.constants.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt',
                          file_type='ELYRA',
                          nrows=10)
    assert (len(dat) == 10)

    dat = io.load_locdata(path=surepy.constants.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt',
                          file_type='elyra',
                          nrows=10)
    assert (len(dat) == 10)

    dat = io.load_locdata(path=surepy.constants.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt',
                          file_type=3,
                          nrows=10)
    assert (len(dat) == 10)
