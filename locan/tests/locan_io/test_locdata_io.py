from pathlib import Path
import tempfile
import pickle
from io import StringIO
import logging

from pandas.testing import assert_frame_equal
import locan.constants
from locan.data import metadata_pb2
from locan.locan_io import save_asdf, load_asdf_file, load_txt_file, load_thunderstorm_file,\
    load_Elyra_file, load_Nanoimager_file, \
    load_locdata

from locan.locan_io.locdata.io_locdata import _map_file_type_to_load_function
from locan.locan_io.locdata.io_locdata import load_Elyra_header
from locan.locan_io.locdata.io_locdata import load_thunderstorm_header
from locan.locan_io.locdata.io_locdata import load_Nanoimager_header


def test_get_correct_column_names_from_Elyra_header():
    columns = load_Elyra_header(path=locan.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt')
    assert (columns == ['original_index', 'frame', 'frames_number', 'frames_missing', 'position_x', 'position_y',
                        'uncertainty', 'intensity', 'local_background_sigma', 'chi_square', 'psf_half_width', 'channel',
                        'slice_z'])

    file_like = StringIO("Index	First Frame	Number Frames	Frames Missing	Position X [nm]	Position Y [nm]\n"
                         "1  1   1   0   15850.6 23502.1")
    columns = load_Elyra_header(path=file_like)
    assert (columns == ['original_index', 'frame', 'frames_number', 'frames_missing', 'position_x', 'position_y'])


def test_loading_Elyra_file():
    dat = load_Elyra_file(path=locan.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt')
    # loading is not limited by nrows=10 to ensure correct treatment of file appendix and NUL character.
    assert (len(dat) == 999)

    file_like = StringIO("Index\tFirst Frame\tNumber Frames\tFrames Missing\tPosition X [nm]\tPosition Y [nm]\n"
                         "1\t1\t1\t0\t15850.6\t23502.1")
    dat = load_Elyra_file(path=file_like)
    # loading is not limited by nrows=10 to ensure correct treatment of file appendix and NUL character.
    assert (len(dat) == 1)


def test_get_correct_column_names_from_Thunderstorm_header():
    columns = load_thunderstorm_header(
        path=locan.ROOT_DIR / 'tests/test_data/Thunderstorm_dstorm_data.csv')
    assert (columns == ['original_index', 'frame', 'position_x', 'position_y', 'psf_sigma', 'intensity',
                        'local_background', 'local_background_sigma', 'chi_square', 'uncertainty'])

    file_like = StringIO("id,frame,x [nm],y [nm]\n"
                         "73897.0,2001.0,1320.109670647555,26344.7124618434")
    columns = load_thunderstorm_header(path=file_like)
    assert (columns == ['original_index', 'frame', 'position_x', 'position_y'])


def test_loading_Thunderstorm_file():
    dat = load_thunderstorm_file(path=locan.ROOT_DIR / 'tests/test_data/Thunderstorm_dstorm_data.csv',
                                    nrows=10)
    # print(dat.data.columns)
    assert (len(dat) == 10)

    file_like = StringIO("id,frame,x [nm],y [nm]\n"
                         "73897.0,2001.0,1320.109670647555,26344.7124618434")
    dat = load_thunderstorm_file(path=file_like, nrows=1)
    assert (len(dat) == 1)


def test_get_correct_column_names_from_Nanoimager_header():
    columns = load_Nanoimager_header(
        path=locan.ROOT_DIR / 'tests/test_data/Nanoimager_dstorm_data.csv')
    assert (columns == ['channel', 'frame', 'position_x', 'position_y', 'position_z', 'intensity',
                        'local_background'])

    file_like = StringIO("Channel,Frame,X (nm),Y (nm),Z (nm),Photons,Background\n"
                         "0,1548,40918.949219,56104.691406,0.000000,139.828232,0.848500")
    columns = load_Nanoimager_header(path=file_like)
    assert (columns == ['channel', 'frame', 'position_x', 'position_y', 'position_z', 'intensity',
                        'local_background'])


def test_loading_Nanoimager_file():
    dat = load_Nanoimager_file(path=locan.ROOT_DIR / 'tests/test_data/Nanoimager_dstorm_data.csv',
                                  nrows=10)
    #print(dat.data.columns)
    assert (len(dat) == 10)
    assert all(dat.data.columns == ['channel', 'frame', 'position_x', 'position_y', 'position_z', 'intensity',
                                    'local_background'])

    file_like = StringIO("Channel,Frame,X (nm),Y (nm),Z (nm),Photons,Background\n"
                         "0,1548,40918.949219,56104.691406,0.000000,139.828232,0.848500")
    dat = load_Nanoimager_file(path=file_like, nrows=1)
    assert (len(dat) == 1)
    assert all(dat.data.columns == ['channel', 'frame', 'position_x', 'position_y', 'position_z', 'intensity',
                                    'local_background'])


def test_loading_txt_file(caplog):
    dat = load_txt_file(path=locan.ROOT_DIR / 'tests/test_data/five_blobs.txt', nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)

    dat = load_txt_file(path=locan.ROOT_DIR / 'tests/test_data/five_blobs.txt',
                           columns=['index', 'position_x', 'position_y', 'cluster_label'], nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)

    file_like = StringIO("index,position_x,position_y,cluster_label\n"
                         "0,624,919,3")
    dat = load_txt_file(path=file_like,
                           columns=['index', 'position_x', 'position_y', 'cluster_label'], nrows=1)
    assert (len(dat) == 1)

    dat = load_txt_file(path=locan.ROOT_DIR / 'tests/test_data/five_blobs.txt',
                           columns=['c1'], nrows=10)
    assert caplog.record_tuples == [('locan.locan_io.locdata.utilities', logging.WARNING,
                                     'Column c1 is not a Locan property standard.')]
    # print(dat.data)
    assert (len(dat) == 10)

    dat = load_txt_file(path=locan.ROOT_DIR / 'tests/test_data/five_blobs.txt',
                           columns=['c1'], nrows=10, property_mapping={'c1': 'something'})
    assert list(dat.data.columns) == ['something']


def test_save_and_load_asdf(locdata_2d):
    # for visual inspection use:
    # io.save_asdf(locdata_2d, path=locan.ROOT_DIR / 'tests/test_data/locdata.asdf')
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / 'locdata.asdf'
        save_asdf(locdata_2d, path=file_path)

        locdata = load_asdf_file(path=file_path)
        # print(locdata.data)
        assert_frame_equal(locdata.data, locdata_2d.data)
        assert (locdata.meta.identifier == locdata_2d.meta.identifier)
        locdata_2d.properties.pop("localization_density_ch", None)
        locdata_2d.properties.pop("region_measure_ch", None)
        assert (locdata.properties == locdata_2d.properties)

        dat = load_asdf_file(path=file_path, nrows=5)
        assert (len(dat) == 5)


def test__map_file_type_to_load_function():
    file_type = _map_file_type_to_load_function(file_type=1)
    assert callable(file_type)
    file_type = _map_file_type_to_load_function(file_type='RAPIDSTORM')
    assert callable(file_type)
    file_type = _map_file_type_to_load_function(file_type=locan.constants.FileType.RAPIDSTORM)
    assert callable(file_type)
    file_type = _map_file_type_to_load_function(file_type=metadata_pb2.RAPIDSTORM)
    assert callable(file_type)


def test_load_locdata():
    dat = load_locdata(path=locan.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
                       file_type='RAPIDSTORM',
                       nrows=10)
    assert (len(dat) == 10)

    dat = load_locdata(path=locan.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt',
                       file_type='ELYRA',
                       nrows=10)
    assert (len(dat) == 10)

    dat = load_locdata(path=locan.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt',
                       file_type='elyra',
                       nrows=10)
    assert (len(dat) == 10)

    dat = load_locdata(path=locan.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt',
                       file_type=3,
                       nrows=10)
    assert (len(dat) == 10)

    dat = load_locdata(path=locan.ROOT_DIR / 'tests/test_data/Nanoimager_dstorm_data.csv',
                       file_type='NANOIMAGER',
                       nrows=10)
    assert (len(dat) == 10)

    dat = load_locdata(path=locan.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_track_data.txt',
                       file_type='RAPIDSTORMTRACK',
                       nrows=10)
    assert (len(dat) == 10)

    dat = load_locdata(path=locan.ROOT_DIR / 'tests/test_data/SMLM_dstorm_data.smlm',
                       file_type='SMLM',
                       nrows=10)
    assert (len(dat) == 10)

    dat = load_locdata(path=locan.ROOT_DIR / 'tests/test_data/decode_dstorm_data.h5',
                       file_type='DECODE',
                       nrows=10)
    assert (len(dat) == 10)

    dat = load_locdata(path=locan.ROOT_DIR / 'tests/test_data/smap_dstorm_data.mat',
                       file_type='SMAP',
                       nrows=10)
    assert (len(dat) == 10)


def test_pickling_locdata(locdata_2d):
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / 'pickled_locdata.pickle'
        with open(file_path, 'wb') as file:
            pickle.dump(locdata_2d, file, pickle.HIGHEST_PROTOCOL)
        with open(file_path, 'rb') as file:
            locdata = pickle.load(file)
        assert len(locdata_2d) == len(locdata)
        assert isinstance(locdata.meta, metadata_pb2.Metadata)
