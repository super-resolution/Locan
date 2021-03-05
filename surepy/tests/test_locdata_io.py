from platform import system
from pathlib import Path
import tempfile
import pickle
from io import StringIO, BytesIO, TextIOWrapper
import logging

import pytest
from pandas.testing import assert_frame_equal
import surepy.constants
import surepy.io.io_locdata as io
from surepy.data import metadata_pb2


def test_open_path_or_file_like():
    if system() == 'Linux':
        inputs = [
            str(surepy.constants.ROOT_DIR) + '/tests/test_data/rapidSTORM_dstorm_data.txt',
            surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
        ]
    elif system() == 'Windows':
        inputs = [
            str(surepy.constants.ROOT_DIR) + '/tests/test_data/rapidSTORM_dstorm_data.txt',
            str(surepy.constants.ROOT_DIR) + r'\tests\test_data\rapidSTORM_dstorm_data.txt',
            surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
            surepy.constants.ROOT_DIR / r'tests\test_data\rapidSTORM_dstorm_data.txt'
        ]
    else:
        inputs = [
            str(surepy.constants.ROOT_DIR) + '/tests/test_data/rapidSTORM_dstorm_data.txt',
            surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
        ]

    for pfl in inputs:
        with io.open_path_or_file_like(path_or_file_like=pfl) as data:
            assert isinstance(data, TextIOWrapper)
            out = data.read(10)
            assert out and isinstance(out, str)
            assert not data.closed
        assert data.closed

    pfl = StringIO("This is a file-like object to be read.")
    with io.open_path_or_file_like(path_or_file_like=pfl) as data:
        assert isinstance(data, StringIO)
        out = data.read(10)
        assert out and isinstance(out, str)
        assert not data.closed
    assert data.closed

    pfl = BytesIO(b"This is a file-like object to be read.")
    with io.open_path_or_file_like(path_or_file_like=pfl) as data:
        assert isinstance(data, BytesIO)
        out = data.read(10)
        assert out and isinstance(out, bytes)
        assert not data.closed
    assert data.closed

    pfl = surepy.constants.ROOT_DIR / 'tests/test_data/some_file_that_does_not_exits.txt'
    with pytest.raises(FileNotFoundError):
        with io.open_path_or_file_like(path_or_file_like=pfl) as data:
            pass


def test_get_correct_column_names_from_rapidSTORM_header():
    columns = io.load_rapidSTORM_header(path=surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt')
    assert columns == ['position_x', 'position_y', 'frame', 'intensity', 'chi_square', 'local_background']

    file_like = StringIO('# <localizations insequence="true" repetitions="variable"><field identifier="Position-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in X" unit="nanometer" min="0 m" max="3.27165e-005 m" /><field identifier="Position-1-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in Y" unit="nanometer" min="0 m" max="3.27165e-005 m" /><field identifier="ImageNumber-0-0" syntax="integer" semantic="frame number" unit="frame" min="0 fr" /><field identifier="Amplitude-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="emission strength" unit="A/D count" /><field identifier="FitResidues-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="fit residue chi square value" unit="dimensionless" /><field identifier="LocalBackground-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="local background" unit="A/D count" /></localizations>\n'
                         '9657.4 24533.5 0 33290.1 1.19225e+006 767.733')
    columns = io.load_rapidSTORM_header(path=file_like)
    assert columns == ['position_x', 'position_y', 'frame', 'intensity', 'chi_square', 'local_background']


def test_loading_rapidSTORM_file():
    dat = io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
                                  nrows=10)
    #print(dat.data.head())
    #dat.print_meta()
    assert (len(dat) == 10)

    file_like = StringIO('# <localizations insequence="true" repetitions="variable"><field identifier="Position-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in X" unit="nanometer" min="0 m" max="3.27165e-005 m" /><field identifier="Position-1-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in Y" unit="nanometer" min="0 m" max="3.27165e-005 m" /><field identifier="ImageNumber-0-0" syntax="integer" semantic="frame number" unit="frame" min="0 fr" /><field identifier="Amplitude-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="emission strength" unit="A/D count" /><field identifier="FitResidues-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="fit residue chi square value" unit="dimensionless" /><field identifier="LocalBackground-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="local background" unit="A/D count" /></localizations>\n'
                         '9657.4 24533.5 0 33290.1 1.19225e+006 767.733')
    dat = io.load_rapidSTORM_file(path=file_like, nrows=1)
    assert (len(dat) == 1)


def test_get_correct_column_names_from_Elyra_header():
    columns = io.load_Elyra_header(path=surepy.constants.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt')
    assert (columns == ['original_index', 'frame', 'frames_number', 'frames_missing', 'position_x', 'position_y',
                        'uncertainty', 'intensity', 'local_background_sigma', 'chi_square', 'psf_half_width', 'channel',
                        'slice_z'])

    file_like = StringIO("Index	First Frame	Number Frames	Frames Missing	Position X [nm]	Position Y [nm]\n"
                         "1  1   1   0   15850.6 23502.1")
    columns = io.load_Elyra_header(path=file_like)
    assert (columns == ['original_index', 'frame', 'frames_number', 'frames_missing', 'position_x', 'position_y'])


def test_loading_Elyra_file():
    dat = io.load_Elyra_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/Elyra_dstorm_data.txt')
    # loading is not limited by nrows=10 to ensure correct treatment of file appendix and NUL character.
    assert (len(dat) == 999)

    file_like = StringIO("Index\tFirst Frame\tNumber Frames\tFrames Missing\tPosition X [nm]\tPosition Y [nm]\n"
                         "1\t1\t1\t0\t15850.6\t23502.1")
    dat = io.load_Elyra_file(path=file_like)
    # loading is not limited by nrows=10 to ensure correct treatment of file appendix and NUL character.
    assert (len(dat) == 1)


def test_get_correct_column_names_from_Thunderstorm_header():
    columns = io.load_thunderstorm_header(
        path=surepy.constants.ROOT_DIR / 'tests/test_data/Thunderstorm_dstorm_data.csv')
    assert (columns == ['original_index', 'frame', 'position_x', 'position_y', 'psf_sigma', 'intensity',
                        'local_background', 'local_background_sigma', 'chi_square', 'uncertainty'])

    file_like = StringIO("id,frame,x [nm],y [nm]\n"
                         "73897.0,2001.0,1320.109670647555,26344.7124618434")
    columns = io.load_thunderstorm_header(path=file_like)
    assert (columns == ['original_index', 'frame', 'position_x', 'position_y'])


def test_loading_Thunderstorm_file():
    dat = io.load_thunderstorm_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/Thunderstorm_dstorm_data.csv',
                                nrows=10)
    #print(dat.data.columns)
    assert (len(dat) == 10)

    file_like = StringIO("id,frame,x [nm],y [nm]\n"
                         "73897.0,2001.0,1320.109670647555,26344.7124618434")
    dat = io.load_thunderstorm_file(path=file_like, nrows=1)
    assert (len(dat) == 1)


def test_get_correct_column_names_from_Nanoimager_header():
    columns = io.load_Nanoimager_header(
        path=surepy.constants.ROOT_DIR / 'tests/test_data/Nanoimager_dstorm_data.csv')
    assert (columns == ['channel', 'frame', 'position_x', 'position_y', 'position_z', 'intensity',
                        'local_background'])

    file_like = StringIO("Channel,Frame,X (nm),Y (nm),Z (nm),Photons,Background\n"
                         "0,1548,40918.949219,56104.691406,0.000000,139.828232,0.848500")
    columns = io.load_Nanoimager_header(path=file_like)
    assert (columns == ['channel', 'frame', 'position_x', 'position_y', 'position_z', 'intensity',
                        'local_background'])


def test_loading_Nanoimager_file():
    dat = io.load_Nanoimager_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/Nanoimager_dstorm_data.csv',
                                nrows=10)
    #print(dat.data.columns)
    assert (len(dat) == 10)
    assert all(dat.data.columns == ['channel', 'frame', 'position_x', 'position_y', 'position_z', 'intensity',
                                    'local_background'])

    file_like = StringIO("Channel,Frame,X (nm),Y (nm),Z (nm),Photons,Background\n"
                         "0,1548,40918.949219,56104.691406,0.000000,139.828232,0.848500")
    dat = io.load_Nanoimager_file(path=file_like, nrows=1)
    assert (len(dat) == 1)
    assert all(dat.data.columns == ['channel', 'frame', 'position_x', 'position_y', 'position_z', 'intensity',
                                    'local_background'])


def test_loading_txt_file(caplog):
    dat = io.load_txt_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/five_blobs.txt', nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)

    dat = io.load_txt_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/five_blobs.txt',
                           columns=['index', 'position_x', 'position_y', 'cluster_label'], nrows=10)
    # print(dat.data)
    assert (len(dat) == 10)

    file_like = StringIO("index,position_x,position_y,cluster_label\n"
                         "0,624,919,3")
    dat = io.load_txt_file(path=file_like,
                           columns=['index', 'position_x', 'position_y', 'cluster_label'], nrows=1)
    assert (len(dat) == 1)

    dat = io.load_txt_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/five_blobs.txt',
                           columns=['c1'], nrows=10)
    assert caplog.record_tuples == [('surepy.io.io_locdata', logging.WARNING,
                                     'Column c1 is not a Surepy property standard.')]
    # print(dat.data)
    assert (len(dat) == 10)


def test_save_and_load_asdf(locdata_2d):
    # for visual inspection use:
    # io.save_asdf(locdata_2d, path=surepy.constants.ROOT_DIR / 'tests/test_data/locdata.asdf')
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / 'locdata.asdf'
        io.save_asdf(locdata_2d, path=file_path)

        locdata = io.load_asdf_file(path=file_path)
        # print(locdata.data)
        assert_frame_equal(locdata.data, locdata_2d.data)
        assert (locdata.meta.identifier == locdata_2d.meta.identifier)
        assert (locdata.properties == locdata_2d.properties)

        dat = io.load_asdf_file(path=file_path, nrows=5)
        assert (len(dat) == 5)


def test__map_file_type_to_load_function():
    file_type = io._map_file_type_to_load_function(file_type=1)
    assert callable(file_type)
    file_type = io._map_file_type_to_load_function(file_type='RAPIDSTORM')
    assert callable(file_type)
    file_type = io._map_file_type_to_load_function(file_type=surepy.constants.FileType.RAPIDSTORM)
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


def test_pickling_locdata(locdata_2d):
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / 'pickled_locdata.pickle'
        with open(file_path, 'wb') as file:
            pickle.dump(locdata_2d, file, pickle.HIGHEST_PROTOCOL)
        with open(file_path, 'rb') as file:
            locdata = pickle.load(file)
        assert len(locdata_2d) == len(locdata)
        assert isinstance(locdata.meta, metadata_pb2.Metadata)
