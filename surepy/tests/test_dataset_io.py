import warnings
import pytest
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data

def test_get_correct_column_names_from_rapidSTORM_header():
    columns = io.load_rapidSTORM_header(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt')
    assert (columns == ['Position_x', 'Position_y', 'Frame', 'Intensity', 'Chi_square', 'Local_background'])

def test_loading_rapidSTORM_file():
    dat = io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt', nrows=10)
    #dat.print_meta()
    assert (len(dat) == 10)

def test_get_correct_column_names_from_Elyra_header():
    columns = io.load_Elyra_header(path=surepy.constants.ROOT_DIR + '/tests/test_data/Elyra_dstorm_data.txt')
    assert (columns == ['Index', 'Frame', 'Frames_number', 'Frames_missing', 'Position_x', 'Position_y',
                        'Precision', 'Intensity', 'Local_background', 'Chi_square', 'Psf_half_width', 'Channel',
                        'Slice_z'])

def test_loading_Elyra_file():
    dat = io.load_Elyra_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/Elyra_dstorm_data.txt', nrows=10)
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


