import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data

def test_get_correct_column_names_from_rapidSTORM_header():
    columns = io.load_rapidSTORM_header(path=surepy.constants.ROOT_DIR + '/tests/test_data/someData.txt')
    assert (columns == ['Position_x', 'Position_y', 'Frame', 'Intensity', 'Chi_square', 'Local_background'])

def test_loading_rapidSTORM_file():
    dat = io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/someData.txt', nrows=10)
    assert (len(dat) == 10)
