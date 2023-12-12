# from locan.utils.miscellaneous import iterate_2d_array
from locan import iterate_2d_array


def test_iterate_2d_array():
    assert list(iterate_2d_array(1, 1)) == [(0, 0)]
    assert list(iterate_2d_array(1, 3)) == [(0, 0)]
    assert list(iterate_2d_array(2, 2)) == [(0, 0), (0, 1)]
    assert list(iterate_2d_array(3, 2)) == [(0, 0), (0, 1), (1, 0)]
    assert list(iterate_2d_array(2, 3)) == [(0, 0), (0, 1)]
    assert list(iterate_2d_array(3, 1)) == [(0, 0), (1, 0), (2, 0)]
    assert list(iterate_2d_array(0, 1)) == []
