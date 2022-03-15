import pytest
import numpy as np

import boost_histogram as bh
from locan.data.aggregate import _is_scalar, _is_single_element, \
    _is_1d_array_of_scalar, _is_1d_array_of_two_scalar,\
    _is_2d_homogeneous_array, _is_2d_inhomogeneous_array, _is_2d_inhomogeneous_array_of_1d_array_of_scalar, \
    _n_bins_to_bin_edges_one_dimension, _bin_size_to_bin_edges_one_dimension, \
    _bin_edges_to_n_bins_one_dimension, _bin_edges_to_n_bins, \
    _bin_edges_to_bin_size_one_dimension, _bin_edges_to_bin_size, \
    _bin_edges_to_bin_centers, _indices_to_bin_centers, \
    _BinsFromBoostHistogramAxis, _BinsFromEdges, _BinsFromNumber, _BinsFromSize
from locan import adjust_contrast, histogram, Bins


data_scalars = {
    '1': 1,
    '()': ()
}

data_tuples = {
    '((),)': ((),),
    '(1,)': (1,),
    '(1, 2)': (1, 2),
    '(1, 2, 3)': (1, 2, 3),
    '((1, 2),)': ((1, 2),),
    '((1, 2), (1, 2))': ((1, 2), (1, 2)),
    '((1, 2), (1, 2, 3))': ((1, 2), (1, 2, 3)),
    '((1, 2), ((1, 2), (1, 2)))': ((1, 2), ((1, 2), (1, 2))),
    '(1, (1, 2))': (1, (1, 2)),
}

data_lists = {
    '[[]]': [[]],
    '[1]': [1],
    '[1, 2]': [1, 2],
    '[1, 2, 3]': [1, 2, 3],
    '[[1, 2]]': [[1, 2]],
    '[[1, 2], [1, 2]]': [[1, 2], [1, 2]],
    '[[1, 2], [1, 2, 3]]': [[1, 2], [1, 2, 3]],
    '[[1, 2], [[1, 2], [1, 2]]]': [[1, 2], [[1, 2], [1, 2]]],
    '[1, [1, 2]]': [1, [1, 2]],
}

data_ndarrays = {
    'np.array((1))': np.array((1)),
    'np.array((1,))': np.array((1,)),
    'np.array((1, 2))': np.array((1, 2)),
    'np.array([(1, 2)])': np.array([(1, 2)]),
    'np.array([(1, 2)], dtype=object)': np.array([(1, 2)], dtype=object),
    'np.array([(1, 2), (1, 2, 3)], dtype=object)': np.array([(1, 2), (1, 2, 3)], dtype=object)
}

data_all = {**data_scalars, **data_tuples, **data_lists, **data_ndarrays}

expect_is_scalar = ['1', 'np.array((1))']

expect_is_single_element = ['1', '(1,)', '[1]', 'np.array((1))', 'np.array((1,))']

expect_is_1d_array_of_scalar = ['(1, 2)', '(1, 2, 3)',
                                '[1, 2]', '[1, 2, 3]',
                                'np.array((1, 2))',
                                '(1,)', '[1]', 'np.array((1,))']

expect_is_1d_array_of_two_scalar = ['(1, 2)', '[1, 2]', 'np.array((1, 2))']

expect_is_2d_homogeneous_array = ['((1, 2),)', '((1, 2), (1, 2))',
                                  '[[1, 2]]', '[[1, 2], [1, 2]]',
                                  'np.array([(1, 2)])']

expect_is_2d_inhomogeneous_array = ['((1, 2), (1, 2, 3))',
                                    '(1, (1, 2))',
                                    '[[1, 2], [1, 2, 3]]',
                                    '[1, [1, 2]]',
                                    'np.array([(1, 2), (1, 2, 3)], dtype=object)']

expect_is_2d_inhomogeneous_array_of_1d_array_of_scalar = ['((1, 2), (1, 2, 3))',
                                                          '[[1, 2], [1, 2, 3]]',
                                                          'np.array([(1, 2), (1, 2, 3)], dtype=object)']


def test__is_scalar():
    for key, value in data_all.items():
        if key in expect_is_scalar:
            assert _is_scalar(value)
        else:
            assert not _is_scalar(value)


def test__is_single_element():
    for key, value in data_all.items():
        if key in expect_is_single_element:
            assert _is_single_element(value)
        else:
            assert not _is_single_element(value)


def test__is_1d_array_of_scalar():
    for key, value in data_all.items():
        if key in expect_is_1d_array_of_scalar:
            assert _is_1d_array_of_scalar(value)
        else:
            assert not _is_1d_array_of_scalar(value)


def test__is_1d_array_of_two_scalar():
    for key, value in data_all.items():
        if key in expect_is_1d_array_of_two_scalar:
            assert _is_1d_array_of_two_scalar(value)
        else:
            assert not _is_1d_array_of_two_scalar(value)


def test__is_2d_homogeneous_array():
    for key, value in data_all.items():
        if key in expect_is_2d_homogeneous_array:
            assert _is_2d_homogeneous_array(value)
        else:
            assert not _is_2d_homogeneous_array(value)


def test__is_2d_inhomogeneous_array():
    for key, value in data_all.items():
        if key in expect_is_2d_inhomogeneous_array:
            assert _is_2d_inhomogeneous_array(value)
        else:
            assert not _is_2d_inhomogeneous_array(value)


def test__is_2d_inhomogeneous_array_of_1d_array_of_scalar():
    for key, value in data_all.items():
        if key in expect_is_2d_inhomogeneous_array_of_1d_array_of_scalar:
            assert _is_2d_inhomogeneous_array_of_1d_array_of_scalar(value)
        else:
            assert not _is_2d_inhomogeneous_array_of_1d_array_of_scalar(value)


def test__n_bins_to_bin_edges_one_dimension():
    bin_edges = _n_bins_to_bin_edges_one_dimension(10, (10, 20))
    assert bin_edges.shape == (11,)


def test__bin_size_to_bin_edges_one_dimension():
    bin_edges = _bin_size_to_bin_edges_one_dimension(4, (1, 10), extend_range=None)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 5, 9))
    bin_edges = _bin_size_to_bin_edges_one_dimension(4, (1, 10), extend_range=True)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 5, 9, 13))
    bin_edges = _bin_size_to_bin_edges_one_dimension(4, (1, 10), extend_range=False)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 5, 9, 10))

    bin_edges = _bin_size_to_bin_edges_one_dimension(20, (1, 10), extend_range=None)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 10))
    bin_edges = _bin_size_to_bin_edges_one_dimension(20, (1, 10), extend_range=True)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 21))
    bin_edges = _bin_size_to_bin_edges_one_dimension(20, (1, 10), extend_range=False)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 10))

    bin_edges = _bin_size_to_bin_edges_one_dimension((1, 2, 3, 3, 2), (1, 11), extend_range=None)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 2, 4, 7, 10))
    bin_edges = _bin_size_to_bin_edges_one_dimension((1, 2, 3, 3, 2), (1, 11), extend_range=True)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 2, 4, 7, 10, 12))
    bin_edges = _bin_size_to_bin_edges_one_dimension((1, 2, 3, 3, 2), (1, 11), extend_range=False)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 2, 4, 7, 10, 11))

    bin_edges = _bin_size_to_bin_edges_one_dimension((10, 20, 30), (1, 2), extend_range=None)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 2))
    bin_edges = _bin_size_to_bin_edges_one_dimension((10, 20, 30), (1, 2), extend_range=True)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 11))
    bin_edges = _bin_size_to_bin_edges_one_dimension((10, 20, 30), (1, 2), extend_range=False)
    isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, (1, 2))

    with pytest.raises(TypeError):
        _bin_size_to_bin_edges_one_dimension(((4,), 2), (0, 10), extend_range=False)


def test__bin_edges_to_n_bins_one_dimension():
    n_bins = _bin_edges_to_n_bins_one_dimension((1, 3, 5))
    assert n_bins == 2
    n_bins = _bin_edges_to_n_bins_one_dimension([1, 2, 4])
    assert n_bins == 2


def test__bin_edges_to_n_bins():
    n_bins = _bin_edges_to_n_bins([1, 3, 5])
    assert n_bins == (2,)
    n_bins = _bin_edges_to_n_bins(([1, 3, 5],))
    assert n_bins == (2,)
    n_bins = _bin_edges_to_n_bins([1, 2, 4])
    assert n_bins == (2,)
    n_bins = _bin_edges_to_n_bins(((1, 3, 5), (1, 2, 4, 5)))
    assert np.array_equal(n_bins, (2, 3))
    n_bins = _bin_edges_to_n_bins([[1, 3, 5], [1, 2, 3, 4]])
    assert np.array_equal(n_bins, (2, 3))
    n_bins = _bin_edges_to_n_bins(np.array([[1, 3, 5], [1, 2, 3]]))
    assert np.array_equal(n_bins, (2, 2))


def test__bin_edges_to_bin_size_one_dimension():
    bin_size = _bin_edges_to_bin_size_one_dimension((1, 3, 5))
    assert bin_size == 2
    bin_size = _bin_edges_to_bin_size_one_dimension([1, 2, 4])
    assert np.array_equal(bin_size, (1, 2))
    bin_size = _bin_edges_to_bin_size_one_dimension([1, 2])
    assert bin_size == 1


def test__bin_edges_to_bin_size():
    bin_size = _bin_edges_to_bin_size([1, 3, 5])
    assert bin_size == (2,)
    bin_size = _bin_edges_to_bin_size(([1, 3, 5],))
    assert bin_size == (2,)
    bin_size = _bin_edges_to_bin_size([1, 2, 4])
    assert np.array_equal(bin_size[0], (1, 2))
    bin_size = _bin_edges_to_bin_size(((1, 3, 5), (1, 2, 4, 5)))
    assert bin_size[0] == 2
    assert np.array_equal(bin_size[1], (1, 2, 1))
    bin_size = _bin_edges_to_bin_size([[1, 3, 5], [1, 2, 3, 4]])
    assert bin_size == (2, 1)
    bin_size = _bin_edges_to_bin_size(np.array([[1, 3, 5], [1, 2, 3]]))
    assert bin_size == (2, 1)


def test__bin_edges_to_bin_centers():
    bin_centers = _bin_edges_to_bin_centers([1, 3, 5])
    assert np.array_equal(bin_centers, ((2, 4),))

    bin_centers = _bin_edges_to_bin_centers(([1, 3, 5],))
    assert np.array_equal(bin_centers, ((2, 4),))

    bin_centers = _bin_edges_to_bin_centers(((1, 3, 5), (1, 2, 4, 6)))
    expected = ((2, 4), (1.5, 3, 5))
    for bc, ex in zip(bin_centers, expected):
        assert np.array_equal(bc, ex)

    bin_edges = np.array([[0, 1, 2, 4, 8, 9], [0, 1, 4, 8]], dtype=object)
    bin_centers = _bin_edges_to_bin_centers(bin_edges)
    expected = (np.array([0.5, 1.5, 3., 6, 8.5]), np.array([0.5, 2.5, 6]))
    for bc, ex in zip(bin_centers, expected):
        assert np.array_equal(bc, ex)


def test__indices_to_bin_centers():
    indices = 2
    bin_edges = np.array([0, 1, 2, 4, 8, 9])
    bin_centers = _indices_to_bin_centers(bin_edges, indices)
    expected = 3
    assert np.array_equal(bin_centers, expected)

    indices = np.array([0, 2, 1])
    bin_edges = np.array([0, 1, 2, 4, 8, 9])
    bin_centers = _indices_to_bin_centers(bin_edges, indices)
    expected = np.array([0.5, 3, 1.5])
    assert np.array_equal(bin_centers, expected)

    indices = np.array([[0, 1], [2, 2], [4, 3]])
    bin_edges = np.array([0, 1, 2, 4, 8, 9])
    bin_centers = _indices_to_bin_centers(bin_edges, indices)
    expected = np.array([[0.5, 1.5], [3, 3], [8.5, 6]])
    assert np.array_equal(bin_centers, expected)

    indices = np.array([[0, 1], [2, 2], [4, 3]])
    bin_edges = np.array([[0, 1, 2, 4, 8, 9], [1, 2, 4, 8, 9]], dtype=object)
    bin_centers = _indices_to_bin_centers(bin_edges, indices)
    expected = np.array([[0.5, 3], [3, 6], [8.5, 8.5]])
    assert np.array_equal(bin_centers, expected)


def test__BinsFromBoostHistogramAxis():
    bhaxis = bh.axis.Regular(5, 0, 10)
    bins = _BinsFromBoostHistogramAxis(bins=bhaxis)
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == ((2, 2, 2, 2, 2),)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))

    bhaxes = bh.axis.AxesTuple((bh.axis.Regular(5, 0, 10), bh.axis.Regular(2, 0, 10)))
    bins = _BinsFromBoostHistogramAxis(bins=bhaxes)
    assert bins.dimension == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 10.0))
    assert bins.n_bins == (5, 2)
    assert bins.bin_size == ((2.0, 2.0, 2.0, 2.0, 2.0), (5.0, 5.0))
    expected_edges = [np.array([0, 2, 4, 6, 8, 10]), np.array([0, 5, 10])]
    for bin_edges, edges in zip(bins.bin_edges, expected_edges):
        assert np.array_equal(bin_edges, edges)
    expected_centers = [np.array([1, 3, 5, 7, 9]), np.array([2.5, 7.5])]
    for bin_centers, expected_cents in zip(bins.bin_centers, expected_centers):
        assert np.array_equal(bin_centers, expected_cents)


def test__BinsFromEdges():
    bins = _BinsFromEdges(bin_edges=(0, 2, 4))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 4.0),)
    assert bins.n_bins == (2,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4]))

    bins = _BinsFromEdges(bin_edges=((0, 2, 4),))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 4.0),)
    assert bins.n_bins == (2,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4]))

    bins = _BinsFromEdges(bin_edges=(1, 2, 5))
    assert bins.dimension == 1
    assert bins.bin_range == ((1, 5),)
    assert bins.n_bins == (2,)
    assert np.array_equal(bins.bin_size[0], np.array([1, 3]))
    assert np.array_equal(bins.bin_edges[0], np.array([1, 2, 5]))

    bins = _BinsFromEdges(bin_edges=((0, 2, 4), (1, 2, 5)))
    assert bins.dimension == 2
    assert bins.bin_range == ((0.0, 4.0), (1.0, 5.0))
    assert bins.n_bins == (2, 2)
    assert bins.bin_size[0] == 2
    assert np.array_equal(bins.bin_size[1], np.array([1, 3]))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4]))
    assert np.array_equal(bins.bin_edges[-1], np.array([1, 2, 5]))

    with pytest.raises(TypeError):
        _BinsFromEdges(bin_edges=())
    with pytest.raises(TypeError):
        _BinsFromEdges(bin_edges=[(0, 2, 4), ((0, 2, 4), (0, 2, 4))])
    with pytest.raises(TypeError):
        _BinsFromEdges(bin_edges=[(0, 2, 4), 2])


def test__BinsFromNumber_():
    bins = _BinsFromNumber(n_bins=5, bin_range=(0, 10))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))

    bins = _BinsFromNumber(n_bins=(5,), bin_range=(0, 10))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))

    bins = _BinsFromNumber(n_bins=(2, 5), bin_range=((0, 10), (0, 5)))
    assert bins.dimension == 2
    assert bins.bin_range == ((0, 10), (0, 5))
    assert bins.n_bins == (2, 5)
    assert bins.bin_size == (5, 1)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 5, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 2, 3, 4, 5]))

    bins = _BinsFromNumber(n_bins=2, bin_range=((0, 10), (0, 5)))
    assert bins.dimension == 2
    assert bins.bin_range == ((0, 10), (0, 5))
    assert bins.n_bins == (2, 2)
    assert bins.bin_size == (5, 2.5)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 5, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 2.5, 5]))

    bins = _BinsFromNumber(n_bins=(2, 5, 2), bin_range=(0, 10))
    assert bins.dimension == 3
    assert bins.bin_range == ((0, 10), (0, 10), (0, 10))
    assert bins.n_bins == (2, 5, 2)
    assert bins.bin_size == (5, 2, 5)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 5, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_edges[2], np.array([0, 5, 10]))

    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=5, bin_range=1)
    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=5, bin_range=(0,))
    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=5, bin_range=(1, 2, 3))
    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=(5, (1, 2)), bin_range=(1, 2, 3))

    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=(5,), bin_range=1)
    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=(5,), bin_range=(0,))
    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=(5,), bin_range=(1, 2, 3))
    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=(2,), bin_range=((0, 10), (0, 5)))

    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=(2, 5, 2), bin_range=1)
    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=(2, 5, 2), bin_range=(0,))
    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=(2, 5, 2), bin_range=(1, 2, 3))

    with pytest.raises(TypeError):
        _BinsFromNumber(n_bins=(5, (1, 2)), bin_range=(1, 2))


def test__BinsFromSize():
    bins = _BinsFromSize(bin_size=2, bin_range=(0, 10))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))

    bins = _BinsFromSize(bin_size=(2,), bin_range=(0, 10))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))

    bins = _BinsFromSize(bin_size=((1, 2, 3, 4),), bin_range=(0, 10))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (4,)
    assert np.array_equal(bins.bin_size[0], (1, 2, 3, 4))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 1, 3, 6, 10]))

    bins = _BinsFromSize(bin_size=3, bin_range=(0, 10), extend_range=False)
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (4,)
    assert np.array_equal(bins.bin_size[0], (3.0, 3.0, 3.0, 1.0))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 3, 6, 9, 10]))

    bins = _BinsFromSize(bin_size=(5, 1), bin_range=((0, 10), (0, 5)))
    assert bins.dimension == 2
    assert bins.bin_range == ((0, 10), (0, 5))
    assert bins.n_bins == (2, 5)
    assert bins.bin_size == (5, 1)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 5, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 2, 3, 4, 5]))

    bins = _BinsFromSize(bin_size=2, bin_range=((0, 10), (0, 5)))
    assert bins.dimension == 2
    assert bins.bin_range == ((0, 10), (0, 4))
    assert bins.n_bins == (5, 2)
    assert bins.bin_size == (2, 2)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 2, 4]))

    bins = _BinsFromSize(bin_size=(2, 5), bin_range=(0, 10))
    assert bins.dimension == 2
    assert bins.bin_range == ((0, 10), (0, 10))
    assert bins.n_bins == (5, 2)
    assert bins.bin_size == (2, 5)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 5, 10]))

    bins = _BinsFromSize(bin_size=((1, 2, 3, 4), (1, 2, 3, 1)), bin_range=(0, 10))
    assert bins.dimension == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 7.0))
    assert bins.n_bins == (4, 4)
    assert np.array_equal(bins.bin_size[0], (1, 2, 3, 4))
    assert np.array_equal(bins.bin_size[1], (1, 2, 3, 1))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 1, 3, 6, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 3, 6, 7]))

    bins = _BinsFromSize(bin_size=((1, 2, 3, 4), (1, 2, 3, 4)), bin_range=((0, 10), (0, 10)))
    assert bins.dimension == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 10.0))
    assert bins.n_bins == (4, 4)
    assert np.array_equal(bins.bin_size[0], (1, 2, 3, 4))
    assert np.array_equal(bins.bin_size[1], (1, 2, 3, 4))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 1, 3, 6, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 3, 6, 10]))

    bins = _BinsFromSize(bin_size=(2, (1, 2, 3, 4)), bin_range=(0, 10))
    assert bins.dimension == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 10.0))
    assert bins.n_bins == (5, 4)
    assert bins.bin_size[0] == 2
    assert np.array_equal(bins.bin_size[1], (1, 2, 3, 4))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 3, 6, 10]))

    bins = _BinsFromSize(bin_size=(2, (1, 2, 3, 4)), bin_range=((0, 10), (0, 20)))
    assert bins.dimension == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 10.0))
    assert bins.n_bins == (5, 4)
    assert bins.bin_size[0] == 2
    assert np.array_equal(bins.bin_size[1], (1, 2, 3, 4))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 3, 6, 10]))

    with pytest.raises(TypeError):
        _BinsFromSize(bin_size=5, bin_range=(0,))
    with pytest.raises(TypeError):
        _BinsFromSize(bin_size=5, bin_range=(1, 2, 3))
    with pytest.raises(TypeError):
        _BinsFromSize(bin_size=(2, 2, (1, 2, 3, 4)), bin_range=((0, 10), (0, 20)))
    with pytest.raises(TypeError):
        _BinsFromSize(bin_size=((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)), bin_range=((0, 10), (0, 20)))


def test_Bins():
    bins = Bins(bin_edges=(0, 2, 4))
    assert bins.dimension == 1
    assert bins.bin_range == ((0, 4),)
    assert bins.n_bins == (2,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3]))
    assert bins.is_equally_sized == (True,)

    bins = Bins(n_bins=5, bin_range=(0, 10))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))
    assert bins.is_equally_sized == (True,)

    bins = Bins(bin_size=2, bin_range=(0, 10))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))
    assert bins.is_equally_sized == (True,)

    bins = Bins(bin_size=(2, (1, 2, 3)), bin_range=(0, 10))
    assert bins.dimension == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 6.0))
    assert bins.n_bins == (5, 3)
    assert np.array_equal(bins.bin_size[0], 2)
    assert np.array_equal(bins.bin_size[1], (1, 2, 3))
    assert bins.is_equally_sized == (True, False)

    bins = Bins(bins=Bins(n_bins=5, bin_range=(0, 10)))
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))
    assert bins.is_equally_sized == (True,)

    bins = Bins(n_bins=(2, 5), bin_range=(0, 10), labels=["position_x", "position_y"])
    assert bins.labels == ["position_x", "position_y"]
    assert bins.dimension == 2
    assert bins.is_equally_sized == (True, True)
    bins = Bins(bins=Bins(n_bins=5, bin_range=(0, 10)), labels=["position_x"])
    assert bins.labels == ["position_x"]
    assert bins.dimension == 1
    bins = Bins(bins=Bins(n_bins=5, bin_range=(0, 10), labels=["position_x"]))
    assert bins.labels == ["position_x"]
    assert bins.dimension == 1
    with pytest.raises(ValueError):
        Bins(n_bins=5, bin_range=(0, 10), labels=["position_x", "position_y"])


def test_Bins_with_boost_histogram():
    bhaxis = bh.axis.Regular(5, 0, 10)
    bins = Bins(bins=bhaxis)
    assert bins.dimension == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == ((2, 2, 2, 2, 2),)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))
    assert bins.is_equally_sized == (True,)


def test_Bins_methods():
    bins = Bins(bin_edges=(0, 1, 2, 4))
    assert bins.dimension == 1
    assert bins.bin_range == ((0, 4),)
    assert bins.n_bins == (3,)
    assert np.array_equal(bins.bin_size[0], (1, 1, 2))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 1, 2, 4]))
    assert bins.is_equally_sized == (False,)

    bins = Bins(bin_edges=(0, 1, 2, 4)).equalize_bin_size()
    assert bins.dimension == 1
    assert bins.bin_range == ((0, 4),)
    assert bins.n_bins == (4,)
    assert bins.bin_size == (1,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 1, 2, 3, 4]))
    assert bins.is_equally_sized == (True,)


def test_adjust_contrast():
    img = np.array((1, 2, 3, 9), dtype=np.float64)
    new_img = adjust_contrast(img, rescale=(50, 100))
    assert np.array_equal(new_img, np.array((0, 0, 0, 1)))

    img = np.array((1, 2, 3, 9), dtype=np.uint8)
    new_img = adjust_contrast(img, rescale=None)
    assert np.array_equal(img, new_img)

    new_img = adjust_contrast(img, rescale=False)
    assert np.array_equal(img, new_img)

    new_img = adjust_contrast(img, rescale='equal')
    assert max(new_img) == 1

    new_img = adjust_contrast(img, rescale=True)
    assert np.array_equal(new_img, np.array((0, 31, 63, 255)))

    new_img = adjust_contrast(img, rescale=(0, 50))
    assert np.array_equal(new_img, np.array((0, 63, 127, 255)))

    new_img = adjust_contrast(img, out_range=(0, 10))
    assert np.array_equal(new_img, np.array((0, 1.25, 2.5, 10)))

    new_img = adjust_contrast(img * 1., rescale=True)
    assert np.array_equal(new_img, np.array((0, 0.125, 0.25, 1.)))

    new_img = adjust_contrast(img, rescale='unity')
    assert np.array_equal(new_img, np.array((0, 0.125, 0.25, 1.)))


def test_histogram(locdata_blobs_2d):
    hist = histogram(locdata_blobs_2d, n_bins=10)
    assert hist.labels == ['position_x', 'position_y', 'counts']
    assert hist.data.dtype == 'float64'
    assert hist.data.ndim == 2
    assert np.max(hist.data) == 7

    hist = histogram(locdata_blobs_2d, n_bins=10, bin_range=((500, 1000), (500, 1000)))
    assert hist.data.ndim == 2
    assert np.max(hist.data) == 5
    assert hist.data.shape == (10, 10)

    hist = histogram(locdata_blobs_2d, bin_size=10, loc_properties='position_x')
    assert hist.labels == ['position_x', 'counts']
    assert hist.data.shape == (89,)

    with pytest.raises(ValueError):
        histogram(locdata_blobs_2d, bin_size=10, loc_properties='position_x', bin_range=((500, 1000), (500, 1000)))

    hist = histogram(locdata_blobs_2d, bin_size=10, loc_properties=['position_x'])
    assert hist.labels == ['position_x', 'counts']
    assert hist.data.shape == (89,)

    hist = histogram(locdata_blobs_2d, bin_size=10, loc_properties=['position_x', 'position_y'])
    assert hist.labels == ['position_x', 'position_y', 'counts']
    assert hist.data.shape == (89, 55)

    hist = histogram(locdata_blobs_2d, bin_size=10, loc_properties=['position_x', 'cluster_label'])
    assert hist.labels == ['position_x', 'cluster_label', 'counts']
    assert hist.data.shape == (89, 1)

    with pytest.raises(ValueError):
        histogram(locdata_blobs_2d, bin_size=10, loc_properties='position_z')

    with pytest.raises(ValueError):
        histogram(locdata_blobs_2d, bin_size=10, loc_properties=['position_z'])

    with pytest.raises(ValueError):
        histogram(locdata_blobs_2d, bin_size=10, loc_properties=['position_x', 'position_z'])

    hist = histogram(locdata_blobs_2d, bin_edges=((500, 600, 700, 800, 900, 1000), (500, 600, 700, 800, 900, 1000)))
    assert hist.data.ndim == 2
    assert np.max(hist.data) == 7

    hist = histogram(locdata_blobs_2d, bin_size=10, bin_range='zero', rescale=True)
    assert np.max(hist.data) == 1
    assert 'counts' in hist.labels

    hist = histogram(locdata_blobs_2d, bin_size=10, other_property='position_y')
    assert hist.labels == ['position_x', 'position_y', 'position_y']
    assert hist.data.shape == (89, 55)


def test_histogram_1d(locdata_1d):
    hist = histogram(locdata_1d, n_bins=10)
    assert hist.labels == ['position_x', 'counts']
    assert hist.data.dtype == 'float64'
    assert hist.data.ndim == 1
    assert np.max(hist.data) == 2
    assert hist.data.shape == (10,)

    hist = histogram(locdata_1d, n_bins=5, bin_range=(5, 10))
    assert np.max(hist.data) == 1
    assert hist.data.shape == (5,)

    hist = histogram(locdata_1d,
                     bin_edges=(5, 6, 7, 8, 9, 10))
    assert hist.data.shape == (5,)

    hist = histogram(locdata_1d, n_bins=10, other_property='intensity')
    assert hist.labels == ['position_x', 'intensity']
    assert hist.data.shape == (10,)
    assert np.nanmax(hist.data) == 125


def test_histogram_3d(locdata_blobs_3d):
    hist = histogram(locdata_blobs_3d, n_bins=10)
    assert hist.labels == ['position_x', 'position_y', 'position_z', 'counts']
    assert hist.data.dtype == 'float64'
    assert hist.data.ndim == 3
    assert np.max(hist.data) == 6
    assert hist.data.shape == (10, 10, 10)

    hist = histogram(locdata_blobs_3d, n_bins=10, bin_range=((500, 1000), (500, 1000), (500, 1000)))
    assert np.max(hist.data) == 4
    assert hist.data.shape == (10, 10, 10)

    hist = histogram(locdata_blobs_3d,
                     bin_edges=((500, 600, 700, 800, 900, 1000),
                                (500, 600, 700, 800, 900, 1000),
                                (500, 600, 700, 800, 900, 1000)
                                ))
    assert hist.data.shape == (5, 5, 5)

    hist = histogram(locdata_blobs_3d, n_bins=10, other_property='position_y')
    assert hist.labels == ['position_x', 'position_y', 'position_z', 'position_y']
    assert hist.data.shape == (10, 10, 10)
    assert np.nanmax(hist.data) == 787


def test_histogram_empty(locdata_empty):
    with pytest.raises(TypeError):
        hist = histogram(locdata_empty, n_bins=10)


def test_histogram_single_value(locdata_single_localization_3d):
    hist = histogram(locdata_single_localization_3d, n_bins=3)
    assert hist.data.shape == (3, 3, 3)
    assert np.array_equal(hist.bins.bin_range, [[1, 2], [1, 2], [1, 2]])

    hist = histogram(locdata_single_localization_3d, bin_size=0.2)
    assert hist.data.shape == (5, 5, 5)
    assert np.array_equal(hist.bins.bin_range, [[1, pytest.approx(2)], [1, pytest.approx(2)], [1, pytest.approx(2)]])

    hist = histogram(locdata_single_localization_3d, bin_size=2)
    assert hist.data.shape == (1, 1, 1)
    assert np.array_equal(hist.bins.bin_range, [[1, 2], [1, 2], [1, 2]])
