import pytest
import numpy as np

from surepy.render.utilities import _coordinate_ranges, _bin_edges, _bin_edges_to_number, _bin_edges_from_size, \
    _bin_edges_to_centers, _indices_to_bin_centers


def test__ranges(locdata_blobs_2d):
    hull_ranges = locdata_blobs_2d.bounding_box.hull
    assert np.array_equal(_coordinate_ranges(locdata_blobs_2d, range=None), hull_ranges.T)
    hull_ranges[0] = (0, 0)
    assert np.array_equal(_coordinate_ranges(locdata_blobs_2d, range='zero'), hull_ranges.T)
    with pytest.raises(ValueError):
        _coordinate_ranges(locdata_blobs_2d, range='test')
    assert np.array_equal(_coordinate_ranges(locdata_blobs_2d, range=((10, 100), (20, 200))), ((10, 100), (20, 200)))
    with pytest.raises(TypeError):
        _coordinate_ranges(locdata_blobs_2d, range=(10, 100))


def test__bin_edges():
    bin_edges = _bin_edges(10, (10, 20))
    assert bin_edges.shape == (11,)

    bin_edges = _bin_edges(10, [10, 20])
    assert bin_edges.shape == (11,)

    bin_edges = _bin_edges((5, 10), (10, 20))
    assert len(bin_edges) == 2
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (11,)

    bin_edges = _bin_edges(5, ((10, 20), (20, 30)))
    assert len(bin_edges) == 2
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (6,)

    bin_edges = _bin_edges((5, 10), ((10, 20), (20, 30)))
    assert len(bin_edges) == 2
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (11,)

    bin_edges = _bin_edges((3, (5, 10)), ((1, 5), ((10, 20), (20, 30))))
    assert len(bin_edges) == 2
    assert len(bin_edges[0]) == 4
    assert len(bin_edges[1]) == 2
    assert bin_edges[1][0].shape == (6,)
    assert bin_edges[1][1].shape == (11,)


def test__bin_edges_from_size():
    bin_edges = _bin_edges_from_size(1, (10, 20))
    assert bin_edges.shape == (11,)

    bin_edges = _bin_edges_from_size(5, (0, 17))
    assert bin_edges.shape == (5,)
    assert bin_edges[-1] > 17

    bin_edges = _bin_edges_from_size(5, (0, 17), extend_range=False)
    assert bin_edges.shape == (5,)
    assert bin_edges[-1] == 17

    bin_edges = _bin_edges_from_size(5, (0, 17), extend_range=None)
    assert bin_edges.shape == (4,)
    assert bin_edges[-1] < 17

    bin_edges = _bin_edges_from_size(1, [10, 20])
    assert bin_edges.shape == (11,)

    bin_edges = _bin_edges_from_size((2, 1), (10, 20))
    assert len(bin_edges) == 2
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (11,)

    bin_edges = _bin_edges_from_size(2, ((10, 20), (20, 30)))
    assert len(bin_edges) == 2
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (6,)

    bin_edges = _bin_edges_from_size((2, 1), ((10, 20), (20, 30)))
    assert len(bin_edges) == 2
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (11,)

    bin_edges = _bin_edges_from_size((2, (2, 1)), ((1, 5), ((10, 20), (20, 30))))
    assert len(bin_edges) == 2
    assert len(bin_edges[0]) == 3
    assert len(bin_edges[1]) == 2
    assert bin_edges[1][0].shape == (6,)
    assert bin_edges[1][1].shape == (11,)


def test__bin_edges_to_number():
    n_bins = _bin_edges_to_number([1, 3, 5])
    assert n_bins == 2

    with pytest.warns(UserWarning):
        n_bins = _bin_edges_to_number([1, 2, 4])
    assert n_bins is None

    with pytest.warns(UserWarning):
        n_bins = _bin_edges_to_number(((1, 3, 5), (1, 2, 4)))
    assert np.array_equal(n_bins, (2, None))

    n_bins = _bin_edges_to_number([[1, 3, 5], [1, 2, 3, 4]])
    assert np.array_equal(n_bins, (2, 3))


def test__bin_edges_to_centers():
    bin_edges = np.array([[0, 1, 2, 4, 8, 9], [0, 1, 4, 8]], dtype=object)
    bin_centers = _bin_edges_to_centers(bin_edges)
    expected = np.array([np.array([0.5, 1.5, 3., 6, 8.5]),np.array([0.5, 2.5, 6])], dtype=object)
    for bc, ex in zip(bin_centers, expected):
        assert np.array_equal(bc, ex)


def test__indices_to_bin_centers():
    indices = np.array([[0, 1], [2, 2], [4, 3]])
    bin_edges = np.array([[0, 1, 2, 4, 8, 9], [1, 2, 4, 8, 9]], dtype=object)
    bin_centers = _indices_to_bin_centers(bin_edges, indices)
    expected = np.array([[0.5, 3], [3, 6], [8.5, 8.5]])
    assert np.array_equal(bin_centers, expected)
