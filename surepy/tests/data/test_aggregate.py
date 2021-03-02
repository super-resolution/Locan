import pytest
import numpy as np

from surepy.constants import _has_boost_histogram
if _has_boost_histogram: import boost_histogram as bh
from surepy.data.aggregate import _n_bins_to_bin_edges_one_dimension, _bin_size_to_bin_edges_one_dimension, \
    _bin_edges_to_n_bins_one_dimension, _bin_edges_to_n_bins, \
    _bin_edges_to_bin_size_one_dimension, _bin_edges_to_bin_size, \
    _bin_edges_to_bin_centers, _indices_to_bin_centers, \
    _BinsFromBoostHistogramAxis, _BinsFromEdges, _BinsFromNumber, _BinsFromSize
from surepy import adjust_contrast, histogram, Bins


def test__n_bins_to_bin_edges_one_dimension():
    bin_edges = _n_bins_to_bin_edges_one_dimension(10, (10, 20))
    assert bin_edges.shape == (11,)


def test__bin_size_to_bin_edges_one_dimension():
    bin_edges = _bin_size_to_bin_edges_one_dimension(4, (0, 10), extend_range=None)
    assert np.array_equal(bin_edges, (0, 4, 8))
    bin_edges = _bin_size_to_bin_edges_one_dimension(4, (0, 10), extend_range=True)
    assert np.array_equal(bin_edges, (0, 4, 8, 12))
    bin_edges = _bin_size_to_bin_edges_one_dimension(4, (0, 10), extend_range=False)
    assert np.array_equal(bin_edges, (0, 4, 8, 10))

    bin_edges = _bin_size_to_bin_edges_one_dimension((1, 2, 3, 3, 2), (0, 10), extend_range=None)
    assert np.array_equal(bin_edges, (0, 1, 3, 6, 9))
    bin_edges = _bin_size_to_bin_edges_one_dimension((1, 2, 3, 3, 2), (0, 10), extend_range=True)
    assert np.array_equal(bin_edges, (0, 1, 3, 6, 9, 11))
    bin_edges = _bin_size_to_bin_edges_one_dimension((1, 2, 3, 3, 2), (0, 10), extend_range=False)
    assert np.array_equal(bin_edges, (0, 1, 3, 6, 9, 10))
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
    assert bin_size == (1, 2)
    bin_size = _bin_edges_to_bin_size_one_dimension([1, 2])
    assert bin_size == 1


def test__bin_edges_to_bin_size():
    bin_size = _bin_edges_to_bin_size([1, 3, 5])
    assert bin_size == (2,)

    bin_size = _bin_edges_to_bin_size(([1, 3, 5],))
    assert bin_size == (2,)

    bin_size = _bin_edges_to_bin_size([1, 2, 4])
    assert bin_size == ((1, 2),)

    bin_size = _bin_edges_to_bin_size(((1, 3, 5), (1, 2, 4, 5)))
    assert bin_size == (2, (1, 2, 1))

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


@pytest.mark.skipif(not _has_boost_histogram, reason="Test requires boost_histogram.")
def test__BinsFromBoostHistogramAxis():
    bhaxis = bh.axis.Regular(5, 0, 10)
    bins = _BinsFromBoostHistogramAxis(bins=bhaxis)
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == ((2, 2, 2, 2, 2),)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))

    bhaxes = bh.axis.AxesTuple((bh.axis.Regular(5, 0, 10), bh.axis.Regular(2, 0, 10)))
    bins = _BinsFromBoostHistogramAxis(bins=bhaxes)
    assert bins.n_dimensions == 2
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
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 4.0),)
    assert bins.n_bins == (2,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4]))

    bins = _BinsFromEdges(bin_edges=((0, 2, 4),))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 4.0),)
    assert bins.n_bins == (2,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4]))

    bins = _BinsFromEdges(bin_edges=(1, 2, 5))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((1, 5),)
    assert bins.n_bins == (2,)
    assert bins.bin_size == ((1, 3),)
    assert np.array_equal(bins.bin_edges[0], np.array([1, 2, 5]))

    bins = _BinsFromEdges(bin_edges=((0, 2, 4), (1, 2, 5)))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0.0, 4.0), (1.0, 5.0))
    assert bins.n_bins == (2, 2)
    assert bins.bin_size == (2, (1, 3))
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
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))

    bins = _BinsFromNumber(n_bins=(5,), bin_range=(0, 10))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))

    bins = _BinsFromNumber(n_bins=(2, 5), bin_range=((0, 10), (0, 5)))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0, 10), (0, 5))
    assert bins.n_bins == (2, 5)
    assert bins.bin_size == (5, 1)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 5, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 2, 3, 4, 5]))

    bins = _BinsFromNumber(n_bins=2, bin_range=((0, 10), (0, 5)))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0, 10), (0, 5))
    assert bins.n_bins == (2, 2)
    assert bins.bin_size == (5, 2.5)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 5, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 2.5, 5]))

    bins = _BinsFromNumber(n_bins=(2, 5, 2), bin_range=(0, 10))
    assert bins.n_dimensions == 3
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


def test__BinsFromSize():
    bins = _BinsFromSize(bin_size=2, bin_range=(0, 10))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))

    bins = _BinsFromSize(bin_size=(2,), bin_range=(0, 10))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))

    bins = _BinsFromSize(bin_size=((1, 2, 3, 4),), bin_range=(0, 10))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (4,)
    assert bins.bin_size == ((1, 2, 3, 4),)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 1, 3, 6, 10]))

    bins = _BinsFromSize(bin_size=3, bin_range=(0, 10), extend_range=False)
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (4,)
    assert bins.bin_size == ((3.0, 3.0, 3.0, 1.0),)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 3, 6, 9, 10]))

    bins = _BinsFromSize(bin_size=(5, 1), bin_range=((0, 10), (0, 5)))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0, 10), (0, 5))
    assert bins.n_bins == (2, 5)
    assert bins.bin_size == (5, 1)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 5, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 2, 3, 4, 5]))

    bins = _BinsFromSize(bin_size=2, bin_range=((0, 10), (0, 5)))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0, 10), (0, 5))
    assert bins.n_bins == (5, 2)
    assert bins.bin_size == (2, 2)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 2, 4]))

    bins = _BinsFromSize(bin_size=(2, 5), bin_range=(0, 10))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0, 10), (0, 10))
    assert bins.n_bins == (5, 2)
    assert bins.bin_size == (2, 5)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 5, 10]))

    bins = _BinsFromSize(bin_size=((1, 2, 3, 4), (1, 2, 3, 4)), bin_range=(0, 10))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 10.0))
    assert bins.n_bins == (4, 4)
    assert bins.bin_size == ((1, 2, 3, 4), (1, 2, 3, 4))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 1, 3, 6, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 3, 6, 10]))

    bins = _BinsFromSize(bin_size=((1, 2, 3, 4), (1, 2, 3, 4)), bin_range=((0, 10), (0, 10)))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 10.0))
    assert bins.n_bins == (4, 4)
    assert bins.bin_size == ((1, 2, 3, 4), (1, 2, 3, 4))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 1, 3, 6, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 3, 6, 10]))

    bins = _BinsFromSize(bin_size=(2, (1, 2, 3, 4)), bin_range=(0, 10))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 10.0))
    assert bins.n_bins == (5, 4)
    assert bins.bin_size == (2, (1, 2, 3, 4))
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_edges[1], np.array([0, 1, 3, 6, 10]))

    with pytest.raises(TypeError):
        _BinsFromSize(bin_size=5, bin_range=(0,))
    with pytest.raises(TypeError):
        _BinsFromSize(bin_size=5, bin_range=(1, 2, 3))


def test_Bins():
    bins = Bins(bin_edges=(0, 2, 4))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0, 4),)
    assert bins.n_bins == (2,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3]))
    assert bins.is_equally_sized == (True,)

    bins = Bins(n_bins=5, bin_range=(0, 10))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))
    assert bins.is_equally_sized == (True,)

    bins = Bins(bin_size=2, bin_range=(0, 10))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))
    assert bins.is_equally_sized == (True,)

    bins = Bins(bin_size=(2, (1, 2, 3)), bin_range=(0, 10))
    assert bins.n_dimensions == 2
    assert bins.bin_range == ((0.0, 10.0), (0.0, 10.0))
    assert bins.n_bins == (5, 3)
    assert bins.bin_size == ((2,), (1, 2, 3))
    assert bins.is_equally_sized == (True, False)

    bins = Bins(bins=Bins(n_bins=5, bin_range=(0, 10)))
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == (2,)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))
    assert bins.is_equally_sized == (True,)

    bins = Bins(n_bins=(2, 5), bin_range=(0, 10), labels=["position_x", "position_y"])
    assert bins.labels == ["position_x", "position_y"]
    assert bins.n_dimensions == 2
    assert bins.is_equally_sized == (True, True)
    bins = Bins(bins=Bins(n_bins=5, bin_range=(0, 10)), labels=["position_x"])
    assert bins.labels == ["position_x"]
    assert bins.n_dimensions == 1
    bins = Bins(bins=Bins(n_bins=5, bin_range=(0, 10), labels=["position_x"]))
    assert bins.labels == ["position_x"]
    assert bins.n_dimensions == 1
    with pytest.raises(ValueError):
        Bins(n_bins=5, bin_range=(0, 10), labels=["position_x", "position_y"])


@pytest.mark.skipif(not _has_boost_histogram, reason="Test requires boost_histogram.")
def test_Bins_with_boost_histogram():
    bhaxis = bh.axis.Regular(5, 0, 10)
    bins = Bins(bins=bhaxis)
    assert bins.n_dimensions == 1
    assert bins.bin_range == ((0.0, 10.0),)
    assert bins.n_bins == (5,)
    assert bins.bin_size == ((2, 2, 2, 2, 2),)
    assert np.array_equal(bins.bin_edges[0], np.array([0, 2, 4, 6, 8, 10]))
    assert np.array_equal(bins.bin_centers[0], np.array([1, 3, 5, 7, 9]))
    assert bins.is_equally_sized == (True,)


def test_Bins_methods():
    # bins = Bins(bin_edges=(0, 1, 2, 4))
    # assert bins.n_dimensions == 1
    # assert bins.bin_range == ((0, 4),)
    # assert bins.n_bins == (3,)
    # assert bins.bin_size == ((1, 1, 2),)
    # assert np.array_equal(bins.bin_edges[0], np.array([0, 1, 2, 4]))
    # assert bins.is_equally_sized == (False,)

    bins = Bins(bin_edges=(0, 1, 2, 4)).equalize_bin_size()
    assert bins.n_dimensions == 1
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
    assert 'counts' in hist.labels
    assert hist.data.dtype == 'float64'
    assert hist.data.ndim == 2
    assert np.max(hist.data) == 7

    hist = histogram(locdata_blobs_2d, n_bins=10, bin_range=((500, 1000), (500, 1000)))
    assert hist.data.ndim == 2
    assert np.max(hist.data) == 5

    # todo: fix the following
    # hist = histogram(locdata_blobs_2d, n_bins=10, bin_range=(500, 1000))

    hist = histogram(locdata_blobs_2d, bin_edges=((500, 600, 700, 800, 900, 1000), (500, 600, 700, 800, 900, 1000)))
    assert hist.data.ndim == 2
    assert np.max(hist.data) == 7

    # todo: fix the following
    # hist = histogram(locdata_blobs_2d, bin_edges=(500, 600, 700, 800, 900, 1000))

    hist = histogram(locdata_blobs_2d, bin_size=10, bin_range='zero', rescale=True)
    assert np.max(hist.data) == 1
    assert 'counts' in hist.labels

    hist = histogram(locdata_blobs_2d, bin_size=10, loc_properties='position_x')
    assert 'counts' in hist.labels
    assert hist.data.shape == (89,)

    with pytest.raises(TypeError):
        histogram(locdata_blobs_2d, bin_size=10, loc_properties=['position_x'])

    hist = histogram(locdata_blobs_2d, bin_size=10, loc_properties=['position_x', 'position_y'])
    assert 'counts' in hist.labels
    assert hist.data.shape == (55, 89)

    hist = histogram(locdata_blobs_2d, bin_size=10, other_property='position_y')
    assert 'position_y' in hist.labels
    assert hist.data.shape == (55, 89)


def test_histogram_empty(locdata_empty):
    with pytest.raises(TypeError):
        hist = histogram(locdata_empty, n_bins=10)


def test_histogram_single_value(locdata_single_localization):
    hist = histogram(locdata_single_localization, n_bins=3)
    assert hist.data.shape == (3, 3)
    assert np.array_equal(hist.bins.bin_range, [[1, 2], [1, 2]])

    hist = histogram(locdata_single_localization, bin_size=0.2)
    assert hist.data.shape == (5, 5)
    assert np.array_equal(hist.bins.bin_range, [[1, 2], [1, 2]])

    # todo: fix
    # hist = histogram(locdata_single_localization, bin_size=2)
