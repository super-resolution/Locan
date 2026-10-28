from collections.abc import Sequence
from typing import Any

import boost_histogram as bh
import numpy as np
import numpy.typing as npt
import pytest

from locan import Bins, LocData, histogram
from locan.process.aggregate import _accumulate_1d, _accumulate_2d

# --- histogram -------------------------------------------------------------


def numpy_histogram(
    locdata: LocData,
    loc_properties: list[str],
    bins: Bins,
    other_property: str | None = None,
) -> npt.NDArray[np.float64]:
    """Reference implementation with numpy on the same bin edges.

    Points on the last edge are excluded to emulate half-open bins.
    """
    sample = locdata.data[loc_properties].to_numpy(dtype=float)
    edges = [np.asarray(edges) for edges in bins.bin_edges]
    inside = np.all(
        [sample[:, index] < edges_[-1] for index, edges_ in enumerate(edges)], axis=0
    )
    counts, _ = np.histogramdd(sample[inside], bins=edges)
    if other_property is None:
        return counts
    weights = locdata.data[other_property].to_numpy(dtype=float)[inside]
    sums, _ = np.histogramdd(sample[inside], bins=edges, weights=weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        return sums / counts


def test_histogram(locdata_blobs_2d: LocData) -> None:
    hist = histogram(locdata_blobs_2d, n_bins=10)
    assert hist.labels == ["position_x", "position_y", "counts"]
    assert hist.bins.labels == ["position_x", "position_y"]
    assert hist.data.dtype == "float64"
    assert hist.data.ndim == 2
    assert hist.data.shape == (10, 10)
    assert np.max(hist.data) == 7
    assert np.min(hist.data) == 0
    assert hist.data.sum() == len(locdata_blobs_2d)

    hist = histogram(locdata_blobs_2d, n_bins=10, bin_range=((500, 1000), (500, 1000)))
    assert np.max(hist.data) == 5
    assert hist.data.shape == (10, 10)
    assert hist.bins.bin_range == ((500.0, 1000.0), (500.0, 1000.0))

    new_bins = Bins.from_n_bins(10, ((500, 1000), (500, 1000)))
    hist = histogram(locdata_blobs_2d, bins=new_bins)
    assert np.max(hist.data) == 5
    assert hist.data.shape == (10, 10)
    assert new_bins.labels is None

    hist = histogram(locdata_blobs_2d, bin_edges=new_bins.bin_edges)
    assert np.max(hist.data) == 5
    assert hist.data.shape == (10, 10)

    hist = histogram(locdata_blobs_2d, bin_size=10, loc_properties="position_x")
    assert hist.labels == ["position_x", "counts"]
    assert hist.data.shape == (89,)

    hist = histogram(locdata_blobs_2d, bin_size=10, loc_properties=["position_x"])
    assert hist.labels == ["position_x", "counts"]
    assert hist.data.shape == (89,)

    hist = histogram(
        locdata_blobs_2d, bin_size=10, loc_properties=["position_x", "position_y"]
    )
    assert hist.labels == ["position_x", "position_y", "counts"]
    assert hist.data.shape == (89, 55)

    hist = histogram(
        locdata_blobs_2d, bin_size=10, loc_properties=["position_x", "cluster_label"]
    )
    assert hist.labels == ["position_x", "cluster_label", "counts"]
    assert hist.data.shape == (89, 1)

    hist = histogram(
        locdata_blobs_2d,
        bin_edges=((500, 600, 700, 800, 900, 1000), (500, 600, 700, 800, 900, 1000)),
    )
    assert hist.data.ndim == 2
    assert np.max(hist.data) == 7

    hist = histogram(locdata_blobs_2d, bin_size=10, other_property="position_y")
    assert hist.labels == ["position_x", "position_y", "position_y"]
    assert hist.data.shape == (89, 55)
    assert np.isnan(hist.data).any()
    assert np.nanmax(hist.data) == 1001
    assert np.nanmin(hist.data) == 465


@pytest.mark.parametrize(
    "loc_properties",
    ["position_z", ["position_z"], ["position_x", "position_z"]],
)
def test_histogram_invalid_loc_properties(
    locdata_blobs_2d: LocData, loc_properties: Any
) -> None:
    with pytest.raises(ValueError, match="not a valid property"):
        histogram(locdata_blobs_2d, bin_size=10, loc_properties=loc_properties)


def test_histogram_1d(locdata_1d: LocData) -> None:
    hist = histogram(locdata_1d, n_bins=10)
    assert hist.labels == ["position_x", "counts"]
    assert hist.data.dtype == "float64"
    assert hist.data.ndim == 1
    assert np.max(hist.data) == 2
    assert hist.data.shape == (10,)
    assert hist.data.sum() == len(locdata_1d)

    hist = histogram(locdata_1d, n_bins=5, bin_range=(5, 10))
    assert np.max(hist.data) == 1
    assert hist.data.shape == (5,)

    hist = histogram(locdata_1d, bin_edges=(5, 6, 7, 8, 9, 10))
    assert hist.data.shape == (5,)

    hist = histogram(locdata_1d, n_bins=10, other_property="intensity")
    assert hist.labels == ["position_x", "intensity"]
    assert hist.data.shape == (10,)
    assert np.isnan(hist.data).any()
    assert np.nanmax(hist.data) == 125


def test_histogram_3d(locdata_blobs_3d: LocData) -> None:
    hist = histogram(locdata_blobs_3d, n_bins=10)
    assert hist.labels == ["position_x", "position_y", "position_z", "counts"]
    assert hist.data.dtype == "float64"
    assert hist.data.ndim == 3
    assert hist.data.shape == (10, 10, 10)
    assert hist.data.sum() == len(locdata_blobs_3d)

    hist = histogram(
        locdata_blobs_3d, n_bins=10, bin_range=((500, 1000), (500, 1000), (500, 1000))
    )
    assert np.max(hist.data) == 4
    assert hist.data.shape == (10, 10, 10)

    hist = histogram(
        locdata_blobs_3d,
        bin_edges=(
            (500, 600, 700, 800, 900, 1000),
            (500, 600, 700, 800, 900, 1000),
            (500, 600, 700, 800, 900, 1000),
        ),
    )
    assert hist.data.shape == (5, 5, 5)

    hist = histogram(locdata_blobs_3d, n_bins=10, other_property="position_y")
    assert hist.labels == ["position_x", "position_y", "position_z", "position_y"]
    assert hist.data.shape == (10, 10, 10)
    assert np.isnan(hist.data).any()


def test_histogram_empty(locdata_empty: LocData) -> None:
    with pytest.raises(TypeError, match="at least one localization"):
        histogram(locdata_empty, n_bins=10)


def test_histogram_single_value(locdata_single_localization_3d: LocData) -> None:
    hist = histogram(locdata_single_localization_3d, n_bins=3)
    assert hist.data.shape == (3, 3, 3)
    assert hist.data.sum() == 1
    assert np.allclose(hist.bins.bin_range, [[1, 2], [1, 2], [1, 2]])

    hist = histogram(locdata_single_localization_3d, bin_size=0.2)
    assert hist.data.shape == (5, 5, 5)
    assert np.allclose(hist.bins.bin_range, [[1, 2], [1, 2], [1, 2]])

    hist = histogram(locdata_single_localization_3d, bin_size=2)
    assert hist.data.shape == (1, 1, 1)
    assert np.allclose(hist.bins.bin_range, [[1, 2], [1, 2], [1, 2]])


def test_histogram_2d_negative_values(locdata_2d_negative: LocData) -> None:
    hist = histogram(locdata_2d_negative, n_bins=10)
    assert hist.labels == ["position_x", "position_y", "counts"]
    assert hist.data.shape == (10, 10)
    assert hist.data.sum() == len(locdata_2d_negative)


@pytest.mark.parametrize("other_property", [None, "frame"])
@pytest.mark.parametrize(
    ("fixture_name", "loc_properties", "bin_kwargs"),
    [
        ("locdata_blobs_2d", ["position_x"], {"n_bins": 7}),
        ("locdata_blobs_2d", ["position_x"], {"bin_edges": (0, 150, 300, 1000, 2000)}),
        ("locdata_blobs_2d", ["position_x", "position_y"], {"n_bins": (7, 5)}),
        (
            "locdata_blobs_2d",
            ["position_x", "position_y"],
            {"bin_edges": ((0, 150, 300, 1000, 2000), (0, 500, 700, 2000))},
        ),
        (
            "locdata_blobs_2d",
            ["position_x", "position_y"],
            {
                "bin_size": (300, (100, 200, 400, 800)),
                "bin_range": ((0, 2000), (0, 2000)),
            },
        ),
        ("locdata_blobs_3d", None, {"n_bins": 4}),
        (
            "locdata_blobs_3d",
            None,
            {"bin_edges": ((0, 150, 300, 2000), (0, 500, 2000), (0, 100, 400, 2000))},
        ),
    ],
)
def test_histogram_matches_numpy(
    request: pytest.FixtureRequest,
    fixture_name: str,
    loc_properties: list[str] | None,
    bin_kwargs: dict[str, Any],
    other_property: str | None,
) -> None:
    locdata = request.getfixturevalue(fixture_name)
    hist = histogram(
        locdata,
        loc_properties=loc_properties,
        other_property=other_property,
        **bin_kwargs,
    )
    labels = loc_properties if loc_properties is not None else locdata.coordinate_keys
    expected = numpy_histogram(locdata, labels, hist.bins, other_property)
    assert hist.data.shape == expected.shape
    # boost-histogram accumulates means incrementally: allow rounding noise
    assert np.allclose(hist.data, expected, rtol=1e-12, atol=0, equal_nan=True)


def test_histogram_bin_size_remainder_consistency(locdata_blobs_2d: LocData) -> None:
    hist = histogram(
        locdata_blobs_2d, loc_properties="position_x", bin_size=30, bin_range=(0, 1000)
    )
    assert hist.bins.bin_range == ((0.0, 990.0),)
    assert hist.bins.n_bins == (33,)
    expected = numpy_histogram(locdata_blobs_2d, ["position_x"], hist.bins)
    assert np.array_equal(hist.data, expected)


@pytest.mark.parametrize("bin_range", [None, "zero", "link"])
@pytest.mark.parametrize(
    ("fixture_name", "loc_properties"),
    [
        ("locdata_1d", None),
        ("locdata_blobs_2d", None),
        ("locdata_blobs_2d", "position_x"),
        ("locdata_blobs_3d", None),
    ],
)
def test_histogram_counts_all_points_with_data_range(
    request: pytest.FixtureRequest,
    fixture_name: str,
    loc_properties: Any,
    bin_range: Any,
) -> None:
    locdata = request.getfixturevalue(fixture_name)
    hist = histogram(
        locdata, loc_properties=loc_properties, n_bins=4, bin_range=bin_range
    )
    assert hist.data.sum() == len(locdata)
    # with bin_size the default remainder="drop" cuts the range back to whole
    # bins; all points below the last edge must be counted.
    hist = histogram(
        locdata, loc_properties=loc_properties, bin_size=10, bin_range=bin_range
    )
    assert hist.bins.labels is not None
    sample = locdata.data[hist.bins.labels].to_numpy(dtype=float)
    inside = np.all(
        [
            sample[:, index] < edges[-1]
            for index, edges in enumerate(hist.bins.bin_edges)
        ],
        axis=0,
    )
    assert hist.data.sum() == inside.sum()


def test_histogram_explicit_range_is_half_open(locdata_1d: LocData) -> None:
    # locdata_1d has coordinates 1, 1, 2, 3, 4, 5: the point at 5 is on the edge
    hist = histogram(locdata_1d, n_bins=4, bin_range=(1, 5))
    assert hist.data.sum() == len(locdata_1d) - 1
    hist = histogram(locdata_1d, n_bins=4)
    assert hist.data.sum() == len(locdata_1d)


def test_histogram_bin_range_special(locdata_blobs_2d: LocData) -> None:
    hist = histogram(locdata_blobs_2d, n_bins=4, bin_range="zero")
    assert all(low == 0 for low, _ in hist.bins.bin_range)
    assert hist.bins.bin_range[0][1] == pytest.approx(1007)
    assert hist.bins.bin_range[1][1] == pytest.approx(1016)

    hist = histogram(locdata_blobs_2d, n_bins=4, bin_range="link")
    assert hist.bins.bin_range[0] == hist.bins.bin_range[1]
    assert hist.bins.bin_range[0] == pytest.approx((112, 1016))


def test_histogram_with_bh_axes(locdata_blobs_2d: LocData) -> None:
    axes = bh.axis.AxesTuple((bh.axis.Regular(5, 0, 1100), bh.axis.Regular(5, 0, 1100)))
    hist = histogram(locdata_blobs_2d, bins=axes)
    assert hist.data.shape == (5, 5)
    assert hist.labels == ["position_x", "position_y", "counts"]
    assert hist.bins.labels == ["position_x", "position_y"]
    assert hist.data.sum() == len(locdata_blobs_2d)

    hist_regular = histogram(
        locdata_blobs_2d, bins=bh.axis.Regular(5, 0, 1100), loc_properties="position_x"
    )
    assert hist_regular.data.shape == (5,)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_bins": 5, "bin_size": 2}, ValueError, "Exactly one"),
        ({}, ValueError, "Exactly one"),
        (
            {"bins": Bins.from_n_bins(5, (0, 10)), "bin_range": "zero"},
            ValueError,
            "must be None",
        ),
        (
            {"bin_edges": (0, 1), "bin_range": ((0, 10), (0, 10))},
            ValueError,
            "derived from",
        ),
        ({"bins": Bins.from_n_bins(5, (0, 10))}, ValueError, "does not match"),
        (
            {"bin_size": 10, "bin_range": ((500, 1000), (500, 1000), (500, 1000))},
            ValueError,
            "does not match",
        ),
        ({"n_bins": (5, 5, 5)}, ValueError, "same number of dimensions"),
        ({"n_bins": 5, "other_property": "not_a_property"}, TypeError, "not a valid"),
        (
            {
                "n_bins": 5,
                "loc_properties": [
                    "position_x",
                    "position_y",
                    "frame",
                    "cluster_label",
                ],
            },
            TypeError,
            "1, 2 or 3",
        ),
        ({"bins": (0, 1, 2)}, TypeError, "must be an instance of Bins"),
    ],
)
def test_histogram_errors(
    locdata_blobs_2d: LocData,
    kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        histogram(locdata_blobs_2d, **kwargs)


# --- accumulate ------------------------------------------------------------


def assert_groups_equal(result: Sequence[Any] | None, expected: Sequence[Any]) -> None:
    assert result is not None
    assert len(result) == len(expected)
    for result_, expected_ in zip(result, expected, strict=True):
        assert np.array_equal(result_, expected_)


def test__accumulate_1d() -> None:
    bins = Bins.from_n_bins(5, (0, 10))
    n_bins = bins.n_bins[0] + 2  # including over- and underflow bins

    # no data
    bin_indices, data_indices, collection, counts = _accumulate_1d(
        np.array([]), bins=bins, return_data=True, return_counts=True
    )
    assert np.array_equal(bin_indices, [])
    assert data_indices == collection == []
    assert counts is not None and np.array_equal(counts, [])

    # single data point
    bin_indices, data_indices, collection, counts = _accumulate_1d(
        np.array([9]), bins=bins, return_data=True, return_counts=True
    )
    assert np.array_equal(bin_indices, [5])
    assert_groups_equal(data_indices, [np.array([0])])
    assert_groups_equal(collection, [np.array([9])])
    assert counts is not None and np.array_equal(counts, [1])

    # return_data=False
    data = np.array([1, -1, 11, 0.1, 6, 2, 1, 2.0])
    bin_indices, data_indices, collection, counts = _accumulate_1d(data, bins=bins)
    assert np.array_equal(bin_indices, [0, 1, 2, 4, 6])
    assert_groups_equal(data_indices, [[1], [0, 3, 6], [5, 7], [4], [2]])
    assert collection is None
    assert counts is None

    bin_indices, data_indices, collection, counts = _accumulate_1d(
        data, bins=bins, return_data=True, return_counts=True
    )
    assert np.array_equal(bin_indices, [0, 1, 2, 4, 6])
    assert_groups_equal(data_indices, [[1], [0, 3, 6], [5, 7], [4], [2]])
    assert_groups_equal(collection, [[-1], [1, 0.1, 1], [2, 2], [6], [11]])
    assert counts is not None and np.array_equal(counts, [1, 3, 2, 1, 1])

    # large data
    n_samples = 1_000
    data = np.random.default_rng(seed=1).normal(loc=5, scale=5, size=n_samples)
    bin_indices, data_indices, collection, counts = _accumulate_1d(
        data, bins=bins, return_data=True, return_counts=True
    )
    assert collection is not None and counts is not None
    assert (
        len(bin_indices)
        == len(data_indices)
        == len(collection)
        == len(counts)
        == n_bins
    )
    assert sum(counts) == n_samples

    # two dimensions: binned by the first dimension
    data = np.array(
        [(1, 2), (-1, 5), (11, -1), (0.1, 11), (6, 5), (2, 5), (1, 2.1), (2.0, 2)]
    )
    bin_indices, data_indices, collection, counts = _accumulate_1d(
        data, bins=bins, return_data=True, return_counts=True
    )
    assert np.array_equal(bin_indices, [0, 1, 2, 4, 6])
    assert_groups_equal(data_indices, [[1], [0, 3, 6], [5, 7], [4], [2]])
    assert_groups_equal(
        collection,
        [
            [(-1, 5)],
            [(1, 2), (0.1, 11), (1, 2.1)],
            [(2, 5), (2, 2)],
            [(6, 5)],
            [(11, -1)],
        ],
    )
    assert counts is not None and np.array_equal(counts, [1, 3, 2, 1, 1])


def test__accumulate_2d() -> None:
    bins = Bins.from_n_bins(5, ((0, 10), (0, 10)))
    n_bins = bins.n_bins[0] + 2  # including over- and underflow bins

    # no data
    bin_indices, data_indices, collection, counts = _accumulate_2d(
        data=[], bins=bins, return_data=True, return_counts=True
    )
    assert bin_indices.shape == (0, 2)
    assert data_indices == collection == []
    assert counts is not None and np.array_equal(counts, [])

    # single data point
    bin_indices, data_indices, collection, counts = _accumulate_2d(
        data=[(1, 1)], bins=bins, return_data=True, return_counts=True
    )
    assert np.array_equal(bin_indices, [[1, 1]])
    assert np.array_equal(data_indices, [[0]])
    assert collection is not None and np.array_equal(collection, [[(1, 1)]])
    assert counts is not None and np.array_equal(counts, [1])

    # return_data=False
    data = np.array([(1, 2), (1, 5), (4, 5), (1, 2)])
    bin_indices, data_indices, collection, counts = _accumulate_2d(data=data, bins=bins)
    assert np.array_equal(bin_indices, [[1, 2], [1, 3], [3, 3]])
    assert_groups_equal(data_indices, [[0, 3], [1], [2]])
    assert collection is None
    assert counts is None

    # including over- and underflow bins
    data = np.array([(1, 2), (-1, 5), (-1, -1), (2, 11), (1, 5), (4, 5), (1.1, 2.1)])
    bin_indices, data_indices, collection, counts = _accumulate_2d(
        data=data, bins=bins, return_data=True, return_counts=True
    )
    assert np.array_equal(bin_indices, [[0, 0], [0, 3], [1, 2], [1, 3], [2, 6], [3, 3]])
    assert_groups_equal(data_indices, [[2], [1], [0, 6], [4], [3], [5]])
    assert_groups_equal(
        collection,
        [[[-1, -1]], [[-1, 5]], [[1, 2], [1.1, 2.1]], [[1, 5]], [[2, 11]], [[4, 5]]],
    )
    assert counts is not None and np.array_equal(counts, [1, 1, 2, 1, 1, 1])

    # large data
    n_samples = 1_000
    data = np.random.default_rng(seed=1).normal(loc=5, scale=5, size=(n_samples, 2))
    bin_indices, data_indices, collection, counts = _accumulate_2d(
        data, bins=bins, return_data=True, return_counts=True
    )
    assert collection is not None and counts is not None
    assert (
        len(bin_indices)
        == len(data_indices)
        == len(collection)
        == len(counts)
        == n_bins**2
    )
    assert sum(counts) == n_samples

    # three dimensions: binned by the first two dimensions
    data = np.array(
        [
            (1, 2, 1),
            (-1, 5, 1),
            (-1, -1, 1),
            (2, 11, 1),
            (1, 5, 1),
            (4, 5, 1),
            (1.1, 2.1, 1),
        ]
    )
    bin_indices, data_indices, collection, counts = _accumulate_2d(
        data, bins=bins, return_data=True, return_counts=True
    )
    assert np.array_equal(bin_indices, [[0, 0], [0, 3], [1, 2], [1, 3], [2, 6], [3, 3]])
    assert_groups_equal(data_indices, [[2], [1], [0, 6], [4], [3], [5]])
    assert_groups_equal(
        collection,
        [
            [[-1, -1, 1]],
            [[-1, 5, 1]],
            [[1, 2, 1], [1.1, 2.1, 1]],
            [[1, 5, 1]],
            [[2, 11, 1]],
            [[4, 5, 1]],
        ],
    )
    assert counts is not None and np.array_equal(counts, [1, 1, 2, 1, 1, 1])
