import copy
import pickle
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import boost_histogram as bh
import numpy as np
import numpy.typing as npt
import pytest

from locan import Bins
from locan.process.bins import _indices_to_bin_centers


def assert_edges_equal(
    bins: Bins, expected: Sequence[Sequence[float]], exact: bool = True
) -> None:
    """Assert bin edges per dimension and that derived ranges match the edges."""
    assert bins.dimension == len(expected)
    assert len(bins.bin_edges) == len(expected)
    for edges, expected_edges in zip(bins.bin_edges, expected, strict=True):
        if exact:
            assert np.array_equal(edges, expected_edges)
        else:
            assert np.allclose(edges, expected_edges)
        assert edges.dtype == np.float64
        assert edges.flags.writeable is False
    assert bins.bin_range == tuple(
        (float(edges[0]), float(edges[-1])) for edges in bins.bin_edges
    )
    assert bins.n_bins == tuple(len(edges) - 1 for edges in bins.bin_edges)


# --- constructors ----------------------------------------------------------


@pytest.mark.parametrize(
    ("bin_edges", "expected_edges", "expected_sizes", "expected_equal"),
    [
        ((0, 2, 4), [[0, 2, 4]], [2.0], (True,)),
        (((0, 2, 4),), [[0, 2, 4]], [2.0], (True,)),
        ([0, 2, 4], [[0, 2, 4]], [2.0], (True,)),
        (np.array([0, 2, 4]), [[0, 2, 4]], [2.0], (True,)),
        ((1, 2, 5), [[1, 2, 5]], [np.array([1.0, 3.0])], (False,)),
        (
            ((0, 2, 4), (1, 2, 5)),
            [[0, 2, 4], [1, 2, 5]],
            [2.0, np.array([1.0, 3.0])],
            (True, False),
        ),
        (
            np.array([[1, 3, 5], [1, 2, 3]]),
            [[1, 3, 5], [1, 2, 3]],
            [2.0, 1.0],
            (True, True),
        ),
        (
            [[0, 1, 2, 4, 8, 9], [0, 1, 4, 8]],
            [[0, 1, 2, 4, 8, 9], [0, 1, 4, 8]],
            [np.array([1, 1, 2, 4, 1]), np.array([1, 3, 4])],
            (False, False),
        ),
    ],
)
def test_from_bin_edges(
    bin_edges: Any,
    expected_edges: list[list[float]],
    expected_sizes: list[float | npt.NDArray[np.float64]],
    expected_equal: tuple[bool, ...],
) -> None:
    bins = Bins.from_bin_edges(bin_edges)
    assert_edges_equal(bins, expected_edges)
    assert bins.is_equally_sized == expected_equal
    for size, expected_size in zip(bins.bin_size, expected_sizes, strict=True):
        if isinstance(expected_size, np.ndarray):
            assert np.array_equal(size, expected_size)
        else:
            assert isinstance(size, float)
            assert size == expected_size
    for centers, edges in zip(bins.bin_centers, expected_edges, strict=True):
        edges_ = np.asarray(edges, dtype=float)
        assert np.array_equal(centers, (edges_[:-1] + edges_[1:]) / 2)
    assert bins.labels is None


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (None, None),
        ("x", ["x"]),
        (["x"], ["x"]),
        (("x",), ["x"]),
    ],
)
def test_from_bin_edges_labels(labels: Any, expected: list[str] | None) -> None:
    bins = Bins.from_bin_edges((0, 1, 2), labels=labels)
    assert bins.labels == expected


@pytest.mark.parametrize(
    ("labels", "error", "match"),
    [
        (["x", "y"], ValueError, "one entry per dimension"),
        ([], ValueError, "one entry per dimension"),
        ([1], TypeError, "must be str"),
        (1, TypeError, "must be str"),
    ],
)
def test_from_bin_edges_invalid_labels(
    labels: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Bins.from_bin_edges((0, 1, 2), labels=labels)


@pytest.mark.parametrize(
    ("n_bins", "bin_range", "expected_edges", "expected_sizes"),
    [
        (5, (0, 10), [[0, 2, 4, 6, 8, 10]], (2.0,)),
        ((5,), (0, 10), [[0, 2, 4, 6, 8, 10]], (2.0,)),
        (np.int64(5), np.array([0, 10]), [[0, 2, 4, 6, 8, 10]], (2.0,)),
        ((2, 5), ((0, 10), (0, 5)), [[0, 5, 10], [0, 1, 2, 3, 4, 5]], (5.0, 1.0)),
        (2, ((0, 10), (0, 5)), [[0, 5, 10], [0, 2.5, 5]], (5.0, 2.5)),
        (
            (2, 5, 2),
            (0, 10),
            [[0, 5, 10], [0, 2, 4, 6, 8, 10], [0, 5, 10]],
            (5.0, 2.0, 5.0),
        ),
        (2, [[0, 10], [0, 5]], [[0, 5, 10], [0, 2.5, 5]], (5.0, 2.5)),
    ],
)
def test_from_n_bins(
    n_bins: Any,
    bin_range: Any,
    expected_edges: list[list[float]],
    expected_sizes: tuple[float, ...],
) -> None:
    bins = Bins.from_n_bins(n_bins, bin_range)
    assert_edges_equal(bins, expected_edges)
    assert bins.bin_size == expected_sizes
    assert all(isinstance(size, float) for size in bins.bin_size)
    assert all(isinstance(value, float) for pair in bins.bin_range for value in pair)
    assert all(bins.is_equally_sized)


@pytest.mark.parametrize(
    ("n_bins", "bin_range", "error", "match"),
    [
        (5, 1, TypeError, "bin_range"),
        (5, (0,), TypeError, "bin_range"),
        (5, (1, 2, 3), TypeError, "bin_range"),
        ((5, (1, 2)), (1, 2), TypeError, "n_bins"),
        ((2,), ((0, 10), (0, 5)), ValueError, "same number of dimensions"),
        ((2, 5, 2), ((0, 10), (0, 5)), ValueError, "same number of dimensions"),
        (0, (0, 10), ValueError, "at least 1"),
        (-2, (0, 10), ValueError, "at least 1"),
        ((5, 0), (0, 10), ValueError, "at least 1"),
        (2.5, (0, 10), TypeError, "integer"),
        (True, (0, 10), TypeError, "integer"),
        ((), (0, 10), ValueError, "must not be empty"),
        (5, (10, 0), ValueError, "min < max"),
        (5, (0, 0), ValueError, "min < max"),
        (5, (0, np.inf), ValueError, "finite"),
    ],
)
def test_from_n_bins_invalid(
    n_bins: Any, bin_range: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Bins.from_n_bins(n_bins, bin_range)


@pytest.mark.parametrize(
    ("bin_size", "bin_range", "remainder", "expected_edges"),
    [
        (2, (0, 10), "drop", [[0, 2, 4, 6, 8, 10]]),
        ((2,), (0, 10), "drop", [[0, 2, 4, 6, 8, 10]]),
        (np.float64(2), np.array([0, 10]), "drop", [[0, 2, 4, 6, 8, 10]]),
        (((1, 2, 3, 4),), (0, 10), "drop", [[0, 1, 3, 6, 10]]),
        # bin_size does not divide the range
        (3, (0, 10), "drop", [[0, 3, 6, 9]]),
        (3, (0, 10), "keep", [[0, 3, 6, 9, 10]]),
        (3, (0, 10), "extend", [[0, 3, 6, 9, 12]]),
        (4, (-1, 10), "drop", [[-1, 3, 7]]),
        # bin_size exceeds the range
        (20, (1, 10), "drop", [[1, 10]]),
        (20, (1, 10), "keep", [[1, 10]]),
        (20, (1, 10), "extend", [[1, 21]]),
        # variable sizes
        (((1, 2, 3, 3, 2),), (1, 11), "drop", [[1, 2, 4, 7, 10]]),
        (((1, 2, 3, 3, 2),), (1, 11), "keep", [[1, 2, 4, 7, 10, 11]]),
        (((1, 2, 3, 3, 2),), (1, 11), "extend", [[1, 2, 4, 7, 10, 12]]),
        (((10, 20, 30),), (1, 2), "drop", [[1, 2]]),
        (((10, 20, 30),), (1, 2), "keep", [[1, 2]]),
        (((10, 20, 30),), (1, 2), "extend", [[1, 11]]),
        # several dimensions
        ((5, 1), ((0, 10), (0, 5)), "drop", [[0, 5, 10], [0, 1, 2, 3, 4, 5]]),
        (2, ((0, 10), (0, 5)), "drop", [[0, 2, 4, 6, 8, 10], [0, 2, 4]]),
        (2, ((0, 10), (0, 5)), "keep", [[0, 2, 4, 6, 8, 10], [0, 2, 4, 5]]),
        (2, ((0, 10), (0, 5)), "extend", [[0, 2, 4, 6, 8, 10], [0, 2, 4, 6]]),
        ((2, 5), (0, 10), "drop", [[0, 2, 4, 6, 8, 10], [0, 5, 10]]),
        (
            ((1, 2, 3, 4), (1, 2, 3, 1)),
            (0, 10),
            "drop",
            [[0, 1, 3, 6, 10], [0, 1, 3, 6, 7]],
        ),
        (
            ((1, 2, 3, 4), (1, 2, 3, 4)),
            ((0, 10), (0, 10)),
            "drop",
            [[0, 1, 3, 6, 10], [0, 1, 3, 6, 10]],
        ),
        ((2, (1, 2, 3, 4)), (0, 10), "drop", [[0, 2, 4, 6, 8, 10], [0, 1, 3, 6, 10]]),
        (
            (2, (1, 2, 3, 4)),
            ((0, 10), (0, 20)),
            "drop",
            [[0, 2, 4, 6, 8, 10], [0, 1, 3, 6, 10]],
        ),
    ],
)
def test_from_bin_size(
    bin_size: Any,
    bin_range: Any,
    remainder: Literal["drop", "keep", "extend"],
    expected_edges: list[list[float]],
) -> None:
    bins = Bins.from_bin_size(bin_size, bin_range, remainder=remainder)
    assert_edges_equal(bins, expected_edges)
    # bin_range is always derived from the edges (regression: stale bin_range)
    for (low, high), edges in zip(bins.bin_range, bins.bin_edges, strict=True):
        assert (low, high) == (edges[0], edges[-1])
    for axis, edges in zip(bins.to_boost_histogram_axes(), bins.bin_edges, strict=True):
        assert np.allclose(axis.edges, edges)


def test_from_bin_size_default_remainder_is_drop() -> None:
    assert Bins.from_bin_size(3, (0, 10)) == Bins.from_bin_size(
        3, (0, 10), remainder="drop"
    )


@pytest.mark.parametrize(
    ("bin_size", "bin_range", "expected_n_bins", "expected_last_edge"),
    [
        (0.1, (0, 1), 10, 1.0),
        (0.1, (0, 0.7), 7, 0.7),
        (0.3, (0, 0.9), 3, 0.9),
        (0.1, (0.2, 0.5), 3, 0.5),
    ],
)
def test_from_bin_size_float_step(
    bin_size: float,
    bin_range: tuple[float, float],
    expected_n_bins: int,
    expected_last_edge: float,
) -> None:
    bins = Bins.from_bin_size(bin_size, bin_range)
    assert bins.n_bins == (expected_n_bins,)
    assert bins.bin_edges[0][-1] == expected_last_edge
    assert bins.is_equally_sized == (True,)


@pytest.mark.parametrize(
    ("bin_size", "bin_range", "remainder", "error", "match"),
    [
        (5, (0,), "drop", TypeError, "bin_range"),
        (5, (1, 2, 3), "drop", TypeError, "bin_range"),
        (5, 1, "drop", TypeError, "bin_range"),
        ((2, 2, (1, 2, 3, 4)), ((0, 10), (0, 20)), "drop", ValueError, "same number"),
        (
            ((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)),
            ((0, 10), (0, 20)),
            "drop",
            ValueError,
            "same number",
        ),
        (((1, 2), (1, 2)), (1, 2, 3), "drop", TypeError, "bin_range"),
        ((((1, 2),),), (0, 10), "drop", TypeError, "0-, 1- or 2-dimensional"),
        (0, (0, 10), "drop", ValueError, "positive"),
        (-1, (0, 10), "drop", ValueError, "positive"),
        ((2, 0), (0, 10), "drop", ValueError, "positive"),
        (((1, -2),), (0, 10), "drop", ValueError, "positive"),
        (np.inf, (0, 10), "drop", ValueError, "positive"),
        ((), (0, 10), "drop", ValueError, "must not be empty"),
        (2, (10, 0), "drop", ValueError, "min < max"),
        (2, (0, 10), "foo", ValueError, "remainder"),
        (2, (0, 10), None, ValueError, "remainder"),
    ],
)
def test_from_bin_size_invalid(
    bin_size: Any,
    bin_range: Any,
    remainder: Any,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        Bins.from_bin_size(bin_size, bin_range, remainder=remainder)


def test_from_boost_histogram_axes() -> None:
    bins = Bins.from_boost_histogram_axes(bh.axis.Regular(5, 0, 10))
    assert_edges_equal(bins, [[0, 2, 4, 6, 8, 10]])
    assert bins.bin_size == (2.0,)
    assert isinstance(bins.bin_size[0], float)
    assert np.array_equal(bins.bin_centers[0], [1, 3, 5, 7, 9])
    assert bins.labels is None

    axes = bh.axis.AxesTuple((bh.axis.Regular(5, 0, 10), bh.axis.Regular(2, 0, 10)))
    bins = Bins.from_boost_histogram_axes(axes)
    assert_edges_equal(bins, [[0, 2, 4, 6, 8, 10], [0, 5, 10]])
    assert bins.bin_size == (2.0, 5.0)
    assert bins.is_equally_sized == (True, True)

    bins = Bins.from_boost_histogram_axes(
        [bh.axis.Regular(5, 0, 10), bh.axis.Variable([0, 1, 3, 10])]
    )
    assert_edges_equal(bins, [[0, 2, 4, 6, 8, 10], [0, 1, 3, 10]])
    assert bins.is_equally_sized == (True, False)
    assert np.array_equal(bins.bin_size[1], [1, 2, 7])

    bins = Bins.from_boost_histogram_axes(bh.axis.Integer(0, 5))
    assert_edges_equal(bins, [[0, 1, 2, 3, 4, 5]])


def test_from_boost_histogram_axes_labels() -> None:
    axes = bh.axis.AxesTuple(
        (
            bh.axis.Regular(5, 0, 10, metadata="x"),
            bh.axis.Regular(2, 0, 10, metadata="y"),
        )
    )
    assert Bins.from_boost_histogram_axes(axes).labels == ["x", "y"]
    assert Bins.from_boost_histogram_axes(axes, labels=["a", "b"]).labels == ["a", "b"]
    # metadata that is not a string for every axis is ignored
    axes = bh.axis.AxesTuple(
        (bh.axis.Regular(5, 0, 10, metadata="x"), bh.axis.Regular(2, 0, 10))
    )
    assert Bins.from_boost_histogram_axes(axes).labels is None


def test_from_boost_histogram_axes_transform_round_trips_as_variable() -> None:
    axis = bh.axis.Regular(5, 1, 100, transform=bh.axis.transform.log)
    bins = Bins.from_boost_histogram_axes(axis)
    assert bins.is_equally_sized == (False,)
    assert np.array_equal(bins.bin_edges[0], axis.edges)
    (new_axis,) = bins.to_boost_histogram_axes()
    assert isinstance(new_axis, bh.axis.Variable)
    assert np.array_equal(new_axis.edges, axis.edges)


@pytest.mark.parametrize(
    ("axes", "error", "match"),
    [
        (bh.axis.Regular(5, 0, 10, growth=True), ValueError, "fixed, ordered"),
        (bh.axis.Regular(5, 0, 10, circular=True), ValueError, "fixed, ordered"),
        (bh.axis.StrCategory(["a", "b"]), ValueError, "fixed, ordered"),
        ((0, 1, 2), TypeError, "boost_histogram axis"),
        ([], TypeError, "boost_histogram axis"),
        (5, TypeError, "boost_histogram axis"),
    ],
)
def test_from_boost_histogram_axes_invalid(
    axes: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Bins.from_boost_histogram_axes(axes)


def test_from_any() -> None:
    expected = Bins.from_n_bins(5, (0, 10))
    assert Bins.from_any(n_bins=5, bin_range=(0, 10)) == expected
    assert Bins.from_any(bin_size=2, bin_range=(0, 10)) == expected
    assert Bins.from_any(bin_edges=[0, 2, 4, 6, 8, 10]) == expected
    assert Bins.from_any(bins=bh.axis.Regular(5, 0, 10)) == expected
    assert Bins.from_any(bins=expected) is expected

    relabelled = Bins.from_any(bins=expected, labels=["x"])
    assert relabelled.labels == ["x"]
    assert expected.labels is None
    assert np.array_equal(relabelled.bin_edges[0], expected.bin_edges[0])

    assert Bins.from_any(bin_size=3, bin_range=(0, 10)) == Bins.from_bin_size(
        3, (0, 10), remainder="drop"
    )
    assert Bins.from_any(
        bin_size=3, bin_range=(0, 10), remainder="extend"
    ) == Bins.from_bin_size(3, (0, 10), remainder="extend")
    assert Bins.from_any(n_bins=5, bin_range=(0, 10), labels="x").labels == ["x"]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({}, ValueError, "Exactly one"),
        ({"n_bins": 5, "bin_size": 2, "bin_range": (0, 10)}, ValueError, "Exactly one"),
        ({"n_bins": 5}, ValueError, "must be given together with `n_bins`"),
        ({"bin_size": 5}, ValueError, "must be given together with `bin_size`"),
        ({"bin_edges": (0, 1), "bin_range": (0, 1)}, ValueError, "derived from"),
        (
            {"bins": Bins.from_n_bins(5, (0, 10)), "bin_range": (0, 10)},
            ValueError,
            "derived from",
        ),
        ({"bins": (0, 1, 2)}, TypeError, "must be an instance of Bins"),
        ({"bins": "bins"}, TypeError, "must be an instance of Bins"),
    ],
)
def test_from_any_invalid(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Bins.from_any(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"n_bins": 5, "bin_range": (0, 10)}, Bins.from_n_bins(5, (0, 10))),
        ({"bin_size": 2, "bin_range": (0, 10)}, Bins.from_bin_size(2, (0, 10))),
        (
            {"bin_size": 3, "bin_range": (0, 10), "extend_range": None},
            Bins.from_bin_size(3, (0, 10), remainder="drop"),
        ),
        (
            {"bin_size": 3, "bin_range": (0, 10), "extend_range": False},
            Bins.from_bin_size(3, (0, 10), remainder="keep"),
        ),
        (
            {"bin_size": 3, "bin_range": (0, 10), "extend_range": True},
            Bins.from_bin_size(3, (0, 10), remainder="extend"),
        ),
        (
            {"bins": bh.axis.Regular(5, 0, 10)},
            Bins.from_boost_histogram_axes(bh.axis.Regular(5, 0, 10)),
        ),
        (
            {"bins": Bins.from_n_bins(5, (0, 10)), "labels": ["x"]},
            Bins.from_n_bins(5, (0, 10), labels=["x"]),
        ),
    ],
)
def test_deprecated_keyword_init(kwargs: dict[str, Any], expected: Bins) -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        bins = Bins(**kwargs)
    assert bins == expected


def test_init_with_bin_edges_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bins = Bins(bin_edges=(0, 1, 2), labels=["x"])
        bins_positional = Bins((0, 1, 2), ["x"])
    assert bins == bins_positional == Bins.from_bin_edges((0, 1, 2), labels=["x"])


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({}, ValueError, "`bin_edges` must be given"),
        ({"bin_edges": (0, 1, 2), "bin_range": (0, 2)}, ValueError, "derived from"),
        ({"bins": (0, 1, 2)}, TypeError, "must be an instance of Bins"),
        (
            {"n_bins": 5, "bin_range": (0, 10), "extend_range": 2},
            ValueError,
            "extend_range",
        ),
    ],
)
def test_deprecated_keyword_init_invalid(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with warnings.catch_warnings(), pytest.raises(error, match=match):
        warnings.simplefilter("ignore", DeprecationWarning)
        Bins(**kwargs)


@pytest.mark.parametrize(
    ("bin_edges", "error", "match"),
    [
        ((4, 2, 0), ValueError, "strictly increasing"),
        ((0, 1, 1, 2), ValueError, "strictly increasing"),
        ((5,), ValueError, "At least two"),
        ((0, np.nan, 2), ValueError, "finite"),
        ((0, 1, np.inf), ValueError, "finite"),
        ((), ValueError, "must not be empty"),
        (((),), ValueError, "At least two"),
        (5, TypeError, "sequence"),
        ("abc", TypeError, "sequence"),
        ([(0, 2, 4), 2], TypeError, "one per dimension"),
        ([(0, 2, 4), ((0, 2, 4), (0, 2, 4))], TypeError, "one-dimensional"),
        (((0, 2, 4), (4, 2, 0)), ValueError, "strictly increasing"),
    ],
)
def test_from_bin_edges_invalid(
    bin_edges: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Bins.from_bin_edges(bin_edges)


# --- behaviour -------------------------------------------------------------


@pytest.mark.parametrize("dimension", [4, 5])
def test_more_than_three_dimensions(dimension: int) -> None:
    bin_range = tuple((0, 1) for _ in range(dimension))
    bins = Bins.from_n_bins(2, bin_range)
    assert bins.dimension == dimension
    assert bins.n_bins == (2,) * dimension
    assert bins.bin_size == (0.5,) * dimension
    assert bins.extent == (0.0, 1.0) * dimension
    assert bins == Bins.from_bin_size(0.5, bin_range)
    assert bins == Bins.from_bin_edges([[0, 0.5, 1]] * dimension)
    assert bins == Bins.from_boost_histogram_axes(bins.to_boost_histogram_axes())
    assert bins.with_labels([f"p{i}" for i in range(dimension)]).dimension == dimension
    points = np.array([[0.25] * dimension, [1.5] + [0.0] * (dimension - 1)])
    assert np.array_equal(
        bins.digitize(points), [[1] * dimension, [3] + [1] * (dimension - 1)]
    )


def test_attribute_types_are_uniform() -> None:
    reference = Bins.from_n_bins(5, (0, 10))
    for bins in (
        Bins.from_bin_size(2, (0, 10)),
        Bins.from_bin_edges([0, 2, 4, 6, 8, 10]),
        Bins.from_boost_histogram_axes(bh.axis.Regular(5, 0, 10)),
        Bins.from_any(bin_edges=np.linspace(0, 10, 6)),
    ):
        assert bins == reference
        assert bins.bin_size == (2.0,)
        assert isinstance(bins.bin_size[0], float)
        assert bins.bin_range == ((0.0, 10.0),)
        assert all(isinstance(value, float) for value in bins.bin_range[0])
        assert isinstance(bins.n_bins[0], int)

    bins = Bins.from_n_bins(2, [[0, 10], [0, 5]])
    assert bins.bin_range == ((0.0, 10.0), (0.0, 5.0))
    assert all(isinstance(pair, tuple) for pair in bins.bin_range)


def test_cached_properties_and_immutability() -> None:
    bins = Bins.from_bin_edges(((0, 2, 4), (1, 2, 5)), labels=["x", "y"])
    assert bins.n_bins is bins.n_bins
    assert bins.bin_size is bins.bin_size
    assert bins.bin_centers is bins.bin_centers

    with pytest.raises(AttributeError, match="immutable"):
        bins.n_bins = (3, 3)
    with pytest.raises(AttributeError, match="immutable"):
        # pyrefly: ignore[read-only]  # reason: setting a property is under test
        bins.labels = ["a", "b"]
    with pytest.raises(AttributeError, match="immutable"):
        bins.new_attribute = 1
    with pytest.raises(AttributeError, match="immutable"):
        del bins.n_bins

    with pytest.raises(ValueError, match="read-only"):
        bins.bin_edges[0][0] = 1
    with pytest.raises(ValueError, match="read-only"):
        bins.bin_centers[0][0] = 1
    with pytest.raises(ValueError, match="read-only"):
        # pyrefly: ignore[unsupported-operation]  # reason: array item is under test
        bins.bin_size[1][0] = 1

    labels = bins.labels
    assert labels is not None
    labels.append("z")
    assert bins.labels == ["x", "y"]

    relabelled = bins.with_labels(["a", "b"])
    assert relabelled is not bins
    assert relabelled.labels == ["a", "b"]
    assert bins.labels == ["x", "y"]
    assert np.array_equal(relabelled.bin_edges[0], bins.bin_edges[0])
    assert bins.with_labels(None).labels is None
    with pytest.raises(ValueError, match="one entry per dimension"):
        bins.with_labels(["a"])


def test_repr_and_eq() -> None:
    bins = Bins.from_bin_size((2, (1, 2, 3)), (0, 10), labels=["x", "y"])
    assert repr(bins) == (
        "Bins(dimension=2, n_bins=(5, 3), bin_range=((0.0, 10.0), (0.0, 6.0)), "
        "labels=['x', 'y'])"
    )
    assert bins == Bins.from_bin_edges(
        ((0, 2, 4, 6, 8, 10), (0, 1, 3, 6)), labels=["x", "y"]
    )
    assert bins != bins.with_labels(None)
    assert bins != Bins.from_bin_edges(((0, 2, 4, 6, 8, 10), (0, 1, 3, 7)))
    assert bins != Bins.from_bin_edges((0, 2, 4, 6, 8, 10))
    assert bins != "bins"
    assert (bins == 1) is False
    with pytest.raises(TypeError, match="unhashable"):
        hash(bins)


def test_isclose() -> None:
    bins = Bins.from_n_bins(7, (0, 0.7))
    other = Bins.from_boost_histogram_axes(bh.axis.Regular(7, 0, 0.7))
    assert bins.isclose(other)
    assert not bins.isclose(Bins.from_n_bins(7, (0, 0.8)))
    assert not bins.isclose(Bins.from_n_bins(8, (0, 0.7)))
    assert not bins.isclose(bins.with_labels(["x"]))
    with pytest.raises(TypeError, match="instance of Bins"):
        # pyrefly: ignore[bad-argument-type]  # reason: wrong type is under test
        bins.isclose("bins")


@pytest.mark.parametrize(
    "bins",
    [
        Bins.from_n_bins(5, (0, 10), labels=["x"]),
        Bins.from_bin_edges(((0, 2, 4), (1, 2, 5))),
        Bins.from_boost_histogram_axes(
            bh.axis.AxesTuple((bh.axis.Regular(5, 0, 10), bh.axis.Variable([0, 1, 3])))
        ),
    ],
)
def test_pickle_and_deepcopy(bins: Bins) -> None:
    _ = bins.n_bins  # populate the cache before pickling
    unpickled = pickle.loads(pickle.dumps(bins))  # noqa: S301
    assert "n_bins" not in unpickled.__dict__  # the cache is not pickled
    assert unpickled == bins
    assert unpickled.labels == bins.labels
    assert all(edges.flags.writeable is False for edges in unpickled.bin_edges)
    with pytest.raises(AttributeError, match="immutable"):
        unpickled.n_bins = (1,)

    copied = copy.deepcopy(bins)
    assert copied == bins
    assert copied is not bins


@pytest.mark.parametrize(
    ("bins", "expected"),
    [
        (Bins.from_n_bins(5, (0, 10)), (0.0, 10.0)),
        (Bins.from_n_bins(2, ((0, 10), (1, 5))), (0.0, 10.0, 1.0, 5.0)),
        (
            Bins.from_bin_size(3, ((0, 10), (1, 5), (2, 8))),
            (0.0, 9.0, 1.0, 4.0, 2.0, 8.0),
        ),
    ],
)
def test_extent(bins: Bins, expected: tuple[float, ...]) -> None:
    assert bins.extent == expected
    assert bins.extent == tuple(
        value for edges in bins.bin_edges for value in (edges[0], edges[-1])
    )


def test_digitize_1d() -> None:
    bins = Bins.from_n_bins(5, (0, 10))
    points = [-1, 0, 2.5, 9.99, 10, 11]
    result = bins.digitize(points)
    assert result.dtype == np.intp
    assert np.array_equal(result, [0, 1, 2, 5, 6, 6])
    assert np.array_equal(result, np.digitize(points, bins.bin_edges[0]))
    assert bins.digitize([]).shape == (0,)
    assert np.array_equal(bins.digitize(np.array([[0], [10]])), [[1], [6]])
    with pytest.raises(ValueError, match=r"must have shape \(n_points, 1\)"):
        bins.digitize([[0, 1], [1, 2]])


@pytest.mark.parametrize(
    ("bins", "points", "expected"),
    [
        (
            Bins.from_n_bins(2, ((0, 10), (0, 10))),
            [[0, 0], [5, 10], [-1, 4.9], [10, -1]],
            [[1, 1], [2, 3], [0, 1], [3, 0]],
        ),
        (
            Bins.from_bin_edges(((0, 1, 3, 10), (0, 5, 10), (0, 2, 4))),
            [[0.5, 5, 2], [3, 10, 4], [1, 0, -1]],
            [[1, 2, 2], [3, 3, 3], [2, 1, 0]],
        ),
    ],
)
def test_digitize_nd(bins: Bins, points: Any, expected: Any) -> None:
    result = bins.digitize(points)
    assert result.dtype == np.intp
    assert np.array_equal(result, expected)
    points_ = np.asarray(points, dtype=float)
    for index, edges in enumerate(bins.bin_edges):
        assert np.array_equal(result[:, index], np.digitize(points_[:, index], edges))
    assert bins.digitize(np.empty((0, bins.dimension))).shape == (0, bins.dimension)
    with pytest.raises(ValueError, match="must have shape"):
        bins.digitize([0, 1, 2])
    with pytest.raises(ValueError, match="must have shape"):
        bins.digitize(np.zeros((3, bins.dimension + 1)))


def test_to_boost_histogram_axes() -> None:
    bins = Bins.from_bin_edges((0, 2, 4))
    assert bins.to_boost_histogram_axes() == (bh.axis.Regular(2, 0, 4),)
    assert bins.to_boost_histogram_axes() is not bins.to_boost_histogram_axes()

    bins = Bins.from_bin_edges((0, 1, 2, 4))
    assert bins.to_boost_histogram_axes() == (bh.axis.Variable([0, 1, 2, 4]),)

    bins = Bins.from_bin_size((2, (1, 2, 3)), (0, 10), labels=["x", "y"])
    axes = bins.to_boost_histogram_axes()
    assert isinstance(axes, bh.axis.AxesTuple)
    assert isinstance(axes[0], bh.axis.Regular)
    assert isinstance(axes[1], bh.axis.Variable)
    assert [axis.metadata for axis in axes] == ["x", "y"]
    for axis, edges in zip(axes, bins.bin_edges, strict=True):
        assert np.array_equal(axis.edges, edges)
    assert Bins.from_boost_histogram_axes(axes) == bins

    # regression: axes derived from bins with a non-dividing bin_size
    bins = Bins.from_bin_size(3, (0, 10))
    assert bins.to_boost_histogram_axes() == (bh.axis.Regular(3, 0, 9),)


def test_boost_histogram_axes_property_deprecated() -> None:
    bins = Bins.from_n_bins(5, (0, 10))
    with pytest.warns(DeprecationWarning, match="to_boost_histogram_axes"):
        axes = bins.boost_histogram_axes
    assert axes == bins.to_boost_histogram_axes()


def test_equalize_bin_size() -> None:
    bins = Bins.from_bin_edges((0, 1, 2, 4), labels=["x"])
    assert bins.is_equally_sized == (False,)
    equalized = bins.equalize_bin_size()
    assert_edges_equal(equalized, [[0, 1, 2, 3, 4]])
    assert equalized.bin_size == (1.0,)
    assert equalized.is_equally_sized == (True,)
    assert equalized.labels == ["x"]  # regression: labels were dropped

    equalized = Bins.from_bin_edges(
        ((-1.0, 1.0, 2.0, 4.0), (-1.0, 1.0, 2.0, 6.0))
    ).equalize_bin_size()
    assert_edges_equal(equalized, [[-1, 1, 3], [-1, 1, 3, 5]])
    assert equalized.bin_size == (2.0, 2.0)

    equalized = Bins.from_bin_size(2.0, (-1, 10)).equalize_bin_size()
    assert_edges_equal(equalized, [[-1, 1, 3, 5, 7, 9]])

    bins = Bins.from_n_bins(5, (0, 10))
    assert bins.equalize_bin_size() == bins


def test_indices_to_bin_centers() -> None:
    bins = Bins.from_bin_edges([0, 1, 2, 4, 8, 9])
    assert _indices_to_bin_centers(bins, 2) == 3
    assert np.array_equal(_indices_to_bin_centers(bins, [0, 2, 1]), [0.5, 3, 1.5])
    assert np.array_equal(
        _indices_to_bin_centers(bins, [[0, 1], [2, 2], [4, 3]]),
        [[0.5, 1.5], [3, 3], [8.5, 6]],
    )

    bins = Bins.from_bin_edges([[0, 1, 2, 4, 8, 9], [1, 2, 4, 8, 9]])
    assert np.array_equal(
        _indices_to_bin_centers(bins, [[0, 1], [2, 2], [4, 3]]),
        [[0.5, 3], [3, 6], [8.5, 8.5]],
    )
    assert np.array_equal(_indices_to_bin_centers(bins, 1), [1.5, 3])
    with pytest.raises(ValueError, match="must have shape"):
        _indices_to_bin_centers(bins, [[0, 1, 2]])
