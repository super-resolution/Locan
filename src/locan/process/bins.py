"""

Bins for histograms and images.

This module provides :class:`Bins` to describe bin edges in one or more
dimensions. `Bins` is used by :func:`locan.process.aggregate.histogram`,
by the render functions and by :class:`locan.data.images.Image`. These
consumers work with one to three dimensions; `Bins` itself supports any
number of dimensions.

Bins can be constructed from bin edges, from a number of bins and a range,
from a bin size and a range, or from boost-histogram axes:

>>> bins = Bins.from_n_bins(n_bins=5, bin_range=(0, 10))
>>> bins = Bins.from_bin_size(bin_size=2, bin_range=((0, 10), (0, 5)))
>>> bins = Bins.from_bin_edges(bin_edges=((0, 1, 3, 10), (0, 5, 10)))

All bins are half-open intervals ``[edge_i, edge_i+1)``. A value equal to the
last edge is not contained in the last bin.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from functools import cached_property
from math import isclose
from typing import Any, Literal

import boost_histogram as bh
import numpy as np
import numpy.typing as npt

__all__: list[str] = ["Bins"]

Remainder = Literal["drop", "keep", "extend"]
_REMAINDER_OPTIONS: tuple[str, ...] = ("drop", "keep", "extend")

# Relative tolerance below which bin sizes within one dimension are considered
# equal (floating-point noise from computing edges).
_EQUAL_SIZE_RTOL: float = 1e-5
# Relative tolerance for deciding whether `bin_size` divides `bin_range`.
_FIT_RTOL: float = 1e-8

NumericArray = npt.NDArray[np.floating] | npt.NDArray[np.integer]
BinRangeInput = (
    tuple[float, float] | Sequence[float] | Sequence[Sequence[float]] | NumericArray
)
BinEdgesInput = (
    Sequence[float] | Sequence[Sequence[float]] | Sequence[NumericArray] | NumericArray
)
BinSizeInput = (
    float
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Sequence[float | Sequence[float]]
    | NumericArray
)
NBinsInput = int | Sequence[int] | NumericArray
LabelsInput = str | Sequence[str] | None


def _is_scalar(element: Any) -> bool:
    """Return True for a python or numpy number (including 0-d arrays)."""
    if isinstance(element, bool | np.bool_):
        return False
    if isinstance(element, int | float | np.integer | np.floating):
        return True
    return isinstance(element, np.ndarray) and element.ndim == 0


def _as_edges(edges: object) -> npt.NDArray[np.float64]:
    """
    Validate bin edges for one dimension and return them as read-only array.

    Raises
    ------
    ValueError
        If fewer than two edges are given, edges are not finite or not
        strictly increasing.
    """
    try:
        array = np.asarray(edges, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("Bin edges must be a sequence of numbers.") from exc
    if array.ndim != 1:
        raise TypeError("Bin edges for one dimension must be one-dimensional.")
    if array.size < 2:
        raise ValueError("At least two bin edges are required in each dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Bin edges must be finite.")
    if not np.all(np.diff(array) > 0):
        raise ValueError("Bin edges must be strictly increasing.")
    if array.flags.writeable:
        array = array.copy()
        array.flags.writeable = False
    return array


def _normalize_bin_edges(
    bin_edges: BinEdgesInput,
) -> tuple[npt.NDArray[np.float64], ...]:
    """
    Turn a bin edge specification into one read-only array per dimension.

    Accepts a sequence of numbers (one dimension) or a sequence of such
    sequences (one per dimension).
    """
    if _is_scalar(bin_edges) or isinstance(bin_edges, str):
        raise TypeError("`bin_edges` must be a sequence of numbers or of sequences.")
    try:
        items = list(bin_edges)
    except TypeError as exc:
        raise TypeError(
            "`bin_edges` must be a sequence of numbers or of sequences."
        ) from exc
    if not items:
        raise ValueError("`bin_edges` must not be empty.")
    if all(_is_scalar(item) for item in items):
        return (_as_edges(items),)
    if all(not _is_scalar(item) for item in items):
        return tuple(_as_edges(item) for item in items)
    raise TypeError(
        "`bin_edges` must be a sequence of numbers or a sequence of such "
        "sequences, one per dimension."
    )


def _normalize_labels(labels: LabelsInput, dimension: int) -> tuple[str, ...] | None:
    """Validate labels and return them as tuple with one entry per dimension."""
    if labels is None:
        return None
    if isinstance(labels, str):
        labels_ = (labels,)
    elif isinstance(labels, Iterable):
        labels_ = tuple(labels)
    else:
        raise TypeError("`labels` must be str, a sequence of str or None.")
    if not all(isinstance(label, str) for label in labels_):
        raise TypeError("`labels` must be str, a sequence of str or None.")
    if len(labels_) != dimension:
        raise ValueError(
            f"`labels` must have one entry per dimension ({dimension}), "
            f"got {len(labels_)}."
        )
    return labels_


def _parse_bin_range(
    bin_range: BinRangeInput,
) -> tuple[list[tuple[float, float]], bool]:
    """
    Parse `bin_range` into a list of (min, max) pairs.

    Returns
    -------
    tuple[list[tuple[float, float]], bool]
        The ranges and a flag that is True if a single (min, max) pair was
        given that applies to all dimensions.
    """
    try:
        array = np.asarray(bin_range, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "`bin_range` must be (min, max) or a sequence of (min, max) pairs."
        ) from exc
    if array.ndim == 1 and array.shape == (2,):
        pairs = [(float(array[0]), float(array[1]))]
        broadcast = True
    elif array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] == 2:
        pairs = [(float(low), float(high)) for low, high in array]
        broadcast = False
    else:
        raise TypeError(
            "`bin_range` must be (min, max) or a sequence of (min, max) pairs."
        )
    for low, high in pairs:
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError("`bin_range` must be finite.")
        if not low < high:
            raise ValueError("`bin_range` must satisfy min < max.")
    return pairs, broadcast


def _as_items(specification: object, name: str) -> list[object]:
    """
    Return the elements of a per-dimension specification.

    The specification is converted to an object array so that ragged input
    like ``(2, (1, 2, 3))`` yields one element per dimension.
    """
    try:
        array = np.asarray(specification, dtype=object)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"`{name}` has an unsupported structure.") from exc
    if array.ndim == 0:
        raise TypeError(f"`{name}` has an unsupported structure.")
    if array.shape[0] == 0:
        raise ValueError(f"`{name}` must not be empty.")
    return list(array)


def _parse_n_bins(n_bins: NBinsInput) -> tuple[list[int], bool]:
    """Parse `n_bins` into a list of positive integers (one per dimension)."""

    def _one(value: object) -> int:
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        if isinstance(value, bool | np.bool_) or not isinstance(
            value, int | np.integer
        ):
            raise TypeError("`n_bins` must be an integer or a sequence of integers.")
        if value < 1:
            raise ValueError("`n_bins` must be at least 1.")
        return int(value)

    if _is_scalar(n_bins) or isinstance(n_bins, bool | np.bool_):
        return [_one(n_bins)], True
    return [_one(item) for item in _as_items(n_bins, "n_bins")], False


def _parse_bin_size(
    bin_size: BinSizeInput,
) -> tuple[list[float | npt.NDArray[np.float64]], bool]:
    """
    Parse `bin_size` into one entry per dimension.

    Each entry is a float (equally sized bins) or a 1d array (variable sizes).
    """

    def _check_positive(array: npt.NDArray[np.float64]) -> None:
        if not np.all(np.isfinite(array)) or not np.all(array > 0):
            raise ValueError("`bin_size` must be positive and finite.")

    def _as_float(value: object) -> float:
        array = np.asarray(value, dtype=np.float64)
        _check_positive(array)
        return float(array.item())

    if isinstance(bin_size, bool | np.bool_):
        raise TypeError("`bin_size` must be a number, not a bool.")
    if _is_scalar(bin_size):
        return [_as_float(bin_size)], True
    sizes: list[float | npt.NDArray[np.float64]] = []
    for item in _as_items(bin_size, "bin_size"):
        if _is_scalar(item):
            sizes.append(_as_float(item))
        else:
            try:
                array = np.asarray(item, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "`bin_size` must be a number, a sequence of numbers or a "
                    "sequence of such sequences."
                ) from exc
            if array.ndim != 1 or array.size == 0:
                raise TypeError(
                    "`bin_size` must be 0-, 1- or 2-dimensional. Use "
                    "`bin_edges` for arbitrary edges."
                )
            _check_positive(array)
            sizes.append(array)
    return sizes, False


def _broadcast_specification(
    spec_items: list[Any],
    spec_broadcast: bool,
    range_items: list[tuple[float, float]],
    range_broadcast: bool,
    spec_name: str,
) -> tuple[list[Any], list[tuple[float, float]]]:
    """Align per-dimension specifications and ranges, broadcasting scalars."""
    if spec_broadcast and range_broadcast:
        dimension = 1
    elif spec_broadcast:
        dimension = len(range_items)
    elif range_broadcast:
        dimension = len(spec_items)
    else:
        if len(spec_items) != len(range_items):
            raise ValueError(
                f"`{spec_name}` and `bin_range` must have the same number of "
                f"dimensions, got {len(spec_items)} and {len(range_items)}."
            )
        dimension = len(spec_items)
    if spec_broadcast:
        spec_items = spec_items * dimension
    if range_broadcast:
        range_items = range_items * dimension
    return spec_items, range_items


def _edges_from_n_bins(
    n_bins: NBinsInput, bin_range: BinRangeInput
) -> tuple[npt.NDArray[np.float64], ...]:
    """Compute bin edges for each dimension from `n_bins` and `bin_range`."""
    n_items, n_broadcast = _parse_n_bins(n_bins)
    range_items, range_broadcast = _parse_bin_range(bin_range)
    n_items, range_items = _broadcast_specification(
        n_items, n_broadcast, range_items, range_broadcast, "n_bins"
    )
    return tuple(
        _as_edges(np.linspace(low, high, n + 1))
        for n, (low, high) in zip(n_items, range_items, strict=True)
    )


def _edges_from_equal_size(
    size: float, low: float, high: float, remainder: Remainder
) -> npt.NDArray[np.float64]:
    """Compute edges for equally sized bins within (low, high)."""
    n_exact = (high - low) / size
    n_rounded = round(n_exact)
    if n_rounded >= 1 and isclose(n_exact, n_rounded, rel_tol=_FIT_RTOL):
        return np.linspace(low, high, n_rounded + 1)
    n_full = int(np.floor(n_exact))
    if n_full == 0:
        # bin_size exceeds the range: a single bin
        if remainder == "extend":
            return np.array([low, low + size])
        return np.array([low, high])
    end = low + n_full * size
    edges = np.linspace(low, end, n_full + 1)
    if remainder == "drop":
        return edges
    if remainder == "keep":
        return np.append(edges, high)
    return np.append(edges, end + size)


def _edges_from_variable_sizes(
    sizes: npt.NDArray[np.float64], low: float, high: float, remainder: Remainder
) -> npt.NDArray[np.float64]:
    """Compute edges for a sequence of bin sizes starting at `low`."""
    all_edges = low + np.concatenate(([0.0], np.cumsum(sizes)))
    is_inside = (all_edges < high) | np.isclose(all_edges, high, rtol=_FIT_RTOL, atol=0)
    inside = all_edges[is_inside]
    if inside.size and np.isclose(inside[-1], high, rtol=_FIT_RTOL, atol=0):
        inside[-1] = high
        return inside
    if inside.size < 2:
        # first bin_size exceeds the range: a single bin
        if remainder == "extend":
            return np.array([low, low + sizes[0]])
        return np.array([low, high])
    if remainder == "drop":
        return inside
    if remainder == "keep":
        return np.append(inside, high)
    if all_edges.size > inside.size:
        return np.append(inside, all_edges[inside.size])
    return inside


def _edges_from_bin_size(
    bin_size: BinSizeInput, bin_range: BinRangeInput, remainder: Remainder
) -> tuple[npt.NDArray[np.float64], ...]:
    """Compute bin edges for each dimension from `bin_size` and `bin_range`."""
    if remainder not in _REMAINDER_OPTIONS:
        raise ValueError(f"`remainder` must be one of {_REMAINDER_OPTIONS}.")
    size_items, size_broadcast = _parse_bin_size(bin_size)
    range_items, range_broadcast = _parse_bin_range(bin_range)
    size_items, range_items = _broadcast_specification(
        size_items, size_broadcast, range_items, range_broadcast, "bin_size"
    )
    edges = []
    for size, (low, high) in zip(size_items, range_items, strict=True):
        if isinstance(size, np.ndarray):
            edges.append(_edges_from_variable_sizes(size, low, high, remainder))
        else:
            edges.append(_edges_from_equal_size(size, low, high, remainder))
    return tuple(_as_edges(edges_) for edges_ in edges)


def _edges_from_axes(
    axes: bh.axis.Axis | bh.axis.AxesTuple | Sequence[bh.axis.Axis],
) -> tuple[tuple[npt.NDArray[np.float64], ...], tuple[str, ...] | None]:
    """
    Extract bin edges and labels from boost-histogram axes.

    Returns
    -------
    tuple[tuple[npt.NDArray[np.float64], ...], tuple[str, ...] | None]
        Edges per dimension and labels taken from axis metadata if all axes
        carry a string as metadata.
    """
    if isinstance(axes, bh.axis.Axis):
        axes_ = [axes]
    else:
        try:
            axes_ = list(axes)
        except TypeError as exc:
            raise TypeError(
                "`axes` must be a boost_histogram axis or a sequence of axes."
            ) from exc
    if not axes_ or not all(isinstance(axis, bh.axis.Axis) for axis in axes_):
        raise TypeError("`axes` must be a boost_histogram axis or a sequence of axes.")
    for axis in axes_:
        traits = axis.traits
        if traits.growth or traits.circular or not traits.ordered:
            raise ValueError(
                "Only boost_histogram axes with fixed, ordered edges are supported "
                "(Regular, Variable, Integer without growth or circular traits)."
            )
    edges = tuple(_as_edges(axis.edges) for axis in axes_)
    metadata = [axis.metadata for axis in axes_]
    labels = (
        tuple(metadata)
        if all(isinstance(item, str) and item for item in metadata)
        else None
    )
    return edges, labels


def _bin_size_one_dimension(
    edges: npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Return a single size for equally sized bins, else the size of each bin."""
    differences = np.diff(edges)
    if np.all(np.isclose(differences, differences[0], rtol=_EQUAL_SIZE_RTOL, atol=0)):
        return float((edges[-1] - edges[0]) / differences.size)
    differences.flags.writeable = False
    return differences


class Bins:
    """
    Bin edges for one or more dimensions to be used in histogram and
    render functions.

    `Bins` supports any number of dimensions. The consumers
    :func:`locan.process.aggregate.histogram`, the render functions and
    :class:`locan.data.images.Image` are limited to one to three dimensions.

    `Bins` is immutable. It stores the bin edges for each dimension and
    derives all other attributes from them on first access.
    Bins are half-open intervals ``[edge_i, edge_i+1)``.

    Use one of the constructor methods :meth:`from_bin_edges`,
    :meth:`from_n_bins`, :meth:`from_bin_size`,
    :meth:`from_boost_histogram_axes` or :meth:`from_any`.

    Parameters
    ----------
    bin_edges
        Bin edges for one dimension as sequence of numbers or for each
        dimension as sequence of such sequences.
    labels
        Names for each bin axis with shape (dimension,).

    Attributes
    ----------
    dimension : int
        The number of dimensions for which bins are provided.
    bin_edges : tuple[npt.NDArray[np.float64], ...]
        Read-only array with bin edges for each dimension.
    bin_range : tuple[tuple[float, float], ...]
        First and last edge for each dimension.
    n_bins : tuple[int, ...]
        Number of bins for each dimension.
    bin_size : tuple[float | npt.NDArray[np.float64], ...]
        For each dimension a single size if all bins are equally sized
        or an array with the size of each bin.
    bin_centers : tuple[npt.NDArray[np.float64], ...]
        Array with bin centers for each dimension.
    is_equally_sized : tuple[bool, ...]
        True for each dimension if all bins are of the same size.
    extent : tuple[float, ...]
        First and last edge of each dimension flattened to
        (x_min, x_max, y_min, y_max, ...) as used by
        :func:`matplotlib.pyplot.imshow`.
    labels : list[str] | None
        Names for each bin axis.

    Examples
    --------
    >>> bins = Bins.from_n_bins(n_bins=(2, 5), bin_range=((0, 10), (0, 5)))
    >>> bins.n_bins
    (2, 5)
    >>> bins.bin_size
    (5.0, 1.0)
    """

    _bin_edges: tuple[npt.NDArray[np.float64], ...]
    _labels: tuple[str, ...] | None
    _initialized: bool

    def __init__(
        self,
        bin_edges: BinEdgesInput | None = None,
        labels: LabelsInput = None,
        *,
        bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
        n_bins: NBinsInput | None = None,
        bin_size: BinSizeInput | None = None,
        bin_range: BinRangeInput | None = None,
        extend_range: bool | None = None,
    ) -> None:
        if any(item is not None for item in (bins, n_bins, bin_size, bin_range)) or (
            extend_range is not None
        ):
            # Deprecated keyword construction: delegate to from_any.
            warnings.warn(
                "Constructing Bins with keyword specifications is deprecated. "
                "Use Bins.from_bin_edges, Bins.from_n_bins, Bins.from_bin_size, "
                "Bins.from_boost_histogram_axes or Bins.from_any instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            remainder_map: dict[bool | None, Remainder] = {
                None: "drop",
                False: "keep",
                True: "extend",
            }
            if extend_range not in remainder_map:
                raise ValueError("`extend_range` must be None, True or False.")
            other = type(self).from_any(
                bins=bins,
                n_bins=n_bins,
                bin_size=bin_size,
                bin_edges=bin_edges,
                bin_range=bin_range,
                labels=labels,
                remainder=remainder_map[extend_range],
            )
            edges_ = other._bin_edges
            labels_ = other._labels
        else:
            if bin_edges is None:
                raise ValueError("`bin_edges` must be given.")
            edges_ = _normalize_bin_edges(bin_edges)
            labels_ = _normalize_labels(labels, len(edges_))
        object.__setattr__(self, "_bin_edges", edges_)
        object.__setattr__(self, "_labels", labels_)
        object.__setattr__(self, "_initialized", True)

    # --- constructors -----------------------------------------------------

    @classmethod
    def from_bin_edges(
        cls, bin_edges: BinEdgesInput, *, labels: LabelsInput = None
    ) -> Bins:
        """
        Construct bins from bin edges.

        Parameters
        ----------
        bin_edges
            Bin edges for one dimension as sequence of numbers or for each
            dimension as sequence of such sequences.
        labels
            Names for each bin axis with shape (dimension,).

        Returns
        -------
        Bins

        Raises
        ------
        ValueError
            If fewer than two edges are given in a dimension, edges are not
            strictly increasing or `labels` does not match the dimension.
        """
        return cls(bin_edges, labels)

    @classmethod
    def from_n_bins(
        cls, n_bins: NBinsInput, bin_range: BinRangeInput, *, labels: LabelsInput = None
    ) -> Bins:
        """
        Construct equally sized bins from the number of bins and a range.

        Parameters
        ----------
        n_bins
            The number of bins for all or each dimension.
            5 yields 5 bins in all dimensions.
            (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
        bin_range
            Minimum and maximum edge for all or each dimension
            with shape (2,) or (dimension, 2).
        labels
            Names for each bin axis with shape (dimension,).

        Returns
        -------
        Bins

        Raises
        ------
        ValueError
            If `n_bins` is smaller than 1, `bin_range` does not satisfy
            min < max, or the dimensions of `n_bins` and `bin_range` differ.
        """
        return cls(_edges_from_n_bins(n_bins, bin_range), labels)

    @classmethod
    def from_bin_size(
        cls,
        bin_size: BinSizeInput,
        bin_range: BinRangeInput,
        *,
        remainder: Remainder = "drop",
        labels: LabelsInput = None,
    ) -> Bins:
        """
        Construct bins from the bin size and a range.

        Parameters
        ----------
        bin_size
            The size of bins for all or each bin and for all or each dimension
            with shape (dimension,) or (dimension, n_bins).
            5 describes a bin size of 5 for all bins in all dimensions.
            ((2, 5),) yields bins of size 2 and 5 in one dimension.
            (2, 5) yields bins of size 2 for one dimension and 5 for the other
            dimension.
            ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
            (1, 3) for the other dimension.
        bin_range
            Minimum and maximum edge for all or each dimension
            with shape (2,) or (dimension, 2).
        remainder
            How to treat the part of `bin_range` that is not covered by full
            bins if `bin_size` does not divide the range:
            "drop" ends the bins at the last full bin (the maximum is reduced);
            "keep" adds a smaller last bin ending at the maximum;
            "extend" adds a full bin reaching beyond the maximum.
            If `bin_size` exceeds the range, "drop" and "keep" yield a single
            bin covering the range.
        labels
            Names for each bin axis with shape (dimension,).

        Returns
        -------
        Bins

        Raises
        ------
        ValueError
            If `bin_size` is not positive, `bin_range` does not satisfy
            min < max, `remainder` is not a valid option, or the dimensions of
            `bin_size` and `bin_range` differ.
        """
        return cls(_edges_from_bin_size(bin_size, bin_range, remainder), labels)

    @classmethod
    def from_boost_histogram_axes(
        cls,
        axes: bh.axis.Axis | bh.axis.AxesTuple | Sequence[bh.axis.Axis],
        *,
        labels: LabelsInput = None,
    ) -> Bins:
        """
        Construct bins from boost-histogram axes.

        Parameters
        ----------
        axes
            A single axis or a sequence of axes with fixed, ordered edges
            (e.g. Regular, Variable, Integer). Transformed Regular axes are
            represented by their edges.
        labels
            Names for each bin axis with shape (dimension,).
            If None, string metadata of the axes is used if available.

        Returns
        -------
        Bins

        Raises
        ------
        ValueError
            If an axis has growth or circular traits or is not ordered.
        """
        edges, metadata_labels = _edges_from_axes(axes)
        return cls(edges, labels if labels is not None else metadata_labels)

    @classmethod
    def from_any(
        cls,
        bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
        n_bins: NBinsInput | None = None,
        bin_size: BinSizeInput | None = None,
        bin_edges: BinEdgesInput | None = None,
        bin_range: BinRangeInput | None = None,
        *,
        labels: LabelsInput = None,
        remainder: Remainder = "drop",
    ) -> Bins:
        """
        Construct bins from exactly one of several specifications.

        The signature mirrors the bin parameters of user-facing functions
        like :func:`histogram`.

        Parameters
        ----------
        bins
            An existing `Bins` instance or boost-histogram axes.
        n_bins
            The number of bins for all or each dimension, see :meth:`from_n_bins`.
        bin_size
            The size of bins, see :meth:`from_bin_size`.
        bin_edges
            Bin edges for all or each dimension, see :meth:`from_bin_edges`.
        bin_range
            Minimum and maximum edge for all or each dimension.
            Required with `n_bins` or `bin_size`; must be None otherwise.
        labels
            Names for each bin axis with shape (dimension,).
            If given together with `bins`, the labels are replaced.
        remainder
            Treatment of a range not divisible by `bin_size`,
            see :meth:`from_bin_size`.

        Returns
        -------
        Bins

        Raises
        ------
        ValueError
            If not exactly one of `bins`, `n_bins`, `bin_size`, `bin_edges`
            is given or `bin_range` is given or missing inappropriately.
        TypeError
            If `bins` is neither `Bins` nor boost-histogram axes.
        """
        given = [
            name
            for name, value in (
                ("bins", bins),
                ("n_bins", n_bins),
                ("bin_size", bin_size),
                ("bin_edges", bin_edges),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "Exactly one of `bins`, `n_bins`, `bin_size`, `bin_edges` must be "
                f"given, got {given if given else 'none'}."
            )
        if bins is not None or bin_edges is not None:
            if bin_range is not None:
                raise ValueError(
                    f"`bin_range` is derived from `{given[0]}` and must be None."
                )
            if bins is not None:
                if isinstance(bins, Bins):
                    return bins if labels is None else bins.with_labels(labels)
                if isinstance(bins, bh.axis.Axis | bh.axis.AxesTuple):
                    return cls.from_boost_histogram_axes(bins, labels=labels)
                raise TypeError(
                    "`bins` must be an instance of Bins, boost_histogram.axis.Axis "
                    "or boost_histogram.axis.AxesTuple."
                )
            if bin_edges is not None:
                return cls.from_bin_edges(bin_edges, labels=labels)
        if bin_range is None:
            raise ValueError(f"`bin_range` must be given together with `{given[0]}`.")
        if n_bins is not None:
            return cls.from_n_bins(n_bins, bin_range, labels=labels)
        if bin_size is not None:
            return cls.from_bin_size(
                bin_size, bin_range, remainder=remainder, labels=labels
            )
        raise AssertionError("unreachable")  # pragma: no cover

    # --- immutability -----------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError(
                f"Bins is immutable; cannot set attribute {name!r}. "
                "Use with_labels() or a constructor method to create new bins."
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Bins is immutable; cannot delete attribute {name!r}.")

    def __getstate__(self) -> dict[str, Any]:
        return {"_bin_edges": self._bin_edges, "_labels": self._labels}

    def __setstate__(self, state: dict[str, Any]) -> None:
        edges = tuple(_as_edges(edges_) for edges_ in state["_bin_edges"])
        object.__setattr__(self, "_bin_edges", edges)
        object.__setattr__(self, "_labels", state["_labels"])
        object.__setattr__(self, "_initialized", True)

    # --- stored attributes ------------------------------------------------

    @property
    def bin_edges(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Read-only arrays with bin edges for each dimension."""
        return self._bin_edges

    @property
    def labels(self) -> list[str] | None:
        """Names for each bin axis."""
        return None if self._labels is None else list(self._labels)

    # --- derived attributes -----------------------------------------------

    @cached_property
    def dimension(self) -> int:
        """The number of dimensions for which bins are provided."""
        return len(self._bin_edges)

    @cached_property
    def n_bins(self) -> tuple[int, ...]:
        """Number of bins for each dimension."""
        return tuple(edges.size - 1 for edges in self._bin_edges)

    @cached_property
    def bin_range(self) -> tuple[tuple[float, float], ...]:
        """First and last edge for each dimension."""
        return tuple((float(edges[0]), float(edges[-1])) for edges in self._bin_edges)

    @cached_property
    def bin_size(self) -> tuple[float | npt.NDArray[np.float64], ...]:
        """Size of bins for each dimension: a float for equally sized bins,
        else an array with the size of each bin."""
        return tuple(_bin_size_one_dimension(edges) for edges in self._bin_edges)

    @cached_property
    def bin_centers(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Bin centers for each dimension."""
        centers = []
        for edges in self._bin_edges:
            center = (edges[:-1] + edges[1:]) / 2
            center.flags.writeable = False
            centers.append(center)
        return tuple(centers)

    @cached_property
    def is_equally_sized(self) -> tuple[bool, ...]:
        """True for each dimension if all bins are of the same size."""
        return tuple(not isinstance(size, np.ndarray) for size in self.bin_size)

    @cached_property
    def extent(self) -> tuple[float, ...]:
        """First and last edge of each dimension flattened to
        (x_min, x_max, y_min, y_max, ...)."""
        return tuple(value for bin_range in self.bin_range for value in bin_range)

    @property
    def boost_histogram_axes(self) -> bh.axis.AxesTuple:
        """
        Axis definitions for boost-histogram.

        .. deprecated::
            Use :meth:`to_boost_histogram_axes` instead.
        """
        warnings.warn(
            "`Bins.boost_histogram_axes` is deprecated, "
            "use `Bins.to_boost_histogram_axes()`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_boost_histogram_axes()

    # --- methods ----------------------------------------------------------

    def to_boost_histogram_axes(self) -> bh.axis.AxesTuple:
        """
        Create boost-histogram axes for these bins.

        Equally sized dimensions yield :class:`boost_histogram.axis.Regular`,
        others :class:`boost_histogram.axis.Variable`. Labels are stored as
        axis metadata. A new object is created on every call.

        Returns
        -------
        boost_histogram.axis.AxesTuple
        """
        axes = []
        for index in range(self.dimension):
            metadata = None if self._labels is None else self._labels[index]
            if self.is_equally_sized[index]:
                axis: bh.axis.Axis = bh.axis.Regular(
                    self.n_bins[index], *self.bin_range[index], metadata=metadata
                )
            else:
                axis = bh.axis.Variable(self._bin_edges[index], metadata=metadata)
            axes.append(axis)
        return bh.axis.AxesTuple(axes)

    def with_labels(self, labels: LabelsInput) -> Bins:
        """
        Return new bins with the same edges and the given labels.

        Parameters
        ----------
        labels
            Names for each bin axis with shape (dimension,), or None.

        Returns
        -------
        Bins
        """
        return type(self)(self._bin_edges, labels)

    def equalize_bin_size(self) -> Bins:
        """
        Return new bins with equally sized bins in each dimension.

        The bin size of each dimension is set to its first bin size;
        the range is covered with ``remainder="drop"``. Labels are kept.

        Returns
        -------
        Bins
        """
        sizes = [
            size if isinstance(size, float) else float(size[0])
            for size in self.bin_size
        ]
        return type(self).from_bin_size(
            sizes, self.bin_range, remainder="drop", labels=self._labels
        )

    def digitize(self, points: npt.ArrayLike) -> npt.NDArray[np.intp]:
        """
        Return the bin index for each point in each dimension.

        Indices follow :func:`numpy.digitize`: 0 marks values below the first
        edge (underflow), 1 to n_bins the bins ``[edge_i, edge_i+1)``, and
        n_bins + 1 values equal to or above the last edge (overflow).

        Parameters
        ----------
        points
            Coordinates with shape (n_points,) for one-dimensional bins or
            (n_points, dimension).

        Returns
        -------
        npt.NDArray[np.intp]
            Bin indices with shape (n_points,) for one-dimensional input,
            else (n_points, dimension).

        Raises
        ------
        ValueError
            If the dimension of `points` does not match the bins.
        """
        points_ = np.asarray(points, dtype=np.float64)
        if points_.ndim == 1:
            if self.dimension != 1:
                raise ValueError(
                    f"`points` must have shape (n_points, {self.dimension})."
                )
            return np.digitize(points_, self._bin_edges[0]).astype(np.intp)
        if points_.ndim != 2 or points_.shape[1] != self.dimension:
            raise ValueError(f"`points` must have shape (n_points, {self.dimension}).")
        if points_.shape[0] == 0:
            return np.empty((0, self.dimension), dtype=np.intp)
        return np.stack(
            [
                np.digitize(points_[:, index], edges)
                for index, edges in enumerate(self._bin_edges)
            ],
            axis=1,
        ).astype(np.intp)

    def isclose(self, other: Bins, rtol: float = 1e-12, atol: float = 0.0) -> bool:
        """
        Return True if `other` has the same labels and bin edges within
        floating-point tolerance.

        Parameters
        ----------
        other
            Bins to compare with.
        rtol
            Relative tolerance passed to :func:`numpy.allclose`.
        atol
            Absolute tolerance passed to :func:`numpy.allclose`.

        Returns
        -------
        bool
        """
        if not isinstance(other, Bins):
            raise TypeError("`other` must be an instance of Bins.")
        return (
            self.n_bins == other.n_bins
            and self._labels == other._labels
            and all(
                np.allclose(edges, other_edges, rtol=rtol, atol=atol)
                for edges, other_edges in zip(
                    self._bin_edges, other._bin_edges, strict=True
                )
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bins):
            return NotImplemented
        return (
            self.n_bins == other.n_bins
            and self._labels == other._labels
            and all(
                np.array_equal(edges, other_edges)
                for edges, other_edges in zip(
                    self._bin_edges, other._bin_edges, strict=True
                )
            )
        )

    __hash__ = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dimension={self.dimension}, n_bins={self.n_bins}, "
            f"bin_range={self.bin_range}, labels={self.labels})"
        )


def _indices_to_bin_centers(
    bins: Bins, indices: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Compute bin centers for given zero-based bin indices.

    Parameters
    ----------
    bins
        The bin specification.
    indices
        Zero-based bin indices. For one-dimensional bins any shape is allowed.
        For multi-dimensional bins either a scalar or 1d array that is applied
        to every dimension, or an array with shape (n_indices, dimension).

    Returns
    -------
    npt.NDArray[np.float64]
        Selected bin centers with shape (n_indices, dimension) or the shape
        of `indices` for one-dimensional bins.
    """
    indices_ = np.asarray(indices)
    centers = bins.bin_centers
    if bins.dimension == 1:
        return np.asarray(centers[0][indices_])
    if indices_.ndim < 2:
        return np.array([center[indices_] for center in centers])
    if indices_.ndim != 2 or indices_.shape[1] != bins.dimension:
        raise ValueError(f"`indices` must have shape (n_indices, {bins.dimension}).")
    return np.stack(
        [center[indices_[:, index]] for index, center in enumerate(centers)], axis=1
    )
