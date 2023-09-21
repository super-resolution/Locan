"""

Register localization data.

This module registers localization data and provides transformation parameters
to put other localization data in registry.

Parts of this code is adapted from https://github.com/jungmannlab/picasso.
(MIT license, Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry)
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from lmfit import Model, Parameters

from locan.data.aggregate import Bins, histogram
from locan.data.locdata import LocData
from locan.data.properties.locdata_statistics import range_from_collection
from locan.data.transform.spatial_transformation import _homogeneous_matrix
from locan.dependencies import HAS_DEPENDENCY, needs_package

if HAS_DEPENDENCY["open3d"]:
    import open3d as o3d

if TYPE_CHECKING:
    import boost_histogram as bh


__all__: list[str] = ["register_icp", "register_cc"]


class Transformation(NamedTuple):
    matrix: npt.NDArray[np.float_]
    offset: npt.NDArray[np.float_]


@needs_package("open3d")
def _register_icp_open3d(
    points: npt.ArrayLike,
    other_points: npt.ArrayLike,
    matrix: npt.ArrayLike | None = None,
    offset: npt.ArrayLike | None = None,
    pre_translation: npt.ArrayLike | None = None,
    max_correspondence_distance: float = 1_000,
    max_iteration: int = 10_000,
    with_scaling: bool = True,
    verbose: bool = True,
) -> Transformation:
    """
    Register `points` by an "Iterative Closest Point" algorithm using open3d.

    Parameters
    ----------
    points
        Points representing the source on which to perform the manipulation.
    other_points
        Points representing the target.
    matrix
        Transformation matrix with shape (d, d)used as initial value.
        If None the unit matrix is used.
    offset
        Translation vector with shape (d,) used as initial value.
        If None a vector of zeros is used.
    pre_translation
        Values for translation of coordinates before registration.
    max_correspondence_distance
        Threshold distance for the icp algorithm. Parameter is passed to
        open3d algorithm.
    max_iteration
        Maximum number of iterations. Parameter is passed to open3d algorithm.
    with_scaling
        Allow scaling transformation. Parameter is passed to open3d algorithm.
    verbose
        Flag indicating if transformation results are printed out.

    Returns
    -------
    Transformation
        Matrix and offset representing the optimized transformation.
    """
    points_ = np.asarray(points)
    other_points_ = np.asarray(other_points)

    # prepare 3d
    if np.shape(points)[1] == np.shape(other_points)[1]:
        dimension = np.shape(points)[1]
    else:
        raise ValueError("Dimensions for locdata and other_locdata are incompatible.")

    if dimension == 2:
        points_3d = np.concatenate([points_, np.zeros((len(points_), 1))], axis=1)
        other_points_3d = np.concatenate(
            [other_points_, np.zeros((len(other_points_), 1))], axis=1
        )
    elif dimension == 3:
        points_3d = points_
        other_points_3d = other_points_
    else:
        raise ValueError("Point array has the wrong shape.")

    # points in open3d
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    other_point_cloud = o3d.geometry.PointCloud()
    other_point_cloud.points = o3d.utility.Vector3dVector(other_points_3d)

    # initial matrix
    matrix_ = np.identity(dimension) if matrix is None else np.asarray(matrix)
    offset_ = np.zeros(dimension) if offset is None else np.asarray(offset)

    matrix_3d = np.identity(3)
    matrix_3d[:dimension, :dimension] = matrix_
    offset_3d = np.zeros(3)
    offset_3d[:dimension] = offset_
    matrix_homogeneous = _homogeneous_matrix(matrix_3d, offset_3d)

    if pre_translation is not None:
        pre_translation_3d = np.zeros(3)
        pre_translation_3d[:dimension] = pre_translation
        other_point_cloud.translate(pre_translation_3d)

    # apply ICP
    registration = o3d.pipelines.registration.registration_icp(
        source=point_cloud,
        target=other_point_cloud,
        max_correspondence_distance=max_correspondence_distance,
        init=matrix_homogeneous,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=with_scaling
        ),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration
        ),
    )

    if dimension == 2:
        new_matrix = registration.transformation[0:2, 0:2]
        new_offset = registration.transformation[0:2, 3]
    else:  # if dimension == 3:
        new_matrix = registration.transformation[0:3, 0:3]
        new_offset = registration.transformation[0:3, 3]

    if verbose:
        print(registration)

    return Transformation(new_matrix, new_offset)


def register_icp(
    locdata: npt.ArrayLike | LocData,
    other_locdata: npt.ArrayLike | LocData,
    matrix: npt.ArrayLike | None = None,
    offset: npt.ArrayLike | None = None,
    pre_translation: npt.ArrayLike | None = None,
    max_correspondence_distance: float = 1_000,
    max_iteration: int = 10_000,
    verbose: bool = True,
) -> Transformation:
    """
    Register `points` or coordinates in `locdata` by an
    "Iterative Closest Point" algorithm using open3d.

    Parameters
    ----------
    locdata
        Localization data representing the source on which to perform the
        manipulation.
    other_locdata
        Localization data representing the target.
    matrix
        Transformation matrix with shape (d, d) used as initial value.
        If None the unit matrix is used.
    offset
        Translation vector with shape (d,) used as initial value.
        If None a vector of zeros is used.
    pre_translation
        Values for translation of coordinates before registration.
    max_correspondence_distance
        Threshold distance for the icp algorithm. Parameter is passed to
        open3d algorithm.
    max_iteration
        Maximum number of iterations. Parameter is passed to open3d algorithm.
    verbose
        Flag indicating if transformation results are printed out.

    Returns
    -------
    Transformation
        Matrix and offset representing the optimized transformation.
    """
    local_parameter = locals()

    # adjust input
    if isinstance(locdata, LocData):
        points = locdata.coordinates
    else:
        points = np.asarray(locdata)

    if isinstance(other_locdata, LocData):
        other_points = other_locdata.coordinates
    else:
        other_points = np.asarray(other_locdata)

    transformation = _register_icp_open3d(
        points,
        other_points,
        matrix=matrix,
        offset=offset,
        pre_translation=pre_translation,
        max_correspondence_distance=1_000,
        max_iteration=10_000,
        verbose=True,
    )
    return transformation


def _xcorr(imageA: npt.ArrayLike, imageB: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """
    This function is adapted from picasso/imageprocess
    by Joerg Schnitzbauer, MPI of Biochemistry
    https://github.com/jungmannlab/picasso/blob/master/picasso/imageprocess.py
    (MIT license, Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry)
    """
    imageA = np.asarray(imageA)
    imageB = np.asarray(imageB)
    FimageA = np.fft.fft2(imageA)
    CFimageB = np.conj(np.fft.fft2(imageB))
    return_value: npt.NDArray[np.float_] = np.fft.fftshift(
        np.real(np.fft.ifft2(FimageA * CFimageB))
    ) / np.sqrt(imageA.size)
    return return_value


def _get_image_shift(
    imageA: npt.ArrayLike,
    imageB: npt.ArrayLike,
    box: Any,
    roi: Any | None = None,
    display: bool = False,
) -> tuple[int | float, int | float]:
    """
    Computes the shift from imageA to imageB.

    This function is adapted from picasso/imageprocess
    by Joerg Schnitzbauer, MPI of Biochemistry
    https://github.com/jungmannlab/picasso/blob/master/picasso/imageprocess.py
    (MIT license, Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry)
    """
    imageA = np.asarray(imageA)
    imageB = np.asarray(imageB)
    if (np.sum(imageA) == 0) or (np.sum(imageB) == 0):
        return 0, 0
    # Compute image correlation
    XCorr = _xcorr(imageA, imageB)
    # Cut out center roi
    Y, X = imageA.shape
    if roi is not None:
        Y_ = int((Y - roi) / 2)
        X_ = int((X - roi) / 2)
        if Y_ > 0:
            XCorr = XCorr[Y_:-Y_, :]
        else:
            Y_ = 0
        if X_ > 0:
            XCorr = XCorr[:, X_:-X_]
        else:
            X_ = 0
    else:
        Y_ = X_ = 0
    # A quarter of the fit ROI
    fit_X = int(box / 2)
    # A coordinate grid for the fitting ROI
    y, x = np.mgrid[-fit_X : fit_X + 1, -fit_X : fit_X + 1]
    # Find the brightest pixel and cut out the fit ROI
    y_max_, x_max_ = np.unravel_index(XCorr.argmax(), XCorr.shape)
    FitROI = XCorr[
        y_max_ - fit_X : y_max_ + fit_X + 1,
        x_max_ - fit_X : x_max_ + fit_X + 1,
    ]

    dimensions = FitROI.shape

    if 0 in dimensions or dimensions[0] != dimensions[1]:
        xc, yc = 0, 0
    else:
        # The fit model based on lmfit
        def flat_2d_gaussian(a, xc, yc, s, b):  # type: ignore
            A = a * np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / s**2) + b
            return A.flatten()

        gaussian2d = Model(flat_2d_gaussian, name="2D Gaussian", independent_vars=[])

        # Set up initial parameters and fit
        params = Parameters()
        params.add("a", value=FitROI.max(), vary=True, min=0)
        params.add("xc", value=0, vary=True)
        params.add("yc", value=0, vary=True)
        params.add("s", value=1, vary=True, min=0)
        params.add("b", value=FitROI.min(), vary=True, min=0)
        results = gaussian2d.fit(FitROI.flatten(), params)

        # Get maximum coordinates and add offsets
        xc = results.best_values["xc"]
        yc = results.best_values["yc"]
        xc += X_ + x_max_  # type: ignore
        yc += Y_ + y_max_  # type: ignore

        if display:
            plt.figure(figsize=(17, 10))
            plt.subplot(1, 3, 1)  # type: ignore
            plt.imshow(imageA, interpolation="none")
            plt.subplot(1, 3, 2)  # type: ignore
            plt.imshow(imageB, interpolation="none")
            plt.subplot(1, 3, 3)  # type: ignore
            plt.imshow(XCorr, interpolation="none")
            plt.plot(xc, yc, "x")
            plt.show()  # type: ignore

        xc -= np.floor(X / 2)
        yc -= np.floor(Y / 2)

    return -yc, -xc


def register_cc(
    locdata: npt.ArrayLike | LocData,
    other_locdata: npt.ArrayLike | LocData,
    max_offset: int | float | None = None,
    bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
    n_bins: int | Sequence[int] | None = None,
    bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_range: tuple[float, float]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Literal["zero", "link"]
    | None = None,
    verbose: bool = False,
) -> Transformation:
    """
    Register `points` or coordinates in `locdata` by a
    cross-correlation algorithm.

    This function is based on code from picasso/imageprocess
    by Joerg Schnitzbauer, MPI of Biochemistry
    https://github.com/jungmannlab/picasso/blob/master/picasso/imageprocess.py

    Parameters
    ----------
    locdata
        Localization data representing the source on which to perform the
        manipulation.
    other_locdata
        Localization data representing the target.
    max_offset
        Maximum possible offset.
    bins
        Specific class specifying the bins.
    bin_edges
        Bin edges for all or each dimension
        with shape (dimension, n_bin_edges).
    bin_range
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
    n_bins
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size
        The size of bins for all or each bin and for all or each dimension
        with shape (dimension,) or (dimension, n_bins).
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
        (1, 3) for the other dimension.
        To specify arbitrary sequence of `bin_size` use `bin_edges` instead.
    verbose
        Flag indicating if transformation results are printed out.

    Returns
    -------
    Transformation
        Matrix and offset representing the optimized transformation.
    """
    if isinstance(locdata, LocData) and isinstance(other_locdata, LocData):
        bin_range_: tuple[float, float] | Sequence[float] | Sequence[
            Sequence[float]
        ] | Literal["zero", "link"] | None
        if bin_range is None:
            bin_range_ = range_from_collection([locdata, other_locdata])
        else:
            bin_range_ = bin_range

        image, bins_, labels_ = histogram(
            locdata,
            bins=bins,
            n_bins=n_bins,
            bin_size=bin_size,
            bin_edges=bin_edges,
            bin_range=bin_range_,
        )
        other_image, _, _ = histogram(
            other_locdata,
            bins=bins,
            n_bins=n_bins,
            bin_size=bin_size,
            bin_edges=bin_edges,
            bin_range=bin_range_,
        )

    else:
        image = np.asarray(locdata)
        other_image = np.asarray(other_locdata)

    dimension = image.ndim
    matrix = np.identity(dimension)
    # todo: turn box into parameter
    offset = _get_image_shift(
        image, other_image, box=5, roi=max_offset, display=verbose
    )
    offset_ = np.asarray(offset) * bin_size

    return Transformation(matrix, offset_)
