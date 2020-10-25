"""

Register localization data.

This module registers localization data and provides transformation parameters to put other localization data
in registry.

"""
from collections import namedtuple

import numpy as np
from lmfit import Model, Parameters
import matplotlib.pyplot as plt

from surepy.data.locdata import LocData
from surepy.data.transform.transformation import _homogeneous_matrix
from surepy.render import histogram
from surepy.constants import _has_open3d
if _has_open3d: import open3d as o3d


__all__ = ['register_icp', 'register_cc']


def _register_icp_open3d(points, other_points, matrix=None, offset=None, pre_translation=None,
                         max_correspondence_distance=1_000, max_iteration=10_000, with_scaling=True, verbose=True):
    """
    Register `points` by an "Iterative Closest Point" algorithm using open3d.

    Parameters
    ----------
    points : array-like
        Points representing the source on which to perform the manipulation.
    other_points : array-like
        Points representing the target.
    matrix : tuple with shape (d, d)
        Transformation matrix used as initial value. If None the unit matrix is used.
    offset : tuple of int or float with shape (d,)
        Translation vector used as initial value. If None a vector of zeros is used.
    pre_translation : tuple of int or float
        Values for translation of coordinates before registration.
    max_correspondence_distance : float
        Threshold distance for the icp algorithm. Parameter is passed to open3d algorithm.
    max_iteration : int
        Maximum number of iterations. Parameter is passed to open3d algorithm.
    with_scaling : bool
        Allow scaling transformation. Parameter is passed to open3d algorithm.
    verbose : bool
        Flag indicating if transformation results are printed out.

    Returns
    -------
    namedtuple('Transformation', 'matrix offset')
        Matrix and offset representing the optimized transformation.
    """
    if not _has_open3d:
        raise ImportError("open3d is required.")

    points_ = np.asarray(points)
    other_points_ = np.asarray(other_points)

    # prepare 3d
    if np.shape(points)[1] == np.shape(other_points)[1]:
        dimension = np.shape(points)[1]
    else:
        raise ValueError('Dimensions for locdata and other_locdata are incompatible.')

    if dimension == 2:
        points_3d = np.concatenate([points_, np.zeros((len(points_), 1))], axis=1)
        other_points_3d = np.concatenate([other_points_, np.zeros((len(other_points_), 1))], axis=1)
    elif dimension == 3:
        points_3d = points_
        other_points_3d = other_points_
    else:
        raise ValueError('Point array has the wrong shape.')

    # points in open3d
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    other_point_cloud = o3d.geometry.PointCloud()
    other_point_cloud.points = o3d.utility.Vector3dVector(other_points_3d)

    # initial matrix
    matrix_ = np.identity(dimension) if matrix is None else matrix
    offset_ = np.zeros(dimension) if offset is None else offset

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
    registration = o3d.registration.registration_icp(
        source=point_cloud, target=other_point_cloud,
        max_correspondence_distance=max_correspondence_distance,
        init=matrix_homogeneous,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(with_scaling=with_scaling),
        criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    if dimension == 2:
        new_matrix = registration.transformation[0:2, 0:2]
        new_offset = registration.transformation[0:2, 3]
    else:  # if dimension == 3:
        new_matrix = registration.transformation[0:3, 0:3]
        new_offset = registration.transformation[0:3, 3]

    if verbose:
        print(registration)

    Transformation = namedtuple('Transformation', 'matrix offset')
    return Transformation(new_matrix, new_offset)


def register_icp(locdata, other_locdata, matrix=None, offset=None, pre_translation=None,
                 max_correspondence_distance=1_000, max_iteration=10_000, verbose=True):
    """
    Register `points` or coordinates in `locdata` by an "Iterative Closest Point" algorithm using open3d.

    Parameters
    ----------
    locdata : array-like or LocData object
        Localization data representing the source on which to perform the manipulation.
    other_locdata : array-like or LocData object
        Localization data representing the target.
    matrix : tuple with shape (d, d)
        Transformation matrix used as initial value. If None the unit matrix is used.
    offset : tuple of int or float with shape (d,)
        Translation vector used as initial value. If None a vector of zeros is used.
    pre_translation : tuple of int or float
        Values for translation of coordinates before registration.
    max_correspondence_distance : float
        Threshold distance for the icp algorithm. Parameter is passed to open3d algorithm.
    max_iteration : int
        Maximum number of iterations. Parameter is passed to open3d algorithm.
    verbose : bool
        Flag indicating if transformation results are printed out.

    Returns
    -------
    namedtuple('Transformation', 'matrix offset')
        Matrix and offset representing the optimized transformation.
    """
    local_parameter = locals()

    # adjust input
    if isinstance(locdata, LocData):
        points = locdata.coordinates
    else:
        points = locdata

    if isinstance(other_locdata, LocData):
        other_points = other_locdata.coordinates
    else:
        other_points = other_locdata

    transformation = _register_icp_open3d(points, other_points, matrix=matrix, offset=offset,
                                                  pre_translation=pre_translation,
                                                  max_correspondence_distance=1_000, max_iteration=10_000,
                                                  verbose=True)
    return transformation

def _xcorr(imageA, imageB):
    """
    This function is adapted from picasso/imageprocess by Joerg Schnitzbauer, MPI of Biochemistry
    https://github.com/jungmannlab/picasso/blob/master/picasso/imageprocess.py
    """
    FimageA = np.fft.fft2(imageA)
    CFimageB = np.conj(np.fft.fft2(imageB))
    return np.fft.fftshift(
        np.real(np.fft.ifft2((FimageA * CFimageB)))
        ) / np.sqrt(imageA.size)

def _get_image_shift(imageA, imageB, box, roi=None, display=False):
    """
    Computes the shift from imageA to imageB.

    This function is adapted from picasso/imageprocess by Joerg Schnitzbauer, MPI of Biochemistry
    https://github.com/jungmannlab/picasso/blob/master/picasso/imageprocess.py
    """
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
    y, x = np.mgrid[-fit_X: fit_X + 1, -fit_X: fit_X + 1]
    # Find the brightest pixel and cut out the fit ROI
    y_max_, x_max_ = np.unravel_index(XCorr.argmax(), XCorr.shape)
    FitROI = XCorr[
        y_max_ - fit_X: y_max_ + fit_X + 1,
        x_max_ - fit_X: x_max_ + fit_X + 1,
    ]

    dimensions = FitROI.shape

    if 0 in dimensions or dimensions[0] != dimensions[1]:
        xc, yc = 0, 0
    else:
        # The fit model based on lmfit
        def flat_2d_gaussian(a, xc, yc, s, b):
            A = a * np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / s ** 2) + b
            return A.flatten()

        gaussian2d = Model(
            flat_2d_gaussian, name="2D Gaussian", independent_vars=[]
        )

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
        xc += X_ + x_max_
        yc += Y_ + y_max_

        if display:
            plt.figure(figsize=(17, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(imageA, interpolation="none")
            plt.subplot(1, 3, 2)
            plt.imshow(imageB, interpolation="none")
            plt.subplot(1, 3, 3)
            plt.imshow(XCorr, interpolation="none")
            plt.plot(xc, yc, "x")
            plt.show()

        xc -= np.floor(X / 2)
        yc -= np.floor(Y / 2)

    return -xc, -yc

def register_cc(locdata, other_locdata, max_offset=None, bin_size=None, verbose=False, **kwargs):
    """
    Register `points` or coordinates in `locdata` by a cross-correlation algorithm.

    This function is based on code from picasso/imageprocess by Joerg Schnitzbauer, MPI of Biochemistry
    https://github.com/jungmannlab/picasso/blob/master/picasso/imageprocess.py

    Parameters
    ----------
    locdata : array-like or LocData object
        Localization data representing the source on which to perform the manipulation.
    other_locdata : array-like or LocData object
        Localization data representing the target.
    max_offset : int or float or None
        Maximum possible offset.
    bin_size : tuple of int or float
        Size per image pixel
    verbose : bool
        Flag indicating if transformation results are printed out.

    Other Parameters
    ----------------
    kwargs : dict
        Other parameters passed to :func:`surepy.render.render2d.histogram`.

    Returns
    -------
    namedtuple('Transformation', 'matrix offset')
        Matrix and offset representing the optimized transformation.
    """
    # adjust input
    if isinstance(locdata, LocData) and isinstance(other_locdata, LocData):
        mins = np.minimum(locdata.coordinates.min(axis=0), other_locdata.coordinates.min(axis=0))
        maxs = np.maximum(locdata.coordinates.max(axis=0), other_locdata.coordinates.max(axis=0))
        ranges = [(min, max) for min, max in zip(mins, maxs)]
        image = histogram(locdata, bin_size=bin_size, **dict({'range': ranges}, **kwargs))[0]
        other_image = histogram(other_locdata, bin_size=bin_size, **dict({'range': ranges}, **kwargs))[0]
    else:
        image = np.asarray(locdata)
        other_image = np.asarray(other_locdata)

    dimension = image.ndim
    matrix = np.identity(dimension)
    # todo: make box into parameter
    offset = _get_image_shift(image, other_image, box=5, roi=max_offset, display=verbose)
    offset = tuple(np.asarray(offset) * bin_size)

    Transformation = namedtuple('Transformation', 'matrix offset')
    return Transformation(matrix, offset)
