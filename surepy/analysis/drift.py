"""
Drift analysis for localization coordinates.

This module provides functions for estimating spatial drift in localization data.

Software based drift correction has been described in several publications [1]_, [2]_, [3]_.
Methods employed for drift estimation comprise single molecule localization analysis or image correlation analysis.

Please use the following procedure to estimate and correct for spatial drift::

    from lmfit import LinearModel
    drift = Drift(chunk_size=1000, target='first').\\
            compute(locdata).\\
            fit_transformations(slice_data=slice(0, -1),
                                matrix_models=None,
                                offset_models=(LinearModel(), LinearModel())).\\
            apply_correction()
    locdata_corrected = drift.locdata_corrected

Note
----
The analysis procedure is in an exploratory state and has not been fully developed and tested.

References
----------
.. [1] C. Geisler,
   Drift estimation for single marker switching based imaging schemes,
   Optics Express. 2012, 20(7):7274-89.

.. [2] Yina Wang et al.,
   Localization events-based sample drift correction for localization microscopy with redundant cross-correlation
   algorithm, Optics Express 2014, 22(13):15982-91.

.. [3] Michael J. Mlodzianoski et al.,
   Sample drift correction in 3D fluorescence photoactivation localization microscopy,
   Opt Express. 2011 Aug 1;19(16):15009-19.

"""
import sys
import warnings
from itertools import accumulate, chain
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from surepy.constants import _has_open3d
if _has_open3d: import open3d as o3d
from surepy.analysis.analysis_base import _Analysis, _list_parameters
from surepy.data.locdata import LocData
from surepy.data.register import _register_icp_open3d, register_cc
from surepy.data.transform.transformation import transform_affine
from surepy.data.metadata_utils import _modify_meta


__all__ = ['Drift']

##### The algorithms

def _estimate_drift_open3d(locdata, chunk_size=1000, target='first'):
    """
    Estimate drift from localization coordinates by registering points in successive time-chunks of localization
    data using an "Iterative Closest Point" algorithm.

    Parameters
    ----------
    locdata : LocData object
       Localization data with properties for coordinates and frame.
    chunk_size : int
       Number of consecutive localizations to form a single chunk of data.
    target : string
       The chunk on which all other chunks are aligned. One of 'first', 'previous'.

    Returns
    -------
    tuple(LocData, list of namedtuple('Transformation', 'matrix offset'))
        Collection and corresponding transformations.
    """
    if not _has_open3d:
        raise ImportError("open3d is required.")

    # split in chunks
    collection = LocData.from_chunks(locdata, chunk_size=chunk_size)

    # register locdatas
    transformations = []
    if target == 'first':
        for locdata in collection.references[1:]:
            transformation = _register_icp_open3d(locdata.coordinates, collection.references[0].coordinates,
                                                  matrix=None, offset=None, pre_translation=None,
                                                  max_correspondence_distance=100, max_iteration=10_000,
                                                  verbose=False)
            transformations.append(transformation)

    elif target == 'previous':
        for n in range(len(collection.references)-1):
            transformation = _register_icp_open3d(collection.references[n+1].coordinates,
                                                  collection.references[n].coordinates,
                                                  matrix=None, offset=None, pre_translation=None,
                                                  max_correspondence_distance=100, max_iteration=10_000,
                                                  with_scaling=False, verbose=False)
            transformations.append(transformation)

    return collection, transformations


def _estimate_drift_cc(locdata, chunk_size=1000, target='first', bin_size=10, **kwargs):
    """
    Estimate drift from localization coordinates by registering points in successive time-chunks of localization
    data using a cross-correlation algorithm.

    Parameters
    ----------
    locdata : LocData object
       Localization data with properties for coordinates and frame.
    chunk_size : int
       Number of consecutive localizations to form a single chunk of data.
    target : string
       The chunk on which all other chunks are aligned. One of 'first', 'previous'.
    bin_size : tuple of int or float
        Size per image pixel

    Other Parameters
    -----------------
    kwargs :
        Other parameters passed to :func:register_cc.

    Returns
    -------
    Locdata and list of namedtuple
        collection and corresponding transformations.
    """
    # split in chunks
    collection = LocData.from_chunks(locdata, chunk_size=chunk_size)
    ranges = [(min, max) for min, max in zip(locdata.coordinates.min(axis=0),
                                             locdata.coordinates.max(axis=0))]

    # register images
    transformations = []
    if target == 'first':
        for reference in collection.references[1:]:
            transformation = register_cc(reference, collection.references[0], range=ranges, bin_size=bin_size, **kwargs)
            transformations.append(transformation)

    elif target == 'previous':
        for n in range(len(collection) - 1):
            transformation = register_cc(collection.references[n + 1], collection.references[n], range=ranges,
                                         bin_size=bin_size, **kwargs)
            transformations.append(transformation)

    return collection, transformations


##### The specific analysis classes

class Drift(_Analysis):
    """
    Estimate drift.

    Parameters
    ----------
    locdata : LocData object
        Localization data representing the source on which to perform the manipulation.
    chunk_size : int
        Number of consecutive localizations to form a single chunk of data.
    target : string
        The chunk on which all other chunks are aligned. One of 'first', 'previous'.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    method : string
        The method (i.e. library or algorithm) used for computation. One of 'open3d', 'cc'.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    locdata : LocData object
        Localization data representing the source on which to perform the manipulation.
    collection : Locdata object
        Collection of locdata chunks
    transformations : list of namedtuple('Transformation', 'matrix offset')
        Transformations for locdata chunks
    transformation_models : namedtuple('Transformation_models', 'matrix offset') with lists of lmfit.ModelResult objects.
        The fitted model objects.
    locdata_corrected : LocData object
        Localization data with drift-corrected coordinates.
    """

    count = 0

    def __init__(self, meta=None, chunk_size=1000, target='first', method='open3d'):
        super().__init__(meta, chunk_size=chunk_size, target=target)
        self.locdata = None
        self.chunk_size = chunk_size
        self.target = target
        self.method = method
        self.collection = None
        self.transformations = None
        self.transformation_models = None
        self.locdata_corrected = None

    def compute(self, locdata, **kwargs):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData object
            Localization data representing the source on which to perform the manipulation.
        bin_size : tuple of int or float
            Only for method='cc': Size per image pixel

        Other Parameters
        -----------------
        kwargs : dict
            Only for method='cc':Other parameters passed to :func:register_cc.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """
        if self.method == 'open3d':
            collection, transformations = _estimate_drift_open3d(locdata=locdata, **self.parameter)
        elif self.method == 'cc':
            collection, transformations = _estimate_drift_cc(locdata=locdata, **self.parameter, **kwargs)
        else:
            raise ValueError(f'Method {self.method} is not defined. One of "open3d", "cc".')
        self.locdata = locdata
        self.collection = collection
        self.transformations = transformations
        self.transformation_models = None
        self.locdata_corrected = None
        return self

    def fit_transformations(self, slice_data=slice(None), matrix_models=None, offset_models=None,
                            verbose=False):
        """
        Fit model parameter to  the estimated transformations.

        Parameters
        ----------
        slice_data : Slice object
            Reduce data to a selection on the localization chunks.
        matrix_models : list of lmfit models or None
            Models to use for fitting each matrix component.
            Length of list must be equal to the square of the transformations dimension (4 or 9).
        offset_models : list of lmfit models or None
            Models to use for fitting each offset component.
            Length of list must be equal to the transformations dimension (2 or 3).
        verbose : bool
            Print the fitted model curves using lmfit.
        """
        dim = self.locdata.dimension
        frames = np.array([locdata.data.frame.mean() for locdata in self.collection.references[1:]])[slice_data]

        def fit_(x, y, model):
            if model is None:
                return None
            else:
                params = model.guess(data=y, x=x)
                fit_results_ = model.fit(data=y, params=params, x=x)
                if verbose:
                    fit_results_.plot()
                return fit_results_

        if self.target == 'first':

            if matrix_models is None:
                model_results_matrix = [None for n in range(dim ** 2)]
            else:
                model_results_matrix = []
                for i, model in enumerate(matrix_models):
                    y = np.array([transformation.matrix[i // dim][i % dim] for transformation in self.transformations])[
                        slice_data]
                    fit_results = fit_(frames, y, model)
                    model_results_matrix.append(fit_results)

            if offset_models is None:
                model_results_offset = [None] * dim
            else:
                model_results_offset = []
                for i, model in enumerate(offset_models):
                    y = np.array([transformation.offset[i] for transformation in self.transformations])[slice_data]
                    fit_results = fit_(frames, y, model)
                    model_results_offset.append(fit_results)

        elif self.target == 'previous':
            raise NotImplementedError("Not yet implemented. Use target = 'first'.")

        Transformation_models = namedtuple('Transformation_models', 'matrix offset')
        self.transformation_models = Transformation_models(model_results_matrix, model_results_offset)
        return self

    def _apply_correction_on_chunks(self):
        """
        Correct drift by applying the estimated transformations to locdata chunks.
        """
        transformed_locdatas = []
        if self.target == 'first':
            transformed_locdatas = [transform_affine(locdata, transformation.matrix, transformation.offset)
                                    for locdata, transformation
                                    in zip(self.collection.references[1:], self.transformations)]

        elif self.target == 'previous':
            for n, locdata in enumerate(self.collection.references[1:]):
                transformed_locdata = locdata
                for transformation in reversed(self.transformations[:n]):
                    transformed_locdata = transform_affine(transformed_locdata,
                                                           transformation.matrix, transformation.offset)
                transformed_locdatas.append(transformed_locdata)

        new_locdata = LocData.concat([self.collection.references[0]] + transformed_locdatas)
        return new_locdata

    def _apply_correction_from_model(self):
        """
        Correct drift by applying the estimated transformations to locdata.
        """
        dimension = self.locdata.dimension
        frames = self.locdata.data.frame
        matrix = np.tile(np.identity(dimension), (len(frames), *([1] * dimension)))
        offset = np.zeros((len(frames), dimension))

        for n, model_result in enumerate(self.transformation_models.matrix):
            if model_result is not None:
                matrix[:, n // dimension, n % dimension] = model_result.eval(x=frames)

        for n, model_result in enumerate(self.transformation_models.offset):
            if model_result is not None:
                offset[:, n] = model_result.eval(x=frames)

        transformed_points = np.einsum("...ij, ...j -> ...i", matrix, self.locdata.coordinates) + offset
        return transformed_points

    def apply_correction(self, locdata=None, from_model=True):
        """
        Correct drift by applying the estimated transformations to locdata.

        Parameters
        ----------
        locdata : LocData or None
            Localization data to apply correction on. If None correction is applied to self.locdata.
        from_model : bool
            If `True` compute transformation matrix from fitted transformation models and apply interpolated
            transformations. If False use the estimated transformation matrix for each data chunk.
        """
        local_parameter = locals()

        if locdata is not None:
            self.locdata = locdata

        if from_model:
            transformed_points = self._apply_correction_from_model()
        else:
            transformed_points = self._apply_correction_on_chunks().coordinates

        # new LocData object
        new_dataframe = self.locdata.data.copy()
        new_dataframe.update(pd.DataFrame(transformed_points, columns=self.locdata.coordinate_labels,
                                          index=self.locdata.data.index))
        new_locdata = LocData.from_dataframe(new_dataframe)

        # update metadata
        meta_ = _modify_meta(self.locdata, new_locdata, function_name=sys._getframe().f_code.co_name,
                            parameter=local_parameter,
                            meta=None)
        new_locdata.meta = meta_

        self.locdata_corrected = new_locdata
        return self

    def plot(self, ax=None, results_field='matrix', element=None, window=1, **kwargs):
        """
        Provide plot as matplotlib axes object showing the running average of results over window size.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image
        results_field : basestring
            One of 'matrix' or 'offset'
        element : int or None
            The element of flattened transformation matrix or offset to be plotted; if None all plots are shown.
        window: int
            Window for running average that is applied before plotting.
            Not implemented yet.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.plot().

        Returns
        -------
        matplotlib Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        n_transformations = len(self.transformations)
        # prepare plot
        x = [reference.data.frame.mean() for reference in self.collection.references[1:]]
        results = np.array([getattr(transformation, results_field) for transformation in self.transformations])
        if element is None:
            ys = results.reshape(n_transformations, -1).T
            for y in ys:
                ax.plot(x, y, **kwargs)
        else:
            y = results.reshape(n_transformations, -1).T[element]
            ax.plot(x, y, **kwargs)

        ax.set(title=f'Drift\n (window={window})',
               xlabel='frame',
               ylabel=''.join([results_field, '[', str(element), ']'])
               )

        return ax
