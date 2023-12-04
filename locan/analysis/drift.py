"""
Drift analysis for localization coordinates.

This module provides functions for estimating spatial drift in localization
data.

Software based drift correction using image correlation has been described in
several publications .
Methods employed for drift estimation comprise single molecule localization
analysis (an iterative closest point (icp)
algorithm as implemented in the open3d library [1]_, [2]_) or image
cross-correlation analysis [3]_, [4]_, [5]_.

Examples
--------
Please use the following procedure to estimate and correct for spatial drift::

    from lmfit import LinearModel
    drift = Drift(chunk_size=1000, target='first').\\
            compute(locdata).\\
            fit_transformations(slice_data=slice(0, -1),
                                matrix_models=None,
                                offset_models=(LinearModel(), LinearModel())).\\
            apply_correction()
    locdata_corrected = drift.locdata_corrected

References
----------
.. [1] Qian-Yi Zhou, Jaesik Park, Vladlen Koltun,
   Open3D: A Modern Library for 3D Data Processing,
   arXiv 2018, 1801.09847

.. [2] Rusinkiewicz and M. Levoy,
   Efficient variants of the ICP algorithm,
   In 3-D Digital Imaging and Modeling, 2001.

.. [3] C. Geisler,
   Drift estimation for single marker switching based imaging schemes,
   Optics Express. 2012, 20(7):7274-89.

.. [4] Yina Wang et al.,
   Localization events-based sample drift correction for localization
   microscopy with redundant cross-correlation
   algorithm, Optics Express 2014, 22(13):15982-91.

.. [5] Michael J. Mlodzianoski et al.,
   Sample drift correction in 3D fluorescence photoactivation localization
   microscopy,
   Opt Express. 2011 Aug 1;19(16):15009-19.

"""
from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import Any, Literal, Protocol

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from lmfit.model import ModelResult as ModelResultLmf
from lmfit.models import ConstantModel, LinearModel, PolynomialModel
from lmfit.models import Model as ModelLmf
from scipy.interpolate import splev, splrep

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta
from locan.data.register import Transformation, _register_icp_open3d, register_cc
from locan.data.transform.spatial_transformation import transform_affine
from locan.dependencies import needs_package

__all__: list[str] = ["Drift", "DriftComponent"]

logger = logging.getLogger(__name__)


class DriftModel(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        ...


# The algorithms


@needs_package("open3d")
def _estimate_drift_icp(
    locdata: LocData,
    chunks: Sequence[tuple[int, ...]] | None = None,
    chunk_size: int | None = None,
    n_chunks: int | None = None,
    target: Literal["first", "previous"] = "first",
    kwargs_chunk: dict[str, Any] | None = None,
    kwargs_register: dict[str, Any] | None = None,
) -> tuple[LocData, list[Transformation]]:
    """
    Estimate drift from localization coordinates by registering points in
    successive time-chunks of localization
    data using an "Iterative Closest Point" algorithm.

    Parameters
    ----------
    locdata
       Localization data with properties for coordinates and frame.
    chunks
        Localization chunks as defined by a list of index-tuples.
        One of `chunks`, `chunk_size` or `n_chunks` must be different
        from None.
    chunk_size
        Number of consecutive localizations to form a single chunk of data.
        One of `chunks`, `chunk_size` or `n_chunks` must be different
        from None.
    n_chunks
        Number of chunks.
        One of `chunks`, `chunk_size` or `n_chunks` must be different
        from None.
    target
       The chunk on which all other chunks are aligned. One of 'first', 'previous'.
    kwargs_chunk
        Other parameter passed to :meth:`LocData.from_chunks`.
    kwargs_register
        Other parameter passed to :func:`_register_icp_open3d`.

    Returns
    -------
    tuple[LocData, list[Transformation]]
        Collection and corresponding transformations.
    """
    if kwargs_chunk is None:
        kwargs_chunk = {}
    if kwargs_register is None:
        kwargs_register = {}

    # split in chunks
    collection = LocData.from_chunks(
        locdata, chunks=chunks, chunk_size=chunk_size, n_chunks=n_chunks, **kwargs_chunk
    )
    assert isinstance(collection.references, Sequence)  # type narrowing # noqa: S101

    # register locdatas
    # initialize with identity transformation for chunk zero.
    transformations = [
        Transformation(np.identity(locdata.dimension), np.zeros(locdata.dimension))
    ]
    if target == "first":
        for locdata in collection.references[1:]:
            transformation = _register_icp_open3d(
                locdata.coordinates,
                collection.references[0].coordinates,
                **dict(
                    dict(
                        matrix=None,
                        offset=None,
                        pre_translation=None,
                        max_correspondence_distance=100,
                        max_iteration=10_000,
                        verbose=False,
                    ),
                    **kwargs_register,
                ),
            )
            transformations.append(transformation)

    elif target == "previous":
        for n in range(len(collection.references) - 1):
            transformation = _register_icp_open3d(
                collection.references[n + 1].coordinates,
                collection.references[n].coordinates,
                **dict(
                    dict(
                        matrix=None,
                        offset=None,
                        pre_translation=None,
                        max_correspondence_distance=100,
                        max_iteration=10_000,
                        with_scaling=False,
                        verbose=False,
                    ),
                    **kwargs_register,
                ),
            )
            transformations.append(transformation)

    return collection, transformations


def _estimate_drift_cc(
    locdata: LocData,
    chunks: Sequence[tuple[int, ...]] | None = None,
    chunk_size: int | None = None,
    n_chunks: int | None = None,
    target: Literal["first", "previous"] = "first",
    bin_size: int | float | tuple[int | float] = 10,
    kwargs_chunk: dict[str, Any] | None = None,
    kwargs_register: dict[str, Any] | None = None,
) -> tuple[LocData, list[Transformation]]:
    """
    Estimate drift from localization coordinates by registering points in
    successive time-chunks of localization
    data using a cross-correlation algorithm.

    Parameters
    ----------
    locdata
       Localization data with properties for coordinates and frame.
    chunks
        Localization chunks as defined by a list of index-tuples.
        One of `chunks`, `chunk_size` or `n_chunks` must be different
        from None.
    chunk_size
        Number of consecutive localizations to form a single chunk of data.
        One of `chunks`, `chunk_size` or `n_chunks` must be different
        from None.
    n_chunks
        Number of chunks.
        One of `chunks`, `chunk_size` or `n_chunks` must be different
        from None.
    target
       The chunk on which all other chunks are aligned. One of 'first', 'previous'.
    bin_size
        Size per image pixel
    kwargs_chunk
        Other parameter passed to :meth:`LocData.from_chunks`.
    kwargs_register
        Other parameter passed to :func:`register_cc`.

    Returns
    -------
    tuple[LocData, list[Transformation]]
        Collection and corresponding transformations.
    """
    if kwargs_chunk is None:
        kwargs_chunk = {}
    if kwargs_register is None:
        kwargs_register = {}

    # split in chunks
    collection = LocData.from_chunks(
        locdata, chunks=chunks, chunk_size=chunk_size, n_chunks=n_chunks, **kwargs_chunk
    )
    assert isinstance(collection.references, Sequence)  # type narrowing # noqa: S101
    assert locdata.bounding_box is not None  # type narrowing # noqa: S101
    ranges = locdata.bounding_box.hull.T

    # register images
    # initialize with identity transformation for chunk zero.
    transformations = [
        Transformation(np.identity(locdata.dimension), np.zeros(locdata.dimension))
    ]
    if target == "first":
        for reference in collection.references[1:]:
            transformation = register_cc(
                reference,
                collection.references[0],
                **dict(dict(bin_range=ranges, bin_size=bin_size), **kwargs_register),
            )
            transformations.append(transformation)

    elif target == "previous":
        for n in range(len(collection) - 1):
            transformation = register_cc(
                collection.references[n + 1],
                collection.references[n],
                **dict(dict(bin_range=ranges, bin_size=bin_size), **kwargs_register),
            )
            transformations.append(transformation)

    return collection, transformations


# The specific analysis classes


class _LmfitModelFacade:
    def __init__(self, model: ModelLmf) -> None:
        self.model: ModelLmf = model
        self.model_result: ModelResultLmf | None = None

    def fit(
        self, x: npt.ArrayLike, y: npt.ArrayLike, verbose: bool = False, **kwargs: Any
    ) -> ModelResultLmf:
        params = self.model.guess(data=y, x=x)
        self.model_result = self.model.fit(x=x, data=y, params=params, **kwargs)
        if verbose:
            self.plot()
        return self.model_result

    def eval(self, x: npt.ArrayLike) -> npt.NDArray[np.float_]:
        x = np.asarray(x)
        if self.model_result is None:
            raise AttributeError("No model_result available. Run fit method first.")
        return_value: npt.NDArray[np.float_] = self.model_result.eval(x=x)
        return return_value

    def plot(self, **kwargs: Any) -> mpl.axes.Axes:
        if self.model_result is None:
            raise AttributeError("No model_result available. Run fit method first.")
        return_value: mpl.axes.Axes = self.model_result.plot(**kwargs)
        return return_value


class _ConstantModelFacade:
    def __init__(self, **kwargs: Any) -> None:
        self.model: ModelLmf = ConstantModel(**kwargs)
        self.model_result: ModelResultLmf | None = None
        self.independent_variable: npt.NDArray[Any] | None = None

    def fit(
        self, x: npt.ArrayLike, y: npt.ArrayLike, verbose: bool = False, **kwargs: Any
    ) -> ModelResultLmf:
        self.independent_variable = np.asarray(x)
        self.model_result = self.model.fit(x=x, data=y, **kwargs)
        if verbose:
            self.plot()
        return self.model_result

    def eval(self, x: npt.ArrayLike) -> npt.NDArray[np.float_]:
        x = np.asarray(x)
        if self.model_result is None:
            raise AttributeError("No model_result available. Run fit method first.")
        result: npt.NDArray[np.float_] = self.model_result.eval(x=x)
        if np.shape(result) != np.shape(x):  # needed to work with lmfit<1.2.0
            result = np.full(shape=np.shape(x), fill_value=result)
        return result

    def plot(self, **kwargs: Any) -> mpl.axes.Axes:
        if self.model_result is None:
            raise AttributeError("No model_result available. Run fit method first.")
        assert self.independent_variable is not None  # type narrowing # noqa: S101
        assert self.model_result.data is not None  # type narrowing # noqa: S101
        x = self.independent_variable
        y = self.model_result.data
        return_value: mpl.axes.Axes = plt.plot(x, y, "o", x, self.eval(x=x), **kwargs)  # type: ignore[assignment]
        return return_value


class _ConstantZeroModelFacade(_ConstantModelFacade):
    def __init__(self) -> None:
        self.model: ModelLmf = ConstantModel()
        self.model.set_param_hint(name="c", value=0, vary=False)
        self.model_result: ModelResultLmf | None = None


class _ConstantOneModelFacade(_ConstantModelFacade):
    def __init__(self) -> None:
        self.model: ModelLmf = ConstantModel()
        self.model.set_param_hint(name="c", value=1, vary=False)
        self.model_result: ModelResultLmf | None = None


class _SplineModelFacade:
    def __init__(self, **kwargs: Any) -> None:
        self.model: str = "spline"
        self.model_result: tuple[Any, Any, Any] | None = None
        self.parameter = kwargs
        self.independent_variable: npt.NDArray[Any] | None = None
        self.data: npt.NDArray[np.float_] | None = None

    def fit(
        self, x: npt.ArrayLike, y: npt.ArrayLike, verbose: bool = False, **kwargs: Any
    ) -> tuple[Any, Any, Any]:
        self.independent_variable = np.asarray(x)
        self.data = np.asarray(y)
        self.model_result = splrep(
            x, y, **dict(dict(k=3, s=100), **dict(**self.parameter, **kwargs))
        )
        if verbose:
            self.plot()
        assert self.model_result is not None  # type narrowing # noqa: S101
        return self.model_result

    def eval(self, x: npt.ArrayLike) -> float | npt.NDArray[np.float_] | list[Any]:
        x = np.asarray(x)
        if self.model_result is None:
            raise AttributeError("No model_result available. Run fit method first.")
        results: npt.NDArray[np.float_] | list[Any] = splev(x, self.model_result)
        if isinstance(x, (tuple, list, np.ndarray)):
            return results
        else:
            return float(results)

    def plot(self, **kwargs: Any) -> mpl.axes.Axes:
        if self.model_result is None:
            raise AttributeError("No model_result available. Run fit method first.")
        x = self.independent_variable
        y = self.data
        x_ = np.linspace(np.min(x), np.max(x), 100)  # type: ignore[arg-type]

        return_value: mpl.axes.Axes = plt.plot(x, y, "o", x_, self.eval(x=x_), **kwargs)  # type: ignore
        return return_value


class DriftComponent:
    """
    Class carrying model functions to describe drift over time
    (in unit of frames).
    DriftComponent provides a transformation to apply a drift correction.

    Standard models for constant, linear or polynomial drift correction are
    taken from :mod:`lmfit.models`.
    For fitting splines we use the scipy function :func:`scipy.interpolate.splrep`.

    Parameters
    ----------
    type : Literal["none", "zero", "one", "constant", "linear", "polynomial", "spline"] | lmfit.models.Model | None
        Model class or indicator for setting up the corresponding model class.

    Attributes
    ----------
    type
        String indicator for model.
    model : lmfit.models.Model | None
        The model definition (return value of :func:`scipy.interpolate.splrep`)
    model_result : lmfit.model.ModelResult, collection of model results.
        The results collected from fitting the model to specified data.
    """

    def __init__(
        self,
        type: Literal[
            "none", "zero", "one", "constant", "linear", "polynomial", "spline"
        ]
        | ModelLmf
        | None = None,
        **kwargs: Any,
    ) -> None:
        self.type = type
        self.model: DriftModel | None
        self.model_result = None

        if type is None:
            self.type = "none"
            self.model = None
        elif type == "zero":
            self.model = _ConstantZeroModelFacade()
        elif type == "one":
            self.model = _ConstantOneModelFacade()
        elif type == "constant":
            self.model = _ConstantModelFacade(**kwargs)
        elif type == "linear":
            self.model = _LmfitModelFacade(LinearModel(**kwargs))
        elif type == "polynomial":
            self.model = _LmfitModelFacade(
                PolynomialModel(**dict(dict(degree=3), **kwargs))
            )
        elif getattr(type, "__module__", None) == "lmfit.models":
            self.type = type.name  # type: ignore
            self.model = _LmfitModelFacade(model=type)
        elif type == "spline":
            self.model = _SplineModelFacade(**kwargs)
        else:
            raise TypeError(f"DriftComponent cannot handle type={type}.")

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Fit model to the given data and create `self.model_results`.

        Parameters
        ----------
        x
            x data
        y
            y values
        verbose
            show plot
        kwargs
            Other parameters passed to :func:`lmfit.model.fit` or to
            :func:`scipy.interpolate.splrep`
            Use the parameter `s` to set the amount of smoothing.

        Returns
        -------
        Self
        """
        if self.model is None:
            raise ValueError("No model available.")

        self.model_result = self.model.fit(x, y, verbose=verbose, **kwargs)
        return self

    def eval(self, x: npt.ArrayLike) -> float | npt.NDArray[np.float_] | list[Any]:
        """
        Compute a transformation for time `x` from the drift model.

        Parameters
        ----------
        x
            frame values

        Returns
        -------
        npt.NDArray[np.float_].
        """
        if self.model is None:
            raise ValueError("No model available.")
        return_value: float | npt.NDArray[np.float_] | list[Any] = self.model.eval(x)
        return return_value


class Drift(_Analysis):
    """
    Estimate drift from localization coordinates by registering points in
    successive time-chunks of localization
    data using an iterative closest point algorithm (icp) or image
    cross-correlation algorithm (cc).

    Parameters
    ----------
    locdata : LocData
        Localization data representing the source on which to perform the manipulation.
    chunks : Sequence[tuple[int, ...]] | None
        Localization chunks as defined by a list of index-tuples.
        One of `chunks`, `chunk_size` or `n_chunks` must be different
        from None.
    chunk_size : int | None
        Number of consecutive localizations to form a single chunk of data.
        One of `chunks`, `chunk_size` or `n_chunks` must be different
        from None.
    n_chunks : int | None
        Number of chunks.
        One of `chunks`, `chunk_size` or `n_chunks` must be different
        from None.
    target : Literal["first", "previous"]
        The chunk on which all other chunks are aligned.
        One of 'first', 'previous'.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    method : Literal["cc", "icp"]
        The method used for computation.
        One of iterative closest point algorithm 'icp' or image
        cross-correlation algorithm 'cc'.
    bin_size : tuple
        Only for method='cc': Size per image pixel
    kwargs_chunk : dict[str, Any] | None
        Other parameter passed to :meth:`LocData.from_chunks`.
    kwargs_icp : dict[str, Any] | None
        Other parameter passed to :func:`_register_icp_open3d`.
    kwargs_cc : dict[str, Any] | None
        Other parameter passed to :func:`register_cc`.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    locdata : LocData | None
        Localization data representing the source on which to perform the manipulation.
    collection : LocData | None
        Collection of locdata chunks
    transformations : list[Transformation] | None
        Transformations for locdata chunks
    transformation_models : dict[str, list[DriftModel | DriftComponent] | None]
        The fitted model objects.
    locdata_corrected : LocData | None
        Localization data with drift-corrected coordinates.
    """

    count = 0

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        chunks: list[tuple[int, ...]] | None = None,
        chunk_size: int | None = None,
        n_chunks: int | None = None,
        target: Literal["first", "previous"] = "first",
        method: Literal["cc", "icp"] = "icp",
        kwargs_chunk: dict[str, Any] | None = None,
        kwargs_register: dict[str, Any] | None = None,
    ):
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.locdata: LocData | None = None
        self.collection: LocData | None = None
        self.transformations: list[Transformation] | None = None
        self.transformation_models: dict[
            str, list[DriftModel | DriftComponent] | None
        ] = dict(matrix=None, offset=None)
        self.locdata_corrected: LocData | None = None

    def __bool__(self) -> bool:
        if self.transformations is not None:
            return True
        else:
            return False

    def compute(self, locdata: LocData) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data representing the source on which to perform the
            manipulation.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        if self.parameter["method"] == "icp":
            collection, transformations = _estimate_drift_icp(
                locdata,
                chunks=self.parameter["chunks"],
                chunk_size=self.parameter["chunk_size"],
                n_chunks=self.parameter["n_chunks"],
                kwargs_chunk=self.parameter["kwargs_chunk"],
                kwargs_register=self.parameter["kwargs_register"],
            )
        elif self.parameter["method"] == "cc":
            collection, transformations = _estimate_drift_cc(
                locdata,
                chunks=self.parameter["chunks"],
                chunk_size=self.parameter["chunk_size"],
                n_chunks=self.parameter["n_chunks"],
                kwargs_chunk=self.parameter["kwargs_chunk"],
                kwargs_register=self.parameter["kwargs_register"],
            )
        else:
            raise ValueError(
                f'Method {self.parameter["method"]} is not defined. One of "icp", "cc".'
            )
        self.locdata = locdata
        self.collection = collection
        self.transformations = transformations
        self.transformation_models = dict(matrix=None, offset=None)
        self.locdata_corrected = None
        return self

    def _transformation_models_for_identity_matrix(
        self,
    ) -> dict[str, list[DriftModel | DriftComponent]]:
        """
        Return transformation_models (dict) with DriftModels according to unit
        matrix.
        """
        dimension = self.locdata.dimension  # type: ignore[union-attr]
        transformation_models: list[DriftModel | DriftComponent] = []
        for k in np.identity(dimension).flatten():
            if k == 0:
                transformation_models.append(DriftComponent("zero"))
            else:  # if k == 1
                transformation_models.append(DriftComponent("one"))
        return dict(matrix=transformation_models)

    def _transformation_models_for_zero_offset(
        self,
    ) -> dict[str, list[DriftModel | DriftComponent]]:
        """
        Return transformation_models (dict) with DriftModels according to zero
        offset.
        """
        dimension = self.locdata.dimension  # type: ignore[union-attr]
        return dict(offset=[DriftComponent("zero") for _ in range(dimension)])

    def fit_transformation(
        self,
        slice_data: slice | None = None,
        transformation_component: Literal["matrix", "offset"] = "matrix",
        element: int = 0,
        drift_model: DriftComponent | str | None = "linear",
        verbose: bool = False,
    ) -> Self:
        """
        Fit drift model to selected component of the estimated transformations.

        Parameters
        ----------
        slice_data
            Reduce data to a selection on the localization chunks.
        transformation_component
            One of 'matrix' or 'offset'
        element
            The element of flattened transformation matrix or offset
        drift_model
            A drift model as defined by a :class:`DriftComponent` instance
            or the parameter `type` as defined in :class:`DriftComponent`.
            For None no change will occur. To reset transformation_models set
            the transformation_component to None:
            e.g. self.transformation_components = None or
            self.transformation_components['matrix'] = None.
        verbose
            Print the fitted model curves using lmfit.

        Returns
        -------
        Self
        """
        if self.transformations is None:
            logger.warning("No transformations available to be fitted.")
            return self

        if slice_data is None:
            slice_data = slice(None)

        dimension = self.locdata.dimension  # type: ignore[union-attr]
        # frames = np.array([locdata.data.frame.mean() for locdata in self.collection.references[1:]])[slice_data]
        frames = np.array(
            [locdata.data.frame.mean() for locdata in self.collection.references]  # type: ignore[union-attr]
        )[slice_data]

        if drift_model is None:
            return self

        if not isinstance(drift_model, DriftComponent):
            drift_model = DriftComponent(type=drift_model)

        if self.parameter["target"] == "first":
            if transformation_component == "matrix":
                y = np.array(
                    [
                        transformation.matrix[element // dimension][element % dimension]
                        for transformation in self.transformations
                    ]
                )[slice_data]
                if self.transformation_models["matrix"] is None:
                    self.transformation_models.update(
                        self._transformation_models_for_identity_matrix()
                    )
            elif transformation_component == "offset":
                y = np.array(
                    [
                        transformation.offset[element]
                        for transformation in self.transformations
                    ]
                )[slice_data]
                if self.transformation_models["offset"] is None:
                    self.transformation_models.update(
                        self._transformation_models_for_zero_offset()
                    )
            else:
                raise ValueError(
                    "transformation_component must be 'matrix' or 'offset'."
                )

            self.transformation_models[transformation_component][element] = drift_model  # type: ignore[index]
            self.transformation_models[transformation_component][element].fit(  # type: ignore[index]
                frames, y, verbose=False
            )

            if verbose:
                self.transformation_models[transformation_component][  # type: ignore[index,union-attr]
                    element
                ].model.plot()

        elif self.parameter["target"] == "previous":
            raise NotImplementedError("Not yet implemented. Use target = 'first'.")

        return self

    def fit_transformations(
        self,
        slice_data: slice | None = None,
        matrix_models: list[DriftComponent] | None = None,
        offset_models: list[DriftComponent] | None = None,
        verbose: bool = False,
    ) -> Self:
        """
        Fit model parameter to all estimated transformation components.

        Parameters
        ----------
        slice_data
            Reduce data to a selection on the localization chunks.
        matrix_models
            Models to use for fitting each matrix component.
            Length of list must be equal to the square of the transformations
            dimension (4 or 9).
            If None, no matrix transformation will be carried out when calling
            :func:`apply_correction`.
        offset_models
            Models to use for fitting each offset component.
            Length of list must be equal to the transformations dimension
            (2 or 3).
            If None, no offset transformation will be carried out when calling
            :func:`apply_correction`.
        verbose
            Print the fitted model curves.

        Returns
        -------
        Self
        """
        if self.transformations is None:
            logger.warning("No transformations available to be fitted.")
            return self

        if slice_data is None:
            slice_data = slice(None)

        if matrix_models is None:
            self.transformation_models["matrix"] = None
        else:
            if len(matrix_models) != self.locdata.dimension**2:  # type: ignore[union-attr]
                raise ValueError(
                    "Length of matrix_models must be equal to the square of the "
                    "transformations dimension (4 or 9)."
                )
            self.transformation_models.update(
                self._transformation_models_for_identity_matrix()
            )
            for n, matrix_model in enumerate(matrix_models):
                if matrix_model is None:
                    matrix_model = self.transformation_models["matrix"][n]
                self.fit_transformation(
                    slice_data=slice_data,
                    transformation_component="matrix",
                    element=n,
                    drift_model=matrix_model,
                    verbose=verbose,
                )

        if offset_models is None:
            self.transformation_models["offset"] = None
        else:
            if len(offset_models) != self.locdata.dimension:  # type: ignore[union-attr]
                raise ValueError(
                    "Length of offset_models must be equal to the the transformations dimension (2 or 3)."
                )
            self.transformation_models.update(
                self._transformation_models_for_zero_offset()
            )
            for n, offset_model in enumerate(offset_models):
                if offset_model is None:
                    offset_model = self.transformation_models["offset"][n]
                self.fit_transformation(
                    slice_data=slice_data,
                    transformation_component="offset",
                    element=n,
                    drift_model=offset_model,
                    verbose=verbose,
                )

        return self

    def _apply_correction_on_chunks(self) -> LocData:
        """
        Correct drift by applying the estimated transformations to locdata chunks.
        """
        transformed_locdatas: list[LocData] = []
        assert self.transformations is not None  # type narrowing # noqa: S101
        assert (  # type narrowing # noqa: S101
            self.collection is not None
            and isinstance(self.collection.references, Sequence)
        )

        if self.parameter["target"] == "first":
            transformed_locdatas = [
                transform_affine(locdata, transformation.matrix, transformation.offset)  # type: ignore
                for locdata, transformation in zip(
                    self.collection.references[1:], self.transformations
                )
            ]

        elif self.parameter["target"] == "previous":
            for n, locdata in enumerate(self.collection.references[1:]):
                transformed_locdata = locdata
                for transformation in reversed(self.transformations[:n]):
                    transformed_locdata = transform_affine(
                        transformed_locdata,
                        transformation.matrix,
                        transformation.offset,
                    )
                transformed_locdatas.append(transformed_locdata)

        new_locdata = LocData.concat(
            [self.collection.references[0]] + transformed_locdatas
        )
        return new_locdata

    def _apply_correction_from_model(self, locdata: LocData) -> npt.NDArray[Any]:
        """
        Correct drift by applying the estimated transformations to locdata.
        If self.transformation_model['matrix'] is None, no matrix
        transformation will be carried out when calling
        :func:`apply_correction` (same for 'offset').

        Parameters
        ----------
        locdata
            Localization data to apply correction on. If None correction is
            applied to self.locdata.
        """
        # check if any models are fitted,
        # otherwise it is likely that the fitting procedure was accidentally omitted.
        if (
            self.transformation_models["matrix"] is None
            and self.transformation_models["offset"] is None
        ):
            raise AttributeError(
                "The transformation_models have to be fitted before they can be evaluated."
            )

        dimension = locdata.dimension
        frames = locdata.data.frame.values
        matrix = np.tile(np.identity(dimension), (len(frames), *([1] * dimension)))
        offset = np.zeros((len(frames), dimension))

        if self.transformation_models["matrix"] is None:
            transformed_points = locdata.coordinates
        else:
            for n, drift_model in enumerate(self.transformation_models["matrix"]):
                matrix[:, n // dimension, n % dimension] = drift_model.eval(frames)  # type: ignore
            transformed_points = np.einsum(
                "...ij, ...j -> ...i", matrix, locdata.coordinates
            )

        if self.transformation_models["offset"] is None:
            pass
        else:
            for n, drift_model in enumerate(self.transformation_models["offset"]):
                offset[:, n] = drift_model.eval(frames)  # type: ignore
            transformed_points += offset

        return transformed_points

    def apply_correction(
        self, locdata: LocData | None = None, from_model: bool = True
    ) -> Self:
        """
        Correct drift by applying the estimated transformations to locdata.

        Parameters
        ----------
        locdata
            Localization data to apply correction on. If None correction is
            applied to self.locdata.
        from_model
            If `True` compute transformation matrix from fitted transformation
            models and apply interpolated
            transformations. If False use the estimated transformation matrix
            for each data chunk.

        Returns
        -------
        Self
        """
        if self.transformations is None:
            logger.warning("No transformations available to be applied.")
            return self

        local_parameter = locals()

        if locdata is None:
            locdata_orig: LocData = self.locdata  # type: ignore[assignment]
        else:
            locdata_orig = locdata

        if locdata_orig is None:
            logger.warning("Locdata is None.")
            self.locdata_corrected = locdata_orig
            return self

        elif not len(locdata_orig):
            logger.warning("Locdata is empty.")
            self.locdata_corrected = locdata_orig
            return self

        if from_model:
            transformed_points = self._apply_correction_from_model(locdata=locdata_orig)
        else:
            if locdata is not None:
                raise TypeError(
                    "Locdata must be None since correction can only be applied to original locdata chunks."
                )
            transformed_points = self._apply_correction_on_chunks().coordinates

        # new LocData object
        new_dataframe = locdata_orig.data.copy()
        new_dataframe.update(
            pd.DataFrame(
                transformed_points,
                columns=locdata_orig.coordinate_keys,
                index=locdata_orig.data.index,
            )
        )
        new_locdata = LocData.from_dataframe(new_dataframe)

        # update metadata
        meta_ = _modify_meta(
            locdata_orig,
            new_locdata,
            function_name=sys._getframe().f_code.co_name,
            parameter=local_parameter,
            meta=None,
        )
        new_locdata.meta = meta_

        self.locdata_corrected = new_locdata
        return self

    def plot(
        self,
        ax: mpl.axes.Axes | None = None,
        transformation_component: Literal["matrix", "offset"] = "matrix",
        element: int | None = None,
        window: int = 1,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Plot the transformation components as function of average frame for
        each locdata chunk.

        Parameters
        ----------
        ax
            The axes on which to show the image
        transformation_component
            One of 'matrix' or 'offset'
        element
            The element of flattened transformation matrix or offset to be
            plotted; if None all plots are shown.
        window
            Window for running average that is applied before plotting.
            Not implemented yet.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.transformations is None:
            return ax

        n_transformations = len(self.transformations)
        # prepare plot
        x = [reference.data.frame.mean() for reference in self.collection.references]  # type: ignore[union-attr]
        results = np.array(
            [
                getattr(transformation, transformation_component)
                for transformation in self.transformations
            ]
        )
        if element is None:
            ys = results.reshape(n_transformations, -1).T
            for i, y in enumerate(ys):
                ax.plot(
                    x,
                    y,
                    **dict(dict(label=f"{transformation_component}[{i}]"), **kwargs),
                )
        else:
            y = results.reshape(n_transformations, -1).T[element]
            ax.plot(
                x,
                y,
                **dict(dict(label=f"{transformation_component}[{element}]"), **kwargs),
            )

        ax.set(
            title=f"Drift\n (window={window})",
            xlabel="frame",
            ylabel="".join([transformation_component]),
        )

        return ax
