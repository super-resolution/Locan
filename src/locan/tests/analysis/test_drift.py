import matplotlib.pyplot as plt  # this import is needed for visual inspection
import numpy as np
import pytest
from lmfit.model import ModelResult
from lmfit.models import ConstantModel, LinearModel, PolynomialModel

from locan import Drift, DriftComponent, LocData
from locan.analysis.drift import (
    _ConstantModelFacade,
    _ConstantOneModelFacade,
    _ConstantZeroModelFacade,
    _estimate_drift_cc,
    _estimate_drift_icp,
    _LmfitModelFacade,
    _SplineModelFacade,
)
from locan.dependencies import HAS_DEPENDENCY

# data to evaluate fitting
x = np.array([1, 2, 4, 6, 9, 10, 11, 15, 16, 20])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test__LmfitModelFacade():
    model = _LmfitModelFacade(LinearModel())
    model.fit(x=x, y=y, verbose=True)
    assert model.eval(5) == pytest.approx(3.401234567901236)
    assert len(model.eval([2, 5])) == 2
    model.plot()
    # plt.show()

    plt.close("all")


def test__ConstantModelFacade():
    model = _ConstantModelFacade()
    model.fit(x=x, y=y, verbose=True)
    assert model.eval(5) == pytest.approx(5.500000000011998)
    assert len(model.eval([2, 5])) == 2
    model.plot()
    # plt.show()

    plt.close("all")


def test__ConstantZeroModelFacade():
    model = _ConstantZeroModelFacade()
    model.fit(x=x, y=y, verbose=True)
    assert model.eval(5) == 0
    assert len(model.eval([2, 5])) == 2
    model.plot()
    # plt.show()

    plt.close("all")


def test__ConstantOneModelFacade():
    model = _ConstantOneModelFacade()
    model.fit(x=x, y=y, verbose=True)
    assert model.eval(5) == 1
    assert len(model.eval([2, 5])) == 2
    model.plot()
    # plt.show()

    plt.close("all")


def test__SplineModelFacade():
    model = _SplineModelFacade()
    model.fit(x=x, y=y, verbose=True)
    # print(model.eval(5))
    # print(model.eval((3, 5)))
    # print(model.eval([3, 5]))
    # print(model.eval(np.array([3, 5])))
    assert model.eval(5) == pytest.approx(3.417849958061275)
    assert len(model.eval([2, 5])) == 2
    # model.plot()
    model.fit(x=x, y=y, s=0)
    assert model.eval(5) == pytest.approx(3.5332295857553917)
    # model.plot()
    # plt.show()
    model = _SplineModelFacade(s=0)
    model.fit(x=x, y=y, verbose=True)
    assert model.eval(5) == pytest.approx(3.5332295857553917)


def test_DriftComponent():
    drift_component = DriftComponent(type="linear")
    assert drift_component.model_result is None
    with pytest.raises(AttributeError):
        drift_component.eval(3)

    with pytest.raises(TypeError):
        DriftComponent(type="undefined")

    drift_component = DriftComponent(type="linear").fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, LinearModel)
    assert isinstance(drift_component.model_result, ModelResult)
    assert drift_component.eval(4) == pytest.approx(2.924242424242425)
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type="zero").fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, ConstantModel)
    assert isinstance(drift_component.model_result, ModelResult)
    try:
        assert all(drift_component.model_result.best_fit == 0)
    except TypeError:  # needed to work with lmfit<1.2.0
        assert drift_component.model_result.best_fit == 0
    assert drift_component.eval(4) == 0
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type="one").fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, ConstantModel)
    assert isinstance(drift_component.model_result, ModelResult)
    try:
        assert all(drift_component.model_result.best_fit == 1)
    except TypeError:  # needed to work with lmfit<1.2.0
        assert drift_component.model_result.best_fit == 1
    assert drift_component.eval(4) == 1
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type="constant").fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, ConstantModel)
    assert isinstance(drift_component.model_result, ModelResult)
    assert drift_component.model_result.best_fit == pytest.approx(5.500000000011998)
    assert drift_component.eval(4) == pytest.approx(5.500000000011998)
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type="polynomial", degree=2).fit(
        x, y, verbose=True
    )
    assert isinstance(drift_component.model.model, PolynomialModel)
    assert isinstance(drift_component.model_result, ModelResult)

    drift_component = DriftComponent(type="spline").fit(x, y, verbose=True, s=0.3)
    assert drift_component.eval(4) == pytest.approx(3.0972390587725087)
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type=LinearModel()).fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, LinearModel)
    assert isinstance(drift_component.model_result, ModelResult)
    assert drift_component.eval(4) == pytest.approx(2.924242424242425)
    assert len(drift_component.eval([1, 4]) == 2)
    # plt.show()

    plt.close("all")


def test_Drift_empty(caplog):
    drift = Drift().compute(LocData())
    drift.plot()
    drift.apply_correction()
    drift.fit_transformation()
    drift.fit_transformations()
    assert caplog.record_tuples == [
        ("locan.analysis.drift", 30, "Locdata is empty."),
        ("locan.analysis.drift", 30, "No transformations available to be applied."),
        ("locan.analysis.drift", 30, "No transformations available to be fitted."),
        ("locan.analysis.drift", 30, "No transformations available to be fitted."),
    ]


def test__estimate_drift_cc(locdata_blobs_2d):
    # locan.render_2d_mpl(locdata_blobs_2d, bin_size=10, rescale=locan.Trafo.EQUALIZE)
    # plt.show()

    collection, transformations = _estimate_drift_cc(
        locdata_blobs_2d,
        chunks=None,
        chunk_size=25,
        target="first",
        bin_size=10,
        kwargs_chunk=None,
        kwargs_register=None,
    )
    assert len(collection) == 2
    assert isinstance(collection, LocData)
    assert len(transformations) == 2
    assert len(transformations[0].matrix) == 2
    assert len(transformations[0].offset) == 2

    index_groups = (range(0, 25), range(25, 50))
    collection, transformations = _estimate_drift_cc(
        locdata_blobs_2d,
        chunks=index_groups,
        chunk_size=None,
        target="first",
        bin_size=10,
        kwargs_chunk=None,
        kwargs_register=None,
    )
    assert len(collection) == 2
    assert isinstance(collection, LocData)
    assert len(transformations) == 2
    assert len(transformations[0].matrix) == 2
    assert len(transformations[0].offset) == 2

    collection, transformations = _estimate_drift_cc(
        locdata_blobs_2d,
        chunks=None,
        chunk_size=25,
        target="previous",
        bin_size=10,
        kwargs_chunk=None,
        kwargs_register=None,
    )
    assert len(collection) == 2
    assert isinstance(collection, LocData)
    assert len(transformations) == 2
    assert len(transformations[0].matrix) == 2
    assert len(transformations[0].offset) == 2


@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
def test__estimate_drift_icp(locdata_blobs_2d):
    collection, transformations = _estimate_drift_icp(
        locdata_blobs_2d,
        chunks=None,
        chunk_size=25,
        target="first",
        kwargs_chunk=None,
        kwargs_register=None,
    )
    assert len(collection) == 2
    assert isinstance(collection, LocData)
    assert len(transformations) == 2
    assert len(transformations[0].matrix) == 2
    assert len(transformations[0].offset) == 2

    index_groups = (range(0, 25), range(25, 50))
    collection, transformations = _estimate_drift_icp(
        locdata_blobs_2d,
        chunks=index_groups,
        chunk_size=None,
        target="first",
        kwargs_chunk=None,
        kwargs_register=None,
    )
    assert len(collection) == 2
    assert isinstance(collection, LocData)
    assert len(transformations) == 2
    assert len(transformations[0].matrix) == 2
    assert len(transformations[0].offset) == 2

    collection, transformations = _estimate_drift_icp(
        locdata_blobs_2d,
        chunks=None,
        chunk_size=25,
        target="previous",
        kwargs_chunk=None,
        kwargs_register=None,
    )
    assert len(collection) == 2
    assert isinstance(collection, LocData)
    assert len(transformations) == 2
    assert len(transformations[0].matrix) == 2
    assert len(transformations[0].offset) == 2


def test_Drift(locdata_blobs_2d):
    drift = Drift(chunk_size=15, target="first", method="cc").compute(locdata_blobs_2d)
    assert isinstance(drift.locdata, LocData)
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ("matrix", "offset")
    assert len(drift.transformations) == 4
    assert drift.transformation_models["matrix"] is None
    assert drift.transformation_models["offset"] is None

    std_trafo_models = drift._transformation_models_for_identity_matrix()
    assert std_trafo_models["matrix"][0].type == "one"
    std_trafo_models = drift._transformation_models_for_zero_offset()
    assert std_trafo_models["offset"][0].type == "zero"

    assert drift.transformation_models["matrix"] is None
    drift.fit_transformation(
        slice_data=slice(None),
        transformation_component="matrix",
        element=1,
        drift_model="linear",
        verbose=False,
    )
    assert isinstance(drift.transformation_models["matrix"][0], DriftComponent)
    assert drift.transformation_models["matrix"][0].type == "one"
    assert drift.transformation_models["matrix"][1].type == "linear"
    assert drift.transformation_models["matrix"][2].type == "zero"

    assert drift.transformation_models["offset"] is None
    drift.fit_transformation(
        slice_data=slice(None),
        transformation_component="offset",
        element=1,
        drift_model="linear",
        verbose=False,
    )
    assert drift.transformation_models["offset"][0].type == "zero"
    assert drift.transformation_models["offset"][1].type == "linear"

    drift.fit_transformation(
        slice_data=slice(None),
        transformation_component="matrix",
        element=1,
        drift_model=None,
        verbose=False,
    )
    assert drift.transformation_models["matrix"][1].type == "linear"

    drift.fit_transformations(matrix_models=None, offset_models=None, verbose=False)
    assert drift.transformation_models["matrix"] is None
    assert drift.transformation_models["offset"] is None

    drift.fit_transformations(
        slice_data=slice(None),
        matrix_models=(
            "constant",
            "linear",
            DriftComponent(type="polynomial", degree=2),
            None,
        ),
        offset_models=("spline", DriftComponent(type="spline", s=100)),
        verbose=False,
    )
    assert drift.transformation_models["matrix"][0].type == "constant"
    assert drift.transformation_models["matrix"][1].type == "linear"
    assert drift.transformation_models["matrix"][2].type == "polynomial"
    assert drift.transformation_models["matrix"][3].type == "one"
    assert drift.transformation_models["offset"][0].type == "spline"
    assert drift.transformation_models["offset"][1].type == "spline"

    assert drift.locdata_corrected is None
    new_locdata = drift._apply_correction_on_chunks()
    assert len(new_locdata) == len(locdata_blobs_2d)

    drift.transformation_models["matrix"] = None
    drift.transformation_models["offset"] = None
    with pytest.raises(AttributeError):
        drift._apply_correction_from_model(drift.locdata_rapidSTORM_2d)

    drift.fit_transformation(
        slice_data=slice(None),
        transformation_component="matrix",
        element=1,
        drift_model="linear",
        verbose=True,
    )
    assert drift.transformation_models["matrix"][1].type == "linear"
    # print(drift.transformation_models['matrix'][1].model)
    # print(drift.transformation_models['matrix'][1].model_result)
    # print(drift.transformation_models['matrix'][1].eval(3))

    drift.fit_transformation(
        slice_data=slice(None),
        transformation_component="matrix",
        element=0,
        drift_model="one",
        verbose=True,
    )
    assert drift.transformation_models["matrix"][0].type == "one"
    # print(drift.transformation_models['matrix'][0].model)
    # print(drift.transformation_models['matrix'][0].model_result)
    # print(drift.transformation_models['matrix'][0].eval(3))

    drift.fit_transformations(
        slice_data=slice(None),
        matrix_models=["linear"] * 4,
        offset_models=["linear"] * 2,
        verbose=False,
    )
    transformed_points = drift._apply_correction_from_model(drift.locdata)
    assert len(transformed_points) == len(locdata_blobs_2d)

    drift.apply_correction(from_model=True)
    assert isinstance(drift.locdata_corrected, LocData)
    drift.apply_correction(from_model=False)
    assert isinstance(drift.locdata_corrected, LocData)
    drift.apply_correction(locdata=drift.collection.references[0], from_model=True)
    assert isinstance(drift.locdata_corrected, LocData)
    assert len(drift.locdata_corrected) == len(drift.collection.references[0])

    # change target and fit_transformations
    drift = Drift(chunk_size=25, target="previous", method="cc").compute(
        locdata_blobs_2d
    )
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ("matrix", "offset")
    assert len(drift.transformations) == 2

    drift.plot(transformation_component="matrix", element=None)
    drift.plot(transformation_component="matrix", element=3)
    drift.plot(transformation_component="offset", element=None)
    drift.plot(transformation_component="offset", element=0)
    # plt.show()

    with pytest.raises(NotImplementedError):
        drift.fit_transformations(
            slice_data=slice(None),
            matrix_models=["linear"] * 4,
            offset_models=["linear"] * 2,
            verbose=False,
        )

    plt.close("all")


def test_Drift_chain(locdata_blobs_2d):
    # apply chained functions
    drift = (
        Drift(chunk_size=25, target="first", method="cc")
        .compute(locdata_blobs_2d)
        .fit_transformations(
            slice_data=slice(0, 2),
            matrix_models=None,
            offset_models=(None, LinearModel()),
        )
        .apply_correction()
    )
    assert len(drift.locdata_corrected) == len(drift.locdata)
    assert isinstance(drift.locdata, LocData)
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ("matrix", "offset")
    assert len(drift.transformations) == 2


@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
def test_Drift_with_icp(locdata_blobs_2d):
    drift = Drift(chunk_size=15, target="first", method="icp").compute(locdata_blobs_2d)
    assert isinstance(drift.locdata, LocData)
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ("matrix", "offset")
    assert len(drift.transformations) == 4
    assert drift.transformation_models["matrix"] is None
    assert drift.transformation_models["offset"] is None

    assert drift.transformation_models["matrix"] is None
    drift.fit_transformation(
        slice_data=slice(None),
        transformation_component="matrix",
        element=1,
        drift_model="linear",
        verbose=False,
    )
    assert isinstance(drift.transformation_models["matrix"][0], DriftComponent)
    assert drift.transformation_models["matrix"][0].type == "one"
    assert drift.transformation_models["matrix"][1].type == "linear"
    assert drift.transformation_models["matrix"][2].type == "zero"

    assert drift.transformation_models["offset"] is None
    drift.fit_transformation(
        slice_data=slice(None),
        transformation_component="offset",
        element=1,
        drift_model="linear",
        verbose=False,
    )
    assert drift.transformation_models["offset"][0].type == "zero"
    assert drift.transformation_models["offset"][1].type == "linear"
