import pytest
import numpy as np
from lmfit.model import ModelResult
from lmfit.models import ConstantModel, LinearModel, PolynomialModel

from locan import LocData
from locan.constants import _has_open3d
from locan import Drift, DriftComponent
from locan.analysis.drift import _LmfitModelFacade, _ConstantModelFacade, _ConstantZeroModelFacade, \
    _ConstantOneModelFacade, _SplineModelFacade


x = np.array([1, 2, 4, 6, 9, 10, 11, 15, 16, 20])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test__LmfitModelFacade():
    model = _LmfitModelFacade(LinearModel())
    model.fit(x=x, y=y, verbose=True)
    assert model.eval(5) == pytest.approx(3.401234567901236)
    assert len(model.eval([2, 5])) == 2
    model.plot()
    # plt.show()


def test__ConstantModelFacade():
    model = _ConstantModelFacade()
    model.fit(x=x, y=y, verbose=True)
    assert model.eval(5) == pytest.approx(5.500000000011998)
    assert len(model.eval([2, 5])) == 2
    model.plot()
    # plt.show()


def test__ConstantZeroModelFacade():
    model = _ConstantZeroModelFacade()
    model.fit(x=x, y=y, verbose=True)
    assert model.eval(5) == 0
    assert len(model.eval([2, 5])) == 2
    model.plot()
    # plt.show()


def test__ConstantOneModelFacade():
    model = _ConstantOneModelFacade()
    model.fit(x=x, y=y, verbose=True)
    assert model.eval(5) == 1
    assert len(model.eval([2, 5])) == 2
    model.plot()
    # plt.show()


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
    drift_component = DriftComponent(type='linear')
    assert drift_component.model_result is None
    with pytest.raises(AttributeError):
        drift_component.eval(3)

    with pytest.raises(TypeError):
        DriftComponent(type='undefined')

    drift_component = DriftComponent(type='linear').fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, LinearModel)
    assert isinstance(drift_component.model_result, ModelResult)
    assert drift_component.eval(4) == pytest.approx(2.924242424242425)
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type='zero').fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, ConstantModel)
    assert isinstance(drift_component.model_result, ModelResult)
    assert drift_component.model_result.best_fit == 0
    assert drift_component.eval(4) == 0
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type='one').fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, ConstantModel)
    assert isinstance(drift_component.model_result, ModelResult)
    assert drift_component.model_result.best_fit == 1
    assert drift_component.eval(4) == 1
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type='constant').fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, ConstantModel)
    assert isinstance(drift_component.model_result, ModelResult)
    assert drift_component.model_result.best_fit == pytest.approx(5.500000000011998)
    assert drift_component.eval(4) == pytest.approx(5.500000000011998)
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type='polynomial', degree=2).fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, PolynomialModel)
    assert isinstance(drift_component.model_result, ModelResult)

    drift_component = DriftComponent(type='spline').fit(x, y, verbose=True, s=0.3)
    assert drift_component.eval(4) == pytest.approx(3.0972390587725087)
    assert len(drift_component.eval([1, 4]) == 2)

    drift_component = DriftComponent(type=LinearModel()).fit(x, y, verbose=True)
    assert isinstance(drift_component.model.model, LinearModel)
    assert isinstance(drift_component.model_result, ModelResult)
    assert drift_component.eval(4) == pytest.approx(2.924242424242425)
    assert len(drift_component.eval([1, 4]) == 2)
    # plt.show()


def test_Drift_empty(caplog):
    drift = Drift().compute(LocData())
    drift.plot()
    drift.apply_correction()
    drift.fit_transformation()
    drift.fit_transformations()
    assert caplog.record_tuples == [('locan.analysis.drift', 30, 'Locdata is empty.'),
                                    ('locan.analysis.drift', 30, 'No transformations available to be applied.'),
                                    ('locan.analysis.drift', 30, 'No transformations available to be fitted.'),
                                    ('locan.analysis.drift', 30, 'No transformations available to be fitted.')]


@pytest.mark.skipif(not _has_open3d, reason="Test requires open3d.")
def test_Drift(locdata_blobs_2d, locdata_rapidSTORM_2d):
    drift = Drift(chunk_size=100, target='first').compute(locdata_rapidSTORM_2d)
    assert isinstance(drift.locdata, LocData)
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ('matrix', 'offset')
    assert len(drift.transformations) == 10
    assert drift.transformation_models['matrix'] is None
    assert drift.transformation_models['offset'] is None

    std_trafo_models = drift._transformation_models_for_identity_matrix()
    assert std_trafo_models['matrix'][0].type == 'one'
    std_trafo_models = drift._transformation_models_for_zero_offset()
    assert std_trafo_models['offset'][0].type == 'zero'

    assert drift.transformation_models['matrix'] is None
    drift.fit_transformation(slice_data=slice(None), transformation_component='matrix', element=1,
                             drift_model='linear', verbose=False)
    assert isinstance(drift.transformation_models['matrix'][0], DriftComponent)
    assert drift.transformation_models['matrix'][0].type == 'one'
    assert drift.transformation_models['matrix'][1].type == 'linear'
    assert drift.transformation_models['matrix'][2].type == 'zero'

    assert drift.transformation_models['offset'] is None
    drift.fit_transformation(slice_data=slice(None), transformation_component='offset', element=1,
                             drift_model='linear', verbose=False)
    assert drift.transformation_models['offset'][0].type == 'zero'
    assert drift.transformation_models['offset'][1].type == 'linear'

    drift.fit_transformation(slice_data=slice(None), transformation_component='matrix', element=1,
                             drift_model=None, verbose=False)
    assert drift.transformation_models['matrix'][1].type == 'linear'

    drift.fit_transformations(matrix_models=None, offset_models=None, verbose=False)
    assert drift.transformation_models['matrix'] is None
    assert drift.transformation_models['offset'] is None

    drift.fit_transformations(slice_data=slice(None),
                              matrix_models=('constant', 'linear', DriftComponent(type='polynomial', degree=2), None),
                              offset_models=('spline', DriftComponent(type='spline', s=100)),
                              verbose=False)
    assert drift.transformation_models['matrix'][0].type == 'constant'
    assert drift.transformation_models['matrix'][1].type == 'linear'
    assert drift.transformation_models['matrix'][2].type == 'polynomial'
    assert drift.transformation_models['matrix'][3].type == 'one'
    assert drift.transformation_models['offset'][0].type == 'spline'
    assert drift.transformation_models['offset'][1].type == 'spline'

    assert drift.transformation_models['offset'][0].eval(3) == pytest.approx(-6.228, abs=1e-3)
    assert drift.transformation_models['offset'][1].eval(3) == pytest.approx(-2.893, abs=1e-3)

    assert drift.locdata_corrected is None
    new_locdata = drift._apply_correction_on_chunks()
    assert len(new_locdata) == len(locdata_rapidSTORM_2d)

    drift.transformation_models['matrix'] = None
    drift.transformation_models['offset'] = None
    with pytest.raises(AttributeError):
        drift._apply_correction_from_model(drift.locdata_rapidSTORM_2d)

    drift.fit_transformation(slice_data=slice(None), transformation_component='matrix', element=1,
                             drift_model='linear', verbose=True)
    assert drift.transformation_models['matrix'][1].type == 'linear'
    # print(drift.transformation_models['matrix'][1].model)
    # print(drift.transformation_models['matrix'][1].model_result)
    # print(drift.transformation_models['matrix'][1].eval(3))

    drift.fit_transformation(slice_data=slice(None), transformation_component='matrix', element=0,
                             drift_model='one', verbose=True)
    assert drift.transformation_models['matrix'][0].type == 'one'
    # print(drift.transformation_models['matrix'][0].model)
    # print(drift.transformation_models['matrix'][0].model_result)
    # print(drift.transformation_models['matrix'][0].eval(3))

    drift.fit_transformations(slice_data=slice(None),
                              matrix_models=['linear'] * 4,
                              offset_models=['linear'] * 2,
                              verbose=False)
    transformed_points = drift._apply_correction_from_model(drift.locdata)
    assert len(transformed_points) == len(locdata_rapidSTORM_2d)

    drift.apply_correction(from_model=True)
    assert isinstance(drift.locdata_corrected, LocData)
    drift.apply_correction(from_model=False)
    assert isinstance(drift.locdata_corrected, LocData)
    drift.apply_correction(locdata=drift.collection.references[0], from_model=True)
    assert isinstance(drift.locdata_corrected, LocData)
    assert len(drift.locdata_corrected) == len(drift.collection.references[0])

    drift = Drift(chunk_size=200, target='previous').compute(locdata_rapidSTORM_2d)
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ('matrix', 'offset')
    assert len(drift.transformations) == 5

    drift.plot(transformation_component='matrix', element=None)
    drift.plot(transformation_component='matrix', element=3)
    drift.plot(transformation_component='offset', element=None)
    drift.plot(transformation_component='offset', element=0)
    # plt.show()

    with pytest.raises(NotImplementedError):
        drift.fit_transformations(slice_data=slice(None),
                                  matrix_models=['linear'] * 4,
                                  offset_models=['linear'] * 2,
                                  verbose=False)

    # apply chained functions
    drift = Drift(chunk_size=200, target='first', method='cc').\
        compute(locdata_rapidSTORM_2d).\
        fit_transformations(slice_data=slice(0, 3),
                            matrix_models=None,
                            offset_models=(None, LinearModel())).\
        apply_correction()
    assert len(drift.locdata_corrected) == len(drift.locdata)
    assert isinstance(drift.locdata, LocData)
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ('matrix', 'offset')
    assert len(drift.transformations) == 5
