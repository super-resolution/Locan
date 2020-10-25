import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import matplotlib.pyplot as plt

from surepy import LocData
from surepy.constants import _has_open3d
from surepy.io.io_locdata import load_rapidSTORM_file
from surepy.constants import ROOT_DIR
from surepy.analysis.drift import Drift


@pytest.mark.skipif(not _has_open3d, reason="Test requires open3d.")
def test_Drift():
    locdata = load_rapidSTORM_file(path=ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt')

    drift = Drift(chunk_size=200, target='first').compute(locdata)
    assert isinstance(drift.locdata, LocData)
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ('matrix', 'offset')
    assert len(drift.transformations) == 4

    assert drift.transformation_models is None
    drift.fit_transformations(matrix_models=None, offset_models=None,
                            verbose=False)
    assert drift.transformation_models == ([None, None, None, None], [None, None])

    from lmfit.models import LinearModel
    drift.fit_transformations(slice_data=slice(0, 3),
                              matrix_models=(LinearModel(), None, LinearModel(), LinearModel()),
                              offset_models=(None, LinearModel()),
                              verbose=True)
    assert isinstance(drift.transformation_models.matrix[0].model, LinearModel)
    assert drift.transformation_models.matrix[1] is None
    assert drift.transformation_models.offset[0] is None
    assert isinstance(drift.transformation_models.offset[1].model, LinearModel)
    # plt.show()

    assert drift.locdata_corrected is None
    new_locdata = drift._apply_correction_on_chunks()
    assert len(new_locdata) == len(locdata)

    transformed_points = drift._apply_correction_from_model()
    assert len(transformed_points) == len(locdata)

    drift.apply_correction(from_model=True)
    assert isinstance(drift.locdata_corrected, LocData)
    drift.apply_correction(from_model=False)
    assert isinstance(drift.locdata_corrected, LocData)
    drift.apply_correction(locdata=drift.collection.references[0], from_model=True)
    assert isinstance(drift.locdata_corrected, LocData)
    assert len(drift.locdata_corrected) == len(drift.collection.references[0])

    drift = Drift(chunk_size=200, target='previous').compute(locdata)
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ('matrix', 'offset')
    assert len(drift.transformations) == 4

    drift.plot(results_field='matrix', element=None)
    drift.plot(results_field='matrix', element=3)
    drift.plot(results_field='offset', element=None)
    drift.plot(results_field='offset', element=0)
    # plt.show()

    # apply chained functions
    drift = Drift(chunk_size=200, target='first').\
        compute(locdata).\
        fit_transformations(slice_data=slice(0, 3),
                              matrix_models=None,
                              offset_models=(None, LinearModel())).\
        apply_correction()
    assert len(drift.locdata_corrected) == len(drift.locdata)

    drift = Drift(chunk_size=200, target='first', method='cc').compute(locdata)
    assert isinstance(drift.locdata, LocData)
    assert isinstance(drift.collection, LocData)
    assert drift.transformations[0]._fields == ('matrix', 'offset')
    assert len(drift.transformations) == 4