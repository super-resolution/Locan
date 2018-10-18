import pytest
from pathlib import Path

import numpy as np
import pandas as pd

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.data.rois import Roi_manager
from surepy.analysis.pipeline import Pipeline_test
from surepy.analysis.batch_processing import batch_process

# fixtures

@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


# tests

def test_batch_processingt(locdata_simple):
    result = batch_process([locdata_simple], Pipeline_test)
    assert(result[0].test == True)

    path = ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt'
    result = batch_process([path], Pipeline_test)
    assert(result[0].test == True)

    path = Path(path)
    print(path)
    result = batch_process([path], Pipeline_test)
    assert(result[0].test == True)

    # the following is not working because I cannot save the right path in the yaml file in a machine-independent way.
    # path = ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data_rois.yaml'
    # result = batch_process([path], Pipeline_test)
    # assert(result[0].test == True)

    roim = Roi_manager()
    roim.reference = locdata_simple
    roim.add_rectangle([1,4,1,4])
    result = batch_process([roim], Pipeline_test)
    assert(result[0].test == True)


