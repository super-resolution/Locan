from pathlib import Path
import tempfile

import pytest
import pandas as pd

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.analysis.pipeline import Pipeline, compute_test, compute_clust


# fixtures

@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        'position_x': [0, 0, 1, 4, 5],
        'position_y': [0, 1, 3, 4, 1],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


# tests

def test_Pipeline(locdata_simple):
    """ use Pipeline by passing an instance """
    pipe = Pipeline(locdata_simple)
    compute_test(pipe)
    assert pipe.test is True
    # print(f'results: {pipe.test}')
    # print(f'meta: {pipe.meta}')


def test_Pipeline_class_method(locdata_simple):
    """ use Pipeline by inheritance - recommended. """
    class MyPipe(Pipeline):
        compute = compute_test

    pipe = MyPipe(locdata_simple)
    pipe.compute()
    assert pipe.test is True
    # print(f'results: {pipe.test}')
    # print(f'meta: {pipe.meta}')

    with tempfile.TemporaryDirectory() as tmp_directory:
        # for visual inspection use:
        # file_path = ROOT_DIR / 'tests/test_data/pipe.txt'
        file_path = Path(tmp_directory) / 'pipe.txt'
        pipe.save_protocol(file_path)
        with open(file_path) as f:
            first_line = f.readline()
            assert first_line == "Analysis Pipeline: MyPipe\n"


def test_Pipeline_from_path_and_roi(locdata_simple):
    class MyPipe(Pipeline):
        compute = compute_test

    pipe = MyPipe(locdata_simple)
    assert isinstance(pipe.locdata, LocData)

    path = ROOT_DIR / 'tests/test_data/five_blobs.txt'
    pipe = MyPipe(dict(file_path=path, file_type=1))
    assert isinstance(pipe.locdata, LocData)

    # todo: remove absolute path in roi.yaml
    # path = ROOT_DIR / 'tests/test_data/roi.yaml'
    # pipe = MyPipe(dict(file_path=path, file_type='roi'))
    # assert isinstance(pipe.locdata, LocData)


def test_Pipeline_clust(locdata_simple):
    class MyPipe(Pipeline):
        compute = compute_clust

    path = ROOT_DIR / 'tests/test_data/five_blobs.txt'
    pipe = MyPipe(dict(file_path=path, file_type=1)).compute()
    assert len(pipe.clust) != 0
    assert isinstance(pipe.locdata, LocData)
