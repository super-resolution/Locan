import pickle
import tempfile
from pathlib import Path

import pytest

from locan.analysis import metadata_analysis_pb2
from locan.analysis.pipeline import Pipeline, computation_test


def test_Pipeline(locdata_2d):
    with pytest.raises(TypeError):
        Pipeline()
    with pytest.raises(TypeError):
        Pipeline(computation=None, parameter="my_test")

    pipe = Pipeline(computation=computation_test, parameter="my_test")
    assert pipe
    assert isinstance(pipe.meta, metadata_analysis_pb2.AMetadata)
    assert pipe.kwargs == dict(parameter="my_test")
    pipe.compute()
    assert pipe.test == "my_test"

    # this is not recommended since self.parameters are not updated automatically.
    computation_test(pipe, parameter="my_next_test")
    assert pipe.test == "my_next_test"
    assert pipe.kwargs == dict(parameter="my_test")

    # several parameter including locdata reference and piped compute method.
    pipe = Pipeline(
        computation=computation_test, locdata=locdata_2d, parameter="my_test"
    ).compute()
    assert pipe.locdata is locdata_2d
    assert pipe.parameter["locdata"] is locdata_2d
    assert pipe.test == "my_test"

    # print(pipe.computation_as_string())
    assert isinstance(pipe.computation_as_string(), str)

    # save compute as text
    with tempfile.TemporaryDirectory() as tmp_directory:
        # for visual inspection use:
        # file_path = ROOT_DIR / 'tests/test_data/pipe.txt'
        file_path = Path(tmp_directory) / "pipe.txt"
        pipe.save_computation(file_path)
        with open(file_path) as f:
            first_line = f.readline()
            assert first_line == "Analysis Pipeline: Pipeline\n"

    # pickling
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / "pickled_pipeline.pickle"
        with open(file_path, "wb") as file:
            pickle.dump(pipe, file, pickle.HIGHEST_PROTOCOL)
        with open(file_path, "rb") as file:
            pipe_pickled = pickle.load(file)  # noqa S301
        assert len(pipe_pickled.parameter) == len(pipe.parameter)
        assert isinstance(pipe_pickled.meta, metadata_analysis_pb2.AMetadata)
        assert pipe_pickled.meta == pipe.meta
        assert pipe_pickled.test == "my_test"
