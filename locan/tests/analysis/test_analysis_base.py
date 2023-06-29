import pickle
import tempfile
from pathlib import Path

import pytest
from scipy import stats

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis, _list_parameters


def test_Analysis():
    ae = _Analysis(limits=(100, 110), meta={"comment": "this is an example"})
    # print(repr(ae))
    assert str(ae) == repr(ae)
    assert ae.parameter == {"limits": (100, 110)}

    # print(ae.meta)
    # print(dir(metadata_analysis_pb2))
    assert isinstance(ae.meta, metadata_analysis_pb2.AMetadata)
    assert ae.meta.identifier
    assert ae.meta.creation_time
    assert ae.meta.method.name == "_Analysis"
    assert ae.meta.comment == "this is an example"

    ae.meta.comment = "new comment"
    assert ae.meta.comment == "new comment"

    ae.meta.map["variable key"] = "value_1"
    assert str(ae.meta.map) == "{'variable key': 'value_1'}"

    ae.meta.map["key_2"] = "value_2"
    assert ae.meta.map["key_2"] == "value_2"

    with pytest.raises(NotImplementedError):
        ae.compute()


def test_pickling_Analysis():
    ae = _Analysis(limits=(100, 110), meta={"comment": "this is an example"})
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / "pickled_analysis.pickle"
        with open(file_path, "wb") as file:
            pickle.dump(ae, file, pickle.HIGHEST_PROTOCOL)
        with open(file_path, "rb") as file:
            ae_pickled = pickle.load(file)  # noqa S301
        assert ae_pickled.parameter == ae.parameter
        assert isinstance(ae_pickled.meta, metadata_analysis_pb2.AMetadata)
        assert ae_pickled.meta == ae.meta


def test__list_parameters():
    with pytest.raises(AttributeError):
        parameters = _list_parameters("fail")
    parameters = _list_parameters("expon")
    assert parameters == ["loc", "scale"]
    parameters = _list_parameters(stats.gamma)
    assert parameters == ["a", "loc", "scale"]
    parameters = _list_parameters(stats.poisson)
    assert parameters == ["mu", "loc"]
