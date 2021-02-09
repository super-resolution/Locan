from pathlib import Path
import tempfile
import pickle

import pytest

from surepy.analysis.analysis_base import _Analysis
from surepy.analysis import metadata_analysis_pb2


def test_Analysis():
    ae = _Analysis(limits=(100, 110), meta={'comment': 'this is an example'})
    # print(repr(ae))
    assert str(ae) == repr(ae)
    assert ae.parameter == {'limits': (100, 110)}

    assert isinstance(ae.meta, metadata_analysis_pb2.AMetadata)
    assert (ae.meta.method.name == "_Analysis")
    assert ae.meta.comment == 'this is an example'

    with pytest.raises(NotImplementedError):
        ae.compute()


def test_pickling_Analysis():
    ae = _Analysis(limits=(100, 110), meta={'comment': 'this is an example'})
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / 'pickled_analysis.pickle'
        with open(file_path, 'wb') as file:
            pickle.dump(ae, file, pickle.HIGHEST_PROTOCOL)
        with open(file_path, 'rb') as file:
            ae_pickled = pickle.load(file)
        assert ae_pickled.parameter == ae.parameter
        assert isinstance(ae_pickled.meta, metadata_analysis_pb2.AMetadata)
        assert ae_pickled.meta == ae.meta
