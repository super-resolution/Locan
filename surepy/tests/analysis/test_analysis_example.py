import pytest

from surepy.analysis.analysis_example import AnalysisExampleAlgorithm_1
from surepy.analysis import metadata_analysis_pb2


def test_Analysis_example():
    ae = AnalysisExampleAlgorithm_1(limits=(100, 110), meta={'comment': 'this is an example'})
    # print(ae)
    assert str(ae) == repr(ae)
    assert ae.parameter == {'limits': (100, 110)}
    assert ae.meta.comment == 'this is an example'
    ae.compute(locdata=None)
    assert (len(list(ae.results)) == 10)
    ae.plot()
    ae_2 = eval(repr(ae))
    assert ae_2.parameter == {'limits': (100, 110)}
