import pytest

import surepy.constants
import surepy.io.io_locdata as io
from surepy.analysis.analysis_example import AnalysisExampleAlgorithm_1
import surepy.tests.test_data


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
                                   nrows=100)


def test_Analysis_example(locdata):
    ae = AnalysisExampleAlgorithm_1(limits=(100, 110), meta={'comment': 'this is an example'})
    # print(ae)
    assert ae.parameter == {'limits': (100, 110)}
    assert ae.meta.comment == 'this is an example'
    ae.compute(locdata=None)
    assert (len(list(ae.results)) == 10)
    ae.plot()
    ae_2 = eval(repr(ae))
    assert ae_2.parameter == {'limits': (100, 110)}
