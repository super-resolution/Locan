from pathlib import Path
import tempfile
import pickle

from locan.analysis.analysis_example import AnalysisExampleAlgorithm_1, AnalysisExampleAlgorithm_2
from locan.analysis import metadata_analysis_pb2


def test_Analysis_example():
    ae = AnalysisExampleAlgorithm_1(limits=(100, 110), meta={'comment': 'this is an example'})
    # print(ae)
    assert repr(ae) == 'AnalysisExampleAlgorithm_1(limits=(100, 110))'
    assert str(ae) == repr(ae)
    assert ae.parameter == {'limits': (100, 110)}
    assert ae.meta.comment == 'this is an example'
    ae.compute(locdata=None)
    assert (len(ae.results) == 10)

    ae.plot()
    ae.plot_2()
    ae.plot_histogram_fit()
    assert ae.attribute_center
    assert ae.attribute_sigma
    ae.report()

    ae_2 = eval(repr(ae))
    assert ae_2.parameter == {'limits': (100, 110)}


def test_Analysis_example_2():
    ae = AnalysisExampleAlgorithm_2()
    assert repr(ae) == 'AnalysisExampleAlgorithm_2(n_sample=100, seed=None)'
    ae.compute()
    assert len(ae.results) == 100
    ae.plot()
    ae.plot_2()
    ae.plot_histogram_fit()
    assert ae.attribute_center
    assert ae.attribute_sigma
    ae.report()


def test_pickling_Analysis():
    ae = AnalysisExampleAlgorithm_1(limits=(100, 110), meta={'comment': 'this is an example'})
    ae.compute(locdata=None)
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / 'pickled_analysis.pickle'
        with open(file_path, 'wb') as file:
            pickle.dump(ae, file, pickle.HIGHEST_PROTOCOL)
        with open(file_path, 'rb') as file:
            ae_pickled = pickle.load(file)
        assert ae_pickled.parameter == ae.parameter
        assert isinstance(ae_pickled.meta, metadata_analysis_pb2.AMetadata)
        assert ae_pickled.meta == ae.meta
