import pytest
import numpy as np
import pandas as pd

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis.localization_precision import Localization_precision, _DistributionFits


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt',
                                   nrows=1000)

@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': range(10),
        'Position_y': range(10),
        'Frame': [1,1,1,2,2,3,5,8,9,10]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


def test_Localization_precision(locdata_simple):
    #print(locdata_simple.data)
    lp = Localization_precision(locdata=locdata_simple).compute()
    #print(lp.results)
    for prop in ['Position_delta_x', 'Position_delta_y', 'Position_distance']:
        assert(prop in lp.results.columns)

def test_Localization_precision_plot(locdata):
    lp = Localization_precision(locdata=locdata).compute()
    lp.plot(window=10, show=False)
    #print(lp.results)

def test_Distribution_fits(locdata_simple):
    #print(locdata_simple.data)
    lp = Localization_precision(locdata=locdata_simple).compute()
    distribution_statistics = _DistributionFits(lp)
    #print(distribution_statistics)
    distribution_statistics.fit()
    distribution_statistics.plot_distribution_fit(show=False)
    # print(dist_fits.Position_delta_x_center)
    # print(dist_fits.Position_distance_sigma)
    assert(distribution_statistics.Position_delta_x_center is None)
    assert(distribution_statistics.Position_distance_sigma is not None)
    lp.fit_distributions(loc_property='Position_delta_x')
    assert(lp.distribution_statistics.Position_delta_x_center==-1.6666666666666667)
    #print(lp.distribution_statistics)

    lp.fit_distributions(loc_property=None)
    assert(lp.distribution_statistics.Position_delta_x_center==-1.6666666666666667)
    #print(lp.distribution_statistics)

def test_Localization_precision_histogram(locdata_simple):
    #print(locdata_simple.data)
    lp = Localization_precision(locdata=locdata_simple).compute()
    #lp.hist(fit=True)

    with pytest.raises(AttributeError):
        lp.hist(loc_property='Position_delta_x',fit=False, show=False)
        assert (lp.distribution_statistics.Position_delta_x_center == -1.6666666666666667)

    lp.hist(loc_property='Position_delta_x',fit=True, show=False)
    #print(f'center: {lp.distribution_statistics.Position_delta_x_center}')
    #print(f'sigma: {lp.distribution_statistics.Position_delta_x_sigma}')
    assert(lp.distribution_statistics.Position_delta_x_center==-1.6666666666666667)
    assert (lp.distribution_statistics.Position_delta_x_sigma == 0.74535599249993)
    print(lp.distribution_statistics.parameter_dict())
