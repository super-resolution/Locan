import pytest
import numpy as np
import pandas as pd
from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
#import surepy.tests.test_data
from surepy.analysis import NearestNeighborDistances
from surepy.analysis.nearest_neighbor import NNDistances_csr_2d, _DistributionFits

# fixtures

@pytest.fixture()
def locdata_simple():
    dict = {
        'position_x': [0, 0, 1, 4, 5],
        'position_y': [0, 1, 3, 4, 1],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))

@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt',
                                   nrows=100)


@pytest.fixture()
def other_locdata_simple():
    dict = {
        'position_x': [10, 11],
        'position_y': [10, 11],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


# tests

def test_NNDistances_csr_2d():
    dist = NNDistances_csr_2d()
    print(dist.shapes)

# todo fit is not working
# def test_DistributionFits(locdata):
#     nn_1 = NearestNeighborDistances(locdata).compute()
#     ds = _DistributionFits(nn_1)
#     print(ds.parameter_dict())
#     ds.fit()
#     print(ds.parameter_dict())
#     #ds.plot()
#     assert(ds.parameters == ['loc', 'scale'])

def test_Nearest_neighbor_distances(locdata_simple, other_locdata_simple):
    nn_1 = NearestNeighborDistances().compute(locdata_simple)
    #nn_1.hist()
    assert(nn_1.localization_density==0.25)
    #print(nn_1.results)
    assert(nn_1.results['nn_index'].iloc[0] == 1)

    nn_2 = NearestNeighborDistances().compute(locdata_simple, other_locdata=other_locdata_simple)
    # print(nn_2.results)
    assert(nn_2.results['nn_distance'].iloc[0] == 14.142135623730951)
