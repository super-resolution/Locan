import pytest
import pandas as pd

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis.localization_precision import LocalizationPrecision, _DistributionFits


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
                                   nrows=1000)


@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        'position_x': range(10),
        'position_y': range(10),
        'frame': [1, 1, 1, 2, 2, 3, 5, 8, 9, 10]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


def test_Localization_precision(locdata_simple):
    # print(locdata_simple.data)
    lp = LocalizationPrecision().compute(locdata=locdata_simple)
    # print(lp.results)
    for prop in ['position_delta_x', 'position_delta_y', 'position_distance']:
        assert(prop in lp.results.columns)


def test_Localization_precision_plot(locdata):
    lp = LocalizationPrecision().compute(locdata=locdata)
    lp.plot(window=10, show=False)
    # print(lp.results)
    # print(lp.meta)


def test_Distribution_fits(locdata_simple):
    # print(locdata_simple.data)
    lp = LocalizationPrecision().compute(locdata=locdata_simple)
    distribution_statistics = _DistributionFits(lp)
    distribution_statistics.fit()
    assert('position_distance_sigma' in distribution_statistics.parameters)
    distribution_statistics.plot(show=False)

    lp.fit_distributions(loc_property=None)
    assert lp.distribution_statistics.position_delta_x_loc
    # print(lp.distribution_statistics)

    lp.fit_distributions(loc_property='position_delta_x', floc=0)
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_delta_x_loc == 0

# todo add tests for 1d and 3d


def test_Localization_precision_histogram(locdata_simple):
    lp = LocalizationPrecision().compute(locdata=locdata_simple)
    # lp.hist(fit=True)

    with pytest.raises(AttributeError):
        lp.hist(loc_property='position_delta_x', fit=False, show=False)
        assert lp.distribution_statistics.position_delta_x_loc

    lp.hist(loc_property='position_delta_x', fit=True, show=False)
    # print(lp.distribution_statistics.parameter_dict())
    assert lp.distribution_statistics.position_delta_x_loc
    assert lp.distribution_statistics.position_delta_x_scale


# standard LocData fixtures

@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', 0),
    ('locdata_single_localization', 1),
    ('locdata_fix', 6),
    ('locdata_non_standard_index', 6)
])
def test_standard_locdata_objects(
        locdata_empty, locdata_single_localization, locdata_fix, locdata_non_standard_index,
        fixture_name, expected):
    dat = eval(fixture_name)
    with pytest.warns(UserWarning):
        LocalizationPrecision(radius=1).compute(locdata=dat)
