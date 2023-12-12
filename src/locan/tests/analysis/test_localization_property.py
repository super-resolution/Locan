import pandas as pd
from scipy import stats

from locan import LocData
from locan.analysis import LocalizationProperty
from locan.analysis.localization_property import _DistributionFits


def test_test_LocalizationProperty_empty(caplog):
    lprop = LocalizationProperty().compute(LocData())
    lprop.fit_distributions()
    lprop.plot()
    lprop.hist()
    assert caplog.record_tuples == [
        ("locan.analysis.localization_property", 30, "Locdata is empty."),
        ("locan.analysis.localization_property", 30, "No results available to fit."),
    ]


def test_DistributionFits(locdata_rapidSTORM_2d):
    lprop = LocalizationProperty().compute(locdata=locdata_rapidSTORM_2d)
    assert repr(lprop) == "LocalizationProperty(loc_property=intensity, index=None)"
    ds = _DistributionFits(lprop)
    ds.fit(distribution=stats.expon)
    # print(ds.parameters)
    # print(ds.__dict__)
    # #print(ds.parameter_dict())
    # ds.plot()
    assert ds.parameters == ["intensity_loc", "intensity_scale"]
    ds.fit(distribution=stats.expon, with_constraints=False, floc=1000)
    assert ds.parameter_dict()["intensity_loc"] == 1000


def test_Localization_property(locdata_rapidSTORM_2d):
    lprop = LocalizationProperty(loc_property="intensity")
    assert not lprop
    lprop.plot()
    lprop.hist()
    lprop.compute(locdata=locdata_rapidSTORM_2d)
    assert lprop
    # print(lprop)
    assert repr(lprop) == "LocalizationProperty(loc_property=intensity, index=None)"
    assert isinstance(lprop.results, pd.DataFrame)
    assert lprop.results.columns == pd.Index(["intensity"], dtype="object")
    assert lprop.distribution_statistics is None
    lprop.plot()
    lprop.hist()
    lprop.fit_distributions()
    assert list(lprop.distribution_statistics.parameter_dict().keys()) == [
        "intensity_loc",
        "intensity_scale",
    ]

    lprop = LocalizationProperty(loc_property="intensity", index="frame").compute(
        locdata=locdata_rapidSTORM_2d
    )
    # print(lprop.results)
    assert lprop.results.index.name == "frame"
    assert lprop.results.columns == pd.Index(["intensity"], dtype="object")
