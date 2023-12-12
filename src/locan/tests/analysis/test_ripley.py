import matplotlib.pyplot as plt  # needed for visualization  # noqa: F401
import numpy as np
import pytest

from locan import LocData
from locan.analysis import RipleysHFunction, RipleysKFunction, RipleysLFunction


class TestRipleysKFunction:
    def test_init(self, locdata_2d):
        rkf = RipleysKFunction(meta={"comment": "this is an example"})
        assert str(rkf).startswith("RipleysKFunction(radii=[")
        assert rkf.results is None
        assert rkf.meta.comment == "this is an example"
        with pytest.raises(AttributeError):
            assert rkf.Ripley_h_maximum is None
        rkf.plot()


class TestRipleysLFunction:
    def test_init(self, locdata_2d):
        rlf = RipleysLFunction(meta={"comment": "this is an example"})
        assert str(rlf).startswith("RipleysLFunction(radii=[")
        assert rlf.results is None
        assert rlf.meta.comment == "this is an example"
        with pytest.raises(AttributeError):
            assert rlf.Ripley_h_maximum is None
        rlf.plot()


class TestRipleysHFunction:
    def test_init(self, locdata_2d):
        rhf = RipleysHFunction(meta={"comment": "this is an example"})
        assert str(rhf).startswith("RipleysHFunction(radii=[")
        assert rhf.results is None
        assert rhf.meta.comment == "this is an example"
        assert rhf.Ripley_h_maximum is None
        rhf.plot()


def test_Ripleys_k_function_empty(caplog):
    rhf = RipleysKFunction().compute(LocData())
    rhf.plot()
    assert caplog.record_tuples == [("locan.analysis.ripley", 30, "Locdata is empty.")]


def test_Ripleys_k_function(locdata_2d):
    radii = np.linspace(0, 20, 11)
    rhf = RipleysKFunction(radii=radii).compute(locdata_2d)
    # print(rhf.results)
    assert all(rhf.results.index == radii)
    assert len(rhf.results) == len(radii)

    rhf.plot()
    # plt.show()

    other_locdata = LocData.from_selection(locdata_2d, indices=[0, 1, 2])
    rhf = RipleysKFunction(radii=radii, region_measure=1).compute(
        locdata_2d, other_locdata=other_locdata
    )
    assert all(rhf.results.index == radii)
    assert len(rhf.results) == len(radii)

    plt.close("all")


def test_Ripleys_k_function_3d(locdata_3d):
    radii = np.linspace(0, 20, 20)
    rhf = RipleysKFunction(radii=radii).compute(locdata_3d)
    assert all(rhf.results.index == radii)
    assert len(rhf.results) == len(radii)

    rhf.plot()
    # plt.show()

    other_locdata = LocData.from_selection(locdata_3d, indices=[0, 1, 2])
    rhf = RipleysKFunction(radii=radii, region_measure=1).compute(
        locdata_3d, other_locdata=other_locdata
    )
    assert all(rhf.results.index == radii)
    assert len(rhf.results) == len(radii)

    plt.close("all")


def test_Ripleys_l_function(locdata_2d):
    radii = np.linspace(0, 20, 20)
    rhf = RipleysLFunction(radii=radii).compute(locdata_2d)
    assert all(rhf.results.index == radii)
    assert len(rhf.results) == len(radii)

    rhf.plot()
    # plt.show()

    other_locdata = LocData.from_selection(locdata_2d, indices=[0, 1, 2])
    rhf = RipleysLFunction(radii=radii, region_measure=1).compute(
        locdata_2d, other_locdata=other_locdata
    )
    assert all(rhf.results.index == radii)
    assert len(rhf.results) == len(radii)

    plt.close("all")


def test_Ripleys_h_function(locdata_2d):
    radii = np.linspace(0, 20, 21)
    rhf = RipleysHFunction(radii=radii).compute(locdata_2d)
    assert all(rhf.results.index == radii)
    assert len(rhf.results) == len(radii)

    rhf.plot()
    # plt.show()

    other_locdata = LocData.from_selection(locdata_2d, indices=[0, 1, 2])
    rhf = RipleysHFunction(radii=radii, region_measure=1).compute(
        locdata_2d, other_locdata=other_locdata
    )
    assert all(rhf.results.index == radii)
    assert len(rhf.results) == len(radii)

    assert len(rhf.Ripley_h_maximum) == 1
    assert rhf.Ripley_h_maximum.iloc[0].radius == 0
    assert rhf.Ripley_h_maximum.iloc[0].Ripley_h_maximum == 0
    del rhf.Ripley_h_maximum
    assert rhf.Ripley_h_maximum.iloc[0].radius == 0
    assert rhf.Ripley_h_maximum.iloc[0].Ripley_h_maximum == 0

    plt.close("all")
