import pytest
import numpy as np
import matplotlib.pyplot as plt  # needed for visualization

from locan import LocData
from locan.analysis import RipleysKFunction, RipleysLFunction, RipleysHFunction


def test_Ripleys_k_function(locdata_2d):
    radii = np.linspace(0, 20, 11)
    rhf = RipleysKFunction(radii=radii).compute(locdata_2d)
    # print(rhf.results)
    assert all(rhf.results.index == radii)
    assert (len(rhf.results) == len(radii))

    rhf.plot()
    # plt.show()

    other_locdata = LocData.from_selection(locdata_2d, indices=[0, 1, 2])
    rhf = RipleysKFunction(radii=radii, region_measure=1).compute(locdata_2d, other_locdata=other_locdata)
    assert all(rhf.results.index == radii)
    assert (len(rhf.results) == len(radii))


def test_Ripleys_k_function_3d(locdata_3d):
    radii = np.linspace(0, 20, 20)
    rhf = RipleysKFunction(radii=radii).compute(locdata_3d)
    assert all(rhf.results.index == radii)
    assert (len(rhf.results) == len(radii))

    rhf.plot()
    # plt.show()

    other_locdata = LocData.from_selection(locdata_3d, indices=[0, 1, 2])
    rhf = RipleysKFunction(radii=radii, region_measure=1).compute(locdata_3d, other_locdata=other_locdata)
    assert all(rhf.results.index == radii)
    assert (len(rhf.results) == len(radii))


def test_Ripleys_l_function(locdata_2d):
    radii = np.linspace(0, 20, 20)
    rhf = RipleysLFunction(radii=radii).compute(locdata_2d)
    assert all(rhf.results.index == radii)
    assert (len(rhf.results) == len(radii))

    rhf.plot()
    # plt.show()

    other_locdata = LocData.from_selection(locdata_2d, indices=[0, 1, 2])
    rhf = RipleysLFunction(radii=radii, region_measure=1).compute(locdata_2d, other_locdata=other_locdata)
    assert all(rhf.results.index == radii)
    assert (len(rhf.results) == len(radii))


def test_Ripleys_h_function(locdata_2d):
    radii = np.linspace(0, 20, 21)
    rhf = RipleysHFunction(radii=radii).compute(locdata_2d)
    assert all(rhf.results.index == radii)
    assert (len(rhf.results) == len(radii))

    rhf.plot()
    # plt.show()

    other_locdata = LocData.from_selection(locdata_2d, indices=[0, 1, 2])
    rhf = RipleysHFunction(radii=radii, region_measure=1).compute(locdata_2d, other_locdata=other_locdata)
    assert all(rhf.results.index == radii)
    assert (len(rhf.results) == len(radii))

    assert len(rhf.Ripley_h_maximum) == 1
    assert rhf.Ripley_h_maximum.iloc[0].radius == 0
    assert rhf.Ripley_h_maximum.iloc[0].Ripley_h_maximum == 0
    del rhf.Ripley_h_maximum
    assert rhf.Ripley_h_maximum.iloc[0].radius == 0
    assert rhf.Ripley_h_maximum.iloc[0].Ripley_h_maximum == 0
