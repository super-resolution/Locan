import pytest
import numpy as np

from surepy.data.regions.region import RoiRegion
from surepy.data.regions.utils import surrounding_region


def test_surrounding_region():
    rr = RoiRegion(region_type='rectangle', region_specs=((0, 0), 1, 1, 0))
    sr = surrounding_region(region=rr, distance=1, support=None)
    assert sr.area == 7.136548490545939

    rr = RoiRegion(region_type='ellipse', region_specs=((0, 0), 1, 3, 0))
    sr = surrounding_region(region=rr, distance=1, support=None)
    assert sr.area == 9.982235792799617

    rr = RoiRegion(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)))
    sr = surrounding_region(region=rr, distance=1, support=None)
    assert sr.area == 6.754579952999869

    rr = RoiRegion(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)))
    sup = RoiRegion(region_type='rectangle', region_specs=((0, 0), 2, 2, 0))
    sr = surrounding_region(region=rr, distance=1, support=sup)
    assert sr.area == 3.012009021216654
