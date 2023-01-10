"""
This LocData test is placed in a dedicated file to get rid of the following
problem:

Under GitHub actions with windows and python 3.10 the following error occurred
sporadically:
>       assert count_0 + 2 == count_2
E       assert (19 + 2) == 15
locan\\tests\\data\\test_locdata.py:60: AssertionError
"""
import copy

from locan import LocData


def test_LocData_count():
    count_0 = copy.deepcopy(LocData.count)
    dat = LocData()
    count_1 = copy.deepcopy(LocData.count)
    dat_2 = LocData()
    count_2 = copy.deepcopy(LocData.count)
    assert dat.properties == dat_2.properties
    del dat
    count_3 = copy.deepcopy(LocData.count)
    assert count_0 + 1 == count_1
    assert count_0 + 2 == count_2
    assert count_2 - 1 == count_3
