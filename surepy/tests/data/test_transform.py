import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.io.io_locdata import load_rapidSTORM_file
from surepy.data.transform import randomize
from surepy.data.transform.bunwarp import _read_matrix, bunwarp

from surepy.render import render2D


@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))

def test_randomize(locdata_simple):
    locdata_randomized = randomize(locdata_simple, hull_region='bb')
    #locdata_randomized.print_meta()
    assert (len(locdata_randomized) == len(locdata_simple))


# todo: create useful small test
def test_read_matrix(locdata_simple):
    dstorm_file = r'C:\Users\Soeren\MyData\Programming\Python\Analysis projects\181023_Bunwarp registration_Patrick\\' \
           r'Twocolor\K35_647_4.txt'
    path = r'C:\Users\Soeren\MyData\Programming\Python\Analysis projects\181023_Bunwarp registration_Patrick\\' \
           r'Twocolor\matrix.txt'
#    print(len(_read_matrix(path)))

    dat = load_rapidSTORM_file(dstorm_file)
    new_loc = bunwarp(dat, path)

    print(new_loc.data.head())
    #render2D(dat)
