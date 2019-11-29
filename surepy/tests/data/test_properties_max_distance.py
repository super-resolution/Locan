from surepy.data.properties import max_distance


def test_max_distance_2d(locdata_2d):
    mdist = max_distance(locdata=locdata_2d)
    assert (mdist == {'max_distance': 5.656854249492381})

def test_max_distance_3d(locdata_3d):
    mdist = max_distance(locdata=locdata_3d)
    assert (mdist == {'max_distance': 6.164414002968976})
