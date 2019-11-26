import pytest
import matplotlib.pyplot as plt

from surepy.render import render_2d


def test_simple_rendering_2D(locdata_blobs_2d):
    # render_2d(locdata_blobs_2d)
    # render_2d(locdata_blobs_2d, bin_size=100, range=[[500, 1000], [500, 1000]], cbar=False)

    render_2d(locdata_blobs_2d, bin_size=100, range='auto', rescale=(0, 100))
    # render_2d(locdata_blobs_2d, bin_size=100, range='zero', rescale=(0, 100))
    # render_2d(locdata_blobs_2d, bin_size=100, range='auto', rescale='equal')
    # render_2d(locdata_blobs_2d, bin_size=100, range='auto', rescale=None)
    #
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # render_2d(locdata_blobs_2d, ax=ax[0])
    # render_2d(locdata_blobs_2d, range='zero', ax=ax[1])
    #
    # render_2d(locdata_blobs_2d, ax=ax[0], colorbar_kws=dict(ax=ax[0]))
    # render_2d(locdata_blobs_2d, range='zero', ax=ax[1])

    # plt.show()
