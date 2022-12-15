import matplotlib.pyplot as plt  # this import is needed for interactive tests

from locan import scatter_3d_mpl


def test_scatter_3d_mpl(locdata_3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter_3d_mpl(locdata_3d, ax=ax, text_kwargs=dict(color="r"), color="r")
    # plt.show()

    plt.close("all")


def test_scatter_3d_mpl_empty(locdata_empty):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter_3d_mpl(locdata_empty, ax=ax)
    # plt.show()

    plt.close("all")


def test_scatter_3d_mpl_single(locdata_single_localization, caplog):
    fig = plt.figure()
    fig.add_subplot(projection="3d")
    scatter_3d_mpl(locdata_single_localization)
    log_msg = (
        "locan.visualize.render_mpl.render3d",
        30,
        "Locdata carries a single localization.",
    )
    assert log_msg in caplog.record_tuples
    # plt.show()

    plt.close("all")
