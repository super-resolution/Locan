import numpy as np

from locan.render.utilities import _napari_shape_to_region


def test__napari_shape_to_region():
    # rectangle
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    bin_edges = np.array([[0, 10, 20], [2, 3, 4, 5]], dtype=object)
    region = _napari_shape_to_region(vertices, bin_edges, "rectangle")
    assert repr(region) == "Rectangle((0.0, 2.0), 31.0, 2.5, 0)"

    # ellipse
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    bin_edges = np.array([[0, 10, 20], [2, 3, 4, 5]], dtype=object)
    region = _napari_shape_to_region(vertices, bin_edges, "ellipse")
    assert repr(region) == "Ellipse((15.5, 3.25), 31.0, 2.5, 0)"

    # polygon
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    bin_edges = np.array([[0, 10, 20], [2, 3, 4, 5]], dtype=object)
    region = _napari_shape_to_region(vertices, bin_edges, "polygon")
    assert (
        repr(region)
        == "Polygon([[0.0, 2.0], [0.0, 4.5], [31.0, 4.5], [31.0, 2.0], [0.0, 2.0]])"
    )
