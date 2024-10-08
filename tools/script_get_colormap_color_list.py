"""
Utility script to print colormap colors as list.

Such color lists are copied into locan.visualize.colormap_definitions for
default colormaps.
"""

import numpy as np
from colorcet import m_coolwarm, m_fire, m_glasbey, m_glasbey_dark, m_gray  # noqa: F401

from locan.visualize.colormap import ColormapType


def get_colors(colormap: ColormapType) -> None:
    """
    Extract RGB color values from colormap and print as list of lists.

    Parameters
    ----------
    colormap
        The colormap to be evaluated.

    Returns
    -------
    None
    """
    print(colormap.name)
    curated_list = colormap(np.linspace(0, 1, 256))
    print("[")
    for item in curated_list:
        print("[" + ", ".join(map(str, item)) + "],")
    print("]")


if __name__ == "__main__":
    get_colors(colormap=m_glasbey_dark)
