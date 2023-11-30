.. _colormaps:

==========================
Colormaps
==========================

For visual inspection and presentation SMLM data is typically rendered in 2D or 3D images.
Various binning algorithms are used to create a representation of localisation densities.
Intensity values are represented by color according to a selected colormap.
Various colormaps can be chosen that accentuate certain data structures.


For SMLM images we aim at using colormaps that SMLM users are used to but that perform well with respect to human
perception.
Therefore, we recommend using colormaps that are optimized for accurate perception as for instance provided by the `colorcet`_
library.

We recommend default use for various types of colormaps:
All 2D one-channel plot functions use the `fire`_ colormap as default if `colorcet` is installed.
Otherwise we use the matplotlib colormap `viridis`_ as default.

For categorical data we use `glasbey_dark`_ from `colorcet` as default colormap or alternatively `tab20` from
`matplotlib`.
For diverging data we use `coolwarm`_ from `colorcet` as default colormap or alternatively `coolwarm` from
`matplotlib`.
For rainbow/jet-like data we use `turbo`_ as default colormap.

For details on how to use colormaps see API documentation for :mod:`locan.visualize.colormap`.

.. _colorcet: https://colorcet.pyviz.org
.. _fire: https://colorcet.pyviz.org/user_guide/Continuous.html
.. _viridis: https://matplotlib.org/tutorials/colors/colormaps.html
.. _glasbey_dark: https://colorcet.pyviz.org/user_guide/Categorical.html
.. _coolwarm: https://colorcet.pyviz.org/user_guide/Continuous.html
.. _turbo: https://blog.research.google/2019/08/turbo-improved-rainbow-colormap-for.html
