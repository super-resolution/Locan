"""

Compute Ripley's k function.

Spatial clustering of localization data is characterized by Ripley's k or related l and h functions [1]_.


Ripley's k function is computed for 2D and 3D data for a series of radii as described in [2]_ in order to provide
evidence for deviations from a spatially homogeneous Poisson process (i.e. complete spatial randomness, CSR).
Ripley' s k function is estimated by summing all points or over test points being a random subset of all points:

.. math::

   k(r) = \\frac{1}{\\lambda (n-1)} \\sum_{i=1}^{n} N_{p_{i}}(r)


here :math:`p_i` is the :math:`i^{th}` point of n test points, :math:`N_{p_{i}}` is the number of points within the region
of radius r around :math:`p_{i}`, and :math:`\\lambda` is the density of all points.

We follow the definition of l and h functions in [2]_. Ripley's l function is:

.. math::

   l(r) &= \\sqrt{k(r)) / \\pi} \\qquad \\text{in 2D}

   l(r) &= \\sqrt[3]{\\frac{3}{4 \\pi} k(r)} \\qquad \\text{in 3D}


And Ripley's h function is:

.. math::

   h(r) = l(r) - r


References
----------
.. [1] B.D. Ripley, Modelling spatial patterns. Journal of the Royal Statistical Society, 1977, 172–212.
.. [2] Kiskowski, M. A., Hancock, J. F., and Kenworthy, A. K., On the use of Ripley's K-function and its
   derivatives to analyze domain size. Biophysical journal, 2009, 97, 1095–1103.

"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from locan.analysis.analysis_base import _Analysis

__all__ = ["RipleysKFunction", "RipleysLFunction", "RipleysHFunction"]

logger = logging.getLogger(__name__)

# The algorithms


def _ripleys_k_function(points, radii, region_measure=1, other_points=None):
    """
    Compute Ripley's K function for two- or three-dimensional data at the given radii.

    Parameters
    ----------
    points : array
        2D or 3D points on which to estimate Ripley's K function.
    radii: array of float
        Radii at which to compute Ripley's k function.
    region_measure : float
        region measure (area or volume) for points.
    other_points : array or None
        2D or 3D points from which to estimate Ripley's K function (e.g. subset of points). For None other_points
        is set to points (default).

    Returns
    -------
    ripley : numpy.ndarray of floats with shape (n_radii,)
        Ripley's K function evaluated at input radii.
    """
    if other_points is None:
        other_points = points

    nn = NearestNeighbors(metric="euclidean").fit(points)

    ripley_0 = []
    for radius in radii:
        indices_list = nn.radius_neighbors(other_points, radius, return_distance=False)
        ripley_0.append(np.array([len(indices) - 1 for indices in indices_list]).sum())

    ripley = (
        region_measure / (len(points) * (len(other_points) - 1)) * np.array(ripley_0)
    )

    return ripley


def _ripleys_l_function(points, radii, region_measure=1, other_points=None):
    """
    Evaluates Ripley's L function which is different for 2D and 3D data points. For parameter description
    see _ripleys_k_function.
    """
    if np.shape(points)[1] == 2:
        return np.sqrt(
            _ripleys_k_function(points, radii, region_measure, other_points) / np.pi
        )
    elif np.shape(points)[1] == 3:
        return np.cbrt(
            _ripleys_k_function(points, radii, region_measure, other_points)
            * 3
            / 4
            / np.pi
        )


def _ripleys_h_function(points, radii, region_measure=1, other_points=None):
    """
    Evaluates Ripley's H function. For parameter description
    see _ripleys_k_function.
    """

    return _ripleys_l_function(points, radii, region_measure, other_points) - radii


# The specific analysis classes


class RipleysKFunction(_Analysis):
    """
    Compute Ripley's K function for two- or three-dimensional data at the given radii.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    radii: array of float
        Radii at which to compute Ripley's k function.
    region_measure : float or 'bb'
        Region measure (area or volume) for point region. For 'bb' the region measure of the bounding_box is used.


    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        Data frame with radii as provided and Ripley's K function.
    """

    count = 0

    def __init__(self, meta=None, radii=None, region_measure="bb"):
        radii = np.linspace(0, 100, 10) if radii is None else radii
        super().__init__(meta=meta, radii=radii, region_measure=region_measure)
        self.results = None

    def compute(self, locdata, other_locdata=None):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data with 2D or 3D coordinates on which to estimate Ripley's K function.
        other_locdata : LocData
            Other localization data from which to estimate Ripley's K function (e.g. subset of points).
            For None other_points is set to points (default).

        Returns
        -------
        Analysis class
           Returns the Analysis class object (self).
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        points = locdata.coordinates
        if other_locdata is not None:
            other_points = other_locdata.coordinates
        else:
            other_points = None

        # choose the right region_measure
        # todo: add other hull regions
        if self.parameter["region_measure"] == "bb":
            region_measure = float(locdata.properties["region_measure_bb"])
        else:
            region_measure = self.parameter["region_measure"]

        ripley = _ripleys_k_function(
            points=points,
            radii=self.parameter["radii"],
            region_measure=region_measure,
            other_points=other_points,
        )
        self.results = pd.DataFrame(
            {"radius": self.parameter["radii"], "Ripley_k_data": ripley}
        )
        self.results = self.results.set_index("radius")
        return self

    def plot(self, ax=None, **kwargs):
        plot(self, ax, **kwargs)


class RipleysLFunction(_Analysis):
    """
    Compute Ripley's L function for two- or three-dimensional data at the given radii.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    radii: array of float
        Radii at which to compute Ripley's k function.
    region_measure : float or 'bb'
        Region measure (area or volume) for point region. For 'bb' the region measure of the bounding_box is used.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        Data frame with radii as provided and Ripley's L function.
    """

    count = 0

    def __init__(self, meta=None, radii=None, region_measure="bb"):
        radii = np.linspace(0, 100, 10) if radii is None else radii
        super().__init__(meta=meta, radii=radii, region_measure=region_measure)
        self.results = None

    def compute(self, locdata, other_locdata=None):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data with 2D or 3D coordinates on which to estimate Ripley's L function.
        other_locdata : LocData
            Other localization data from which to estimate Ripley's L function (e.g. subset of points).
            For None other_points is set to points (default).

        Returns
        -------
        Analysis class
           Returns the Analysis class object (self).
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        points = locdata.coordinates
        if other_locdata is not None:
            other_points = other_locdata.coordinates
        else:
            other_points = None

        # choose the right region_measure
        # todo: add other hull regions
        if self.parameter["region_measure"] == "bb":
            region_measure = float(locdata.properties["region_measure_bb"])
        else:
            region_measure = self.parameter["region_measure"]

        ripley = _ripleys_l_function(
            points=points,
            radii=self.parameter["radii"],
            region_measure=region_measure,
            other_points=other_points,
        )
        self.results = pd.DataFrame(
            {"radius": self.parameter["radii"], "Ripley_l_data": ripley}
        )
        self.results = self.results.set_index("radius")
        return self

    def plot(self, ax=None, **kwargs):
        plot(self, ax, **kwargs)


class RipleysHFunction(_Analysis):
    """
    Compute Ripley's H function for two- or three-dimensional data at the given radii.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    radii: array of float
        Radii at which to compute Ripley's k function.
    region_measure : float or 'bb'
        Region measure (area or volume) for point region. For 'bb' the region measure of the bounding_box is used.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        Data frame with radii as provided and Ripley's H function.
    Ripley_h_maximum : pandas.DataFrame
        Data frame with radius and Ripley's H value for the radius at which the H function has its maximum.
    """

    count = 0

    def __init__(self, meta=None, radii=None, region_measure="bb"):
        radii = np.linspace(0, 100, 10) if radii is None else radii
        super().__init__(meta=meta, radii=radii, region_measure=region_measure)
        self.results = None
        self._Ripley_h_maximum = None

    def compute(self, locdata, other_locdata=None):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data with 2D or 3D coordinates on which to estimate Ripley's H function.
        other_locdata : LocData
            Other localization data from which to estimate Ripley's H function (e.g. subset of points).
            For None other_points is set to points (default).

        Returns
        -------
        Analysis class
           Returns the Analysis class object (self).
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        # reset secondary results
        self._Ripley_h_maximum = None

        points = locdata.coordinates
        if other_locdata is not None:
            other_points = other_locdata.coordinates
        else:
            other_points = None

        # choose the right region_measure
        # todo: add other hull regions
        if self.parameter["region_measure"] == "bb":
            region_measure = float(locdata.properties["region_measure_bb"])
        else:
            region_measure = self.parameter["region_measure"]

        ripley = _ripleys_h_function(
            points=points,
            radii=self.parameter["radii"],
            region_measure=region_measure,
            other_points=other_points,
        )
        self.results = pd.DataFrame(
            {"radius": self.parameter["radii"], "Ripley_h_data": ripley}
        )
        self.results = self.results.set_index("radius")
        return self

    @property
    def Ripley_h_maximum(self):
        if self._Ripley_h_maximum is None:
            index = self.results["Ripley_h_data"].idxmax()
            self._Ripley_h_maximum = pd.DataFrame(
                {"radius": index, "Ripley_h_maximum": self.results.loc[index]}
            )
            return self._Ripley_h_maximum
        else:
            return self._Ripley_h_maximum

    @Ripley_h_maximum.deleter
    def Ripley_h_maximum(self):
        self._Ripley_h_maximum = None

    def plot(self, ax=None, **kwargs):
        plot(self, ax, **kwargs)


# Interface functions


def plot(self, ax=None, **kwargs):
    """
    Provide plot of results as :class:`matplotlib.axes.Axes` object.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        The axes on which to show the image
    kwargs : dict
        Other parameters passed to :func:`matplotlib.pyplot.plot`.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes object with the plot.
    """
    if ax is None:
        ax = plt.gca()

    if not self:
        return ax

    self.results.plot(ax=ax, **kwargs)

    if self.results.columns[0] == "Ripley_k_data":
        ylabel = "K-function"
    elif self.results.columns[0] == "Ripley_l_data":
        ylabel = "L-function"
    elif self.results.columns[0] == "Ripley_h_data":
        ylabel = "H-function"
    else:
        ylabel = None

    ax.set(title="Ripley's " + ylabel, xlabel="radius", ylabel=ylabel)

    return ax
