"""

Region as support for localization data.

This module provides classes to define regions for localization data.

The roi region is provided as a RoiRegion object that provides methods for getting a
printable representation (that can be saved in a yaml file), for returning a matplotlib patch that can be
shown in a graph, and for finding points within the region.
"""

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mPath
import matplotlib.patches as mPatches
from scipy.spatial.distance import pdist
from shapely.geometry import Polygon as shPolygon


class RoiRegion:
    """
    Region object to specify regions of interest.

    A region that defines a region of interest with methods for getting a printable representation (that can also be
    saved in a yaml file), for returning a matplotlib patch that can be shown in a graph, for finding points within
    the region.

    Parameters
    ----------
    region_type : str
        A string indicating the roi shape.
        In 1D it can be `interval`.
        In 2D it can be either `rectangle`, `ellipse`, or closed `polygon`.
        In 3D it can be either `cuboid` or `ellipsoid` or `polyhedron` (not implemented yet).
    region_specs : tuple
        1D rois are defined by the following tuple:
        * interval: (start, stop)
        2D rois are defined by the following tuples:
        * rectangle: ((corner_x, corner_y), width, height, angle)
        * ellipse: ((center_x, center_y), width, height, angle)
        * polygon: ((point1_x, point1_y), (point2_x, point2_y), ..., (point1_x, point1_y))
        3D rois are defined by the following tuples:
        * cuboid: ((corner_x, corner_y, corner_z), length, width, height, angle_1, angle_2, angle_3)
        * ellipsoid: ((center_x, center_y, center_z), length, width, height, angle_1, angle_2, angle_3)
        * polyhedron: (...)


    Attributes
    ----------
    region_type : str
        Type of region
    region_specs : tuple
        Specifications for region
    _region : RoiRegion object
        RoiRegion instance for the specified region type.
    polygon : ndarray of tuples
        Array of points for a closed polygon approximating the region of interest in clockwise orientation. The first
        and last point must be identical.
    dimension : int
        Spatial dimension of region
    centroid : tuple of float
        Centroid coordinates
    max_distance : array-like of float
        Maximum distance between any two points in the region
    region_measure : float
        Hull measure, i.e. area or volume
    subregion_measure : float
        Measure of the sub-dimensional region, i.e. circumference or surface.
    """

    def __init__(self, region_type, region_specs):
        self.region_specs = region_specs
        self.region_type = region_type

        if region_type == 'interval':
            self._region = _RoiInterval(region_specs)

        elif region_type == 'rectangle':
            self._region = _RoiRectangle(region_specs)

        elif region_type == 'ellipse':
            self._region = _RoiEllipse(region_specs)

        elif region_type == 'polygon':
            self._region = _RoiPolygon(region_specs)

        else:
            raise NotImplementedError(f'Region_type {region_specs} has not been implemented yet.')

        self.polygon = self._region.polygon
        self.dimension = self._region.dimension
        self.centroid = self._region.centroid
        self.max_distance = self._region.max_distance
        self.region_measure = self._region.region_measure
        self.subregion_measure = self._region.subregion_measure

    def __repr__(self):
        """
        Readable, printable and savable representation of RoiRegion.
        """
        return str(dict(region_type=self.region_type, region_specs=self.region_specs))

    def contains(self, points):
        """
        Return list of indices for all points that are inside the region of interest.

        Parameters
        ----------
        points : array of 2D or 3D coordinates
            Points that are tested for being inside the specified region.

        Returns
        -------
        ndarray of ints
            Array with indices for all points in original point array that are within the region.
        """
        return self._region.contains(points)

    def as_artist(self, origin=(0, 0), **kwargs):
        """
        Matplotlib patch object for this region (e.g. `matplotlib.patches.Ellipse`).

        Parameters:
        -----------
        origin : array_like
            The (x, y) pixel position of the origin of the displayed image.
            Default is (0, 0).

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        patch : matplotlib.patches object
            Matplotlib patch for the specified region.
        """
        return self._region.as_artist(origin=origin, **kwargs)


class _RoiInterval:

    def __init__(self, region_specs):
        if len(region_specs) != 2:
            raise AttributeError('The shape of region_specs must be (2,).')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return dict(region_type='interval', region_specs=self.region_specs)

    def contains(self, points):
        inside_indices = np.where((points >= self.region_specs[0]) & (points <= self.region_specs[1]))
        return inside_indices

    @property
    def polygon(self):
        return np.array(self.region_specs)

    @property
    def dimension(self):
        return 1

    @property
    def centroid(self):
        minimum, maximum = self.region_specs
        return minimum + (maximum - minimum)/2

    @property
    def max_distance(self):
        minimum, maximum = self.region_specs
        return maximum - minimum

    @property
    def region_measure(self):
        minimum, maximum = self.region_specs
        return maximum - minimum

    @property
    def subregion_measure(self):
        return None

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError


class _RoiRectangle:

    def __init__(self, region_specs):
        if len(region_specs) != 4:
            raise AttributeError('The shape of region_specs must be (2,).')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return dict(region_type='rectangle', region_specs=self.region_specs)

    def contains(self, points):
        corner, width, height, angle = self.region_specs
        fig, ax = plt.subplots()
        rect = mPatches.Rectangle(corner, width, height, angle=angle, fill=False, edgecolor='b', linewidth=1)
        ax.add_patch(rect)
        polygon = rect.get_patch_transform().transform(rect.get_path().vertices)

        polygon_path = mPath.Path(polygon, closed=True)
        mask = polygon_path.contains_points(points)
        inside_indices = np.where(mask)[0]
        plt.close()
        return inside_indices

    @property
    def polygon(self):
        corner, width, height, angle = self.region_specs
        fig, ax = plt.subplots()
        rect = mPatches.Rectangle(corner, width, height, angle=angle, fill=False, edgecolor='b', linewidth=1)
        ax.add_patch(rect)
        polygon = rect.get_patch_transform().transform(rect.get_path().vertices)
        plt.close()
        return polygon[::-1]

    @property
    def dimension(self):
        return 2

    @property
    def centroid(self):
        corner, width, height, angle = self.region_specs
        center_before_rotation_x, center_before_rotation_y = corner[0] + width / 2, corner[1] + height / 2

        center_x = center_before_rotation_x * np.cos(angle) - center_before_rotation_y * np.sin(angle)
        center_y = center_before_rotation_x * np.sin(angle) + center_before_rotation_y * np.cos(angle)

        return center_x, center_y

    @property
    def max_distance(self):
        _, width, height, _ = self.region_specs
        return np.sqrt(width**2 + height**2)

    @property
    def region_measure(self):
        _, width, height, _ = self.region_specs
        return width * height

    @property
    def subregion_measure(self):
        _, width, height, _ = self.region_specs
        return 2 * width + 2 * height

    def as_artist(self, origin=(0, 0), **kwargs):
        from matplotlib.patches import Rectangle
        corner, width, height, angle = self.region_specs
        xy = corner[0] - origin[0], corner[1] - origin[1]
        return Rectangle(xy=xy, width=width, height=height, angle=angle, **kwargs)


class _RoiEllipse:

    def __init__(self, region_specs):
        if len(region_specs) != 4:
            raise AttributeError('The shape of region_specs must be (2,).')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return dict(region_type='ellipse', region_specs=self.region_specs)

    def contains(self, points):
        center, width, height, angle = self.region_specs

        cos_angle = np.cos(np.radians(-angle))
        sin_angle = np.sin(np.radians(-angle))

        xc = points[:,0] - center[0]
        yc = points[:,1] - center[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)

        inside_indices = np.where(rad_cc < 1.)[0]
        return inside_indices

    @property
    def polygon(self):
        center, width, height, angle = self.region_specs
        fig, ax = plt.subplots()
        rect = mPatches.Ellipse(center, width, height, angle=angle, fill=False, edgecolor='b', linewidth=1)
        ax.add_patch(rect)
        polygon = rect.get_patch_transform().transform(rect.get_path().vertices)
        plt.close()
        return polygon[::-1]

    @property
    def dimension(self):
        return 2

    @property
    def centroid(self):
        center, _, _, _ = self.region_specs
        return center

    @property
    def max_distance(self):
        _, width, height, _ = self.region_specs
        return np.max([width, height])

    @property
    def region_measure(self):
        _, width, height, _ = self.region_specs
        return np.pi * width/2 * height/2

    @property
    def subregion_measure(self):
        _, width, height, _ = self.region_specs
        # using Ramanujan approximation
        a, b = width / 2, height / 2
        t = ((a - b) / (a + b)) ** 2
        circumference = np.pi * (a + b) * (1 + 3 * t / (10 + np.sqrt(4 - 3 * t)))
        return circumference

    def as_artist(self, origin=(0, 0), **kwargs):
        from matplotlib.patches import Ellipse
        center, width, height, angle = self.region_specs
        xy = center[0] - origin[0], center[1] - origin[1]
        return Ellipse(xy=xy, width=width, height=height, angle=angle, **kwargs)


class _RoiPolygon:

    def __init__(self, region_specs):
        if not np.all(region_specs[0] == region_specs[-1]):
            raise ValueError('First and last element of polygon must be identical.')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return dict(region_type='polygon', region_specs=self.region_specs)

    def contains(self, points):
        polygon_path = mPath.Path(self.region_specs, closed=True)
        mask = polygon_path.contains_points(points)
        inside_indices = np.where(mask)[0]
        return inside_indices

    @property
    def polygon(self):
        return np.array(self.region_specs)

    @property
    def dimension(self):
        return 2

    @property
    def centroid(self):
        polygon = shPolygon(self.region_specs)
        return polygon.centroid.coords

    @property
    def max_distance(self):
        distances = pdist(self.region_specs[0:-1])
        return np.nanmax(distances)

    @property
    def region_measure(self):
        polygon = shPolygon(self.region_specs)
        return polygon.area

    @property
    def subregion_measure(self):
        polygon = shPolygon(self.region_specs)
        return polygon.length

    def as_artist(self, origin=(0, 0), **kwargs):
        from matplotlib.patches import Polygon
        #xy = self.region_specs[:,0] - origin[0], self.region_specs[:,1] - origin[1]
        xy = self.region_specs
        return Polygon(xy=xy, closed=True, **kwargs)
