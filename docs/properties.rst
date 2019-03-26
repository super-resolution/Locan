.. _properties:

===========================
Properties
===========================

Locdata, the data class for localization data, carries certain properties that describe individual or
averaged features of the underlying localizations or groups thereof.
In the following we provide names (or keys) for those properties.

The datatype for all keys is string. We stick to the following conventions:

    * start with lower case letters
    * use underscore
    * do not use CamelCase or blanks
    * add coordinate identifiers or identifiers of statistical functions in the end ('position_x_mean_mean')

A list of well defined property keys is given in the constant: DATA_PROPERTY_KEYS.


Localization properties:
========================

Each localization has properties that can usually be identified in the various file formats. We will use the following
keys (where 'c' stands for the coordinate 'x', 'y' or 'z' ):

    * 'index'
        localization index
    * 'position_c'
        coordinate for the c-position
    * 'frame'
        frame  number in which the localization occurs
    * 'intensity'
        intensity or emission strength as estimated by the fitter
    * 'local_background'
        background in the neighborhood of localization as estimated by the fitter
    * 'chi_square'
        chi-square value of the fitting procedure as estimated by the fitter
    * 'psf_sigma_c'
        sigma of the fitted Gauss-function in c-dimension as estimated by the fitter
    * 'uncertainty_c'
        localization error in c-dimension estimated by a fitter
    * 'channel'
        identifier for various channels
    * 'two_kernel_improvement'
        a rapidSTORM parameter describing the improvement from two kernel fitting.

Additional localization properties are computed by various analysis procedures including:

    * 'cluster_label'
        identifier for a localization cluster
    * 'nn_distance'
        nearest-neighbor distance
    * 'nn_distance_k'
        k-nearest-neighbor distance (k can be any integer)
    * 'colocalization_cbc'
        coordinate-based colocalization value


Locdata properties:
========================

A group of localizations makes up a LocData entity for which further properties are defined. Groups of locdata are
further combined in Locdata entities with additional properties.

The new locdata coordinates are computed from the centroids of grouped localizations.

We will use the following keys for statistics of localization properties:

    * 'property_stats'
        where property is the localization property key and stats is one of the following:

            * count (number of elements)
            * min (minimum of all elements)
            * max (maximum of all elements)
            * sum (sum of all elements)
            * mean (the mean of all elements)
            * std (standard deviation of all elements)
            * sem (standard error of the mean of all elements)

For example:

    * 'intensity_sum'
        total intensity of all localizations in the group

Some properties are derived from a hull of all localizations or centroids. We provide four hulls:

    1. bounding box
    2. convex hull
    3. alpha shape
    4. oriented bounding box

From each hull a region measure (the area in 2D) and a subregion measure (the circumference in 2D) is computed.

We will use the following keys for additional properties (where 'c' stands for the coordinate 'x', 'y' or 'z'
and 'h' stands for the corresponding hull 'bb', 'ch', 'as', 'obb'):

    * centroid
        tuple with mean of all localization coordinates
    * 'localization_count'
        number of localizations from all groups
    * 'region_measure_h'
        area/volume (for all possible hulls)
    * 'subregion_measure_h'
        circumference/surface (for all possible hulls)
    * 'localization_density_h'
        density of localizations (for all possible hulls)
    * 'boundary_localizations_h'
        absolute number of localizations on boundary (for all possible hulls)
    * 'boundary_localizations_ratio_h'
        ratio between number of localizations on hull boundary and within hull (for all possible hulls)
    * 'max_distance'
        maximum distance between any two localizations
    * 'inertia_moments'
        inertia moments of all localizations
    * 'orientation_obb'
        angle between the x-axis and the long axis of the oriented bounding box
    * 'orientation_im'
        angle between inertia moment principal component vectors
    * 'circularity_obb'
        elongation estimated from oriented bounding box
    * 'circularity_im'
        excentricity estimated from inertia moments
