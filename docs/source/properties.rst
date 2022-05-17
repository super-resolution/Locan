.. _properties:

===========================
Properties
===========================

:class:`LocData`, the data class for localization data, carries certain properties that describe individual
(localization properties) or averaged features of the underlying localizations or groups thereof (LocData properties).
In the following we provide names (or keys) for those properties.

The datatype for all keys is `string`. We stick to the following conventions:

* be explicit
* start with lower case letters
* use underscore
* do not use CamelCase or blanks
* use reverse notation in the sense that coordinate identifiers or identifiers of statistical functions are added
  in the end (``position_x_mean_mean``)

A list of well defined property keys used throughout locan is given by the constant:
:class:`locan/constants/PropertyKey`. An up-to-date description can be inspected by
`locan/constants/PropertyKey.index.value.description`.


Localization properties:
========================

Each localization has properties that can usually be identified in the various input (file) formats.
We will use the following keys (where `c` stands for the coordinate `x`, `y` or `z` ):

    * `index`
        localization index
    * `position_c`
        coordinate for the c-position
    * `frame`
        frame  number in which the localization occurs
    * `intensity`
        intensity or emission strength as estimated by the fitter
    * `local_background`
        background in the neighborhood of localization as estimated by the fitter
    * `local_background_sigma`
        variation of local background in terms of standard deviation
    * `signal_noise_ratio`
        ratio between mean intensity (i.e. intensity for a single localization)
        and the standard deviation of local_background (i.e. local_background_sigma for a single localization)
    * `signal_background_ratio`
        ratio between mean intensity (i.e. `intensity` for a single localization) and the local_background
    * `chi_square`
        chi-square value of the fitting procedure as estimated by the fitter
    * `psf_sigma_c`
        sigma of the fitted Gauss-function in c-dimension as estimated by the fitter
    * `psf_sigma`
        sigma of the fitted Gauss-function -
        being isotropic or representing the root-mean-square of psf_sigma_c for all dimensions
    * `psf_width_c`
        full-width-half-max of the fitted Gauss-function in c-dimension as estimated by the fitter
    * `psf_width`
        full-width-half-max of the fitted Gauss-function -
        being isotropic or representing the root-mean-square of psf_width_c for all dimensions
    * `uncertainty_c`
        localization error in c-dimension estimated by a fitter
        or representing a value proportional to psf_sigma_c / sqrt(intensity)
    * `uncertainty`
        localization error for all dimensions
        or representing a value proportional to psf_sigma / sqrt(intensity)
        or representing the root-mean-square of uncertainty_c for all dimensions.
    * `channel`
        identifier for various channels
    * `two_kernel_improvement`
        a rapidSTORM parameter describing the improvement from two kernel fitting
    * `frames_number`
        number of frames that contribute to a merged localization
    * `frames_missing`
        number of frames that occurred between two successive localizations


Additional localization properties are computed by various analysis procedures including (among others):

    * `cluster_label`
        identifier for a localization cluster
    * `nn_distance`
        nearest-neighbor distance
    * `nn_distance_k`
        k-nearest-neighbor distance (k can be any integer)
    * `colocalization_cbc`
        coordinate-based colocalization value


LocData properties:
========================

A group of localizations makes up a LocData entity for which further properties are defined.

The coordinates for a localization group are defined by their centroids.

In general, we will use the following keys for statistics of localization properties:

    * `property_stats`
        where property is the localization property key and stats is one of the following:

        * count (number of elements)
        * min (minimum of all elements)
        * max (maximum of all elements)
        * sum (sum of all elements)
        * mean (the mean of all elements)
        * std (standard deviation of all elements)
        * sem (standard error of the mean of all elements)

For example:

    * `intensity_sum`
        total intensity of all localizations in the group

Some properties are derived from a hull of all element positions. We provide four hulls:

1. bounding box
2. convex hull
3. alpha shape
4. oriented bounding box

From each hull a region measure (e.g. the area in 2D) and a subregion measure (e.g. the circumference in 2D) is computed.

We will use the following keys for additional properties (where `c` stands for the coordinate `x`, `y` or `z`
and `h` stands for the corresponding hull `bb`, `ch`, `as`, `obb`):

    * centroid
        tuple with mean of all localization coordinates
    * `localization_count`
        number of localizations within a group
    * `region_measure_h`
        area/volume (for all possible hulls)
    * `subregion_measure_h`
        circumference/surface (for all possible hulls)
    * `localization_density_h`
        density of localizations (for all possible hulls)
    * `boundary_localizations_h`
        absolute number of localizations on boundary (for all possible hulls)
    * `boundary_localizations_ratio_h`
        ratio between number of localizations on hull boundary and within hull (for all possible hulls)
    * `max_distance`
        maximum distance between any two localizations
    * `inertia_moments`
        inertia moments of all localizations
    * `orientation_obb`
        angle between the x-axis and the long axis of the oriented bounding box
    * `orientation_im`
        angle between inertia moment principal component vectors
    * `circularity_obb`
        elongation estimated from oriented bounding box
    * `circularity_im`
        excentricity estimated from inertia moments
