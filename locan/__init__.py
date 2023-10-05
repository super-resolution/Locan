"""
.. currentmodule:: locan

Locan consists of the following modules:

.. autosummary::
   :toctree: generated/

   analysis
   constants
   configuration
   data
   datasets
   dependencies
   gui
   locan_io
   scripts
   simulation
   utils
   visualize
   tests
"""
from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from locan._version import version as __version__
except ImportError:
    __version__ = "not-installed"


# Root directory for path operations.
ROOT_DIR: Path = Path(__file__).parent

# Identifier for LocData objects that is reset for each locan session and incremented with each LocData instantiation.
locdata_id: int = 0


# isort: off
from locan.dependencies import *  # imported by locan.configuration
from locan.constants import *  # imported by locan.configuration
from locan.configuration import *  # has to be imported before any others due to pyqt5/sidepy2 issues.

# isort: on

submodules: list[str] = [
    "analysis",
    "configuration",
    "constants",
    "data",
    "datasets",
    "dependencies",
    "gui",
    "locan_io",
    "simulation",
    "tests",
    "utils",
    "visualize",
]

__all__: list[str] = ["__version__", "ROOT_DIR"]

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="locan")
    __all__.extend(module_.__all__)

# the following re-exports are required by mypy
from locan.analysis import (  # type:ignore[attr-defined]
    AccumulationClusterCheck as AccumulationClusterCheck,
    BlinkStatistics as BlinkStatistics,
    ConvexHullExpectation as ConvexHullExpectation,
    ConvexHullExpectationBatch as ConvexHullExpectationBatch,
    CoordinateBasedColocalization as CoordinateBasedColocalization,
    Drift as Drift,
    DriftComponent as DriftComponent,
    GroupedPropertyExpectation as GroupedPropertyExpectation,
    LocalizationPrecision as LocalizationPrecision,
    LocalizationProperty as LocalizationProperty,
    LocalizationProperty2d as LocalizationProperty2d,
    LocalizationPropertyCorrelations as LocalizationPropertyCorrelations,
    LocalizationUncertainty as LocalizationUncertainty,
    LocalizationUncertaintyFromIntensity as LocalizationUncertaintyFromIntensity,
    LocalizationsPerFrame as LocalizationsPerFrame,
    NearestNeighborDistances as NearestNeighborDistances,
    Pipeline as Pipeline,
    PositionVarianceExpectation as PositionVarianceExpectation,
    RipleysHFunction as RipleysHFunction,
    RipleysKFunction as RipleysKFunction,
    RipleysLFunction as RipleysLFunction,
    SubpixelBias as SubpixelBias,
)
from locan.configuration import (  # type:ignore[attr-defined]
    COLORMAP_CATEGORICAL as COLORMAP_CATEGORICAL,
    COLORMAP_CONTINUOUS as COLORMAP_CONTINUOUS,
    COLORMAP_DIVERGING as COLORMAP_DIVERGING,
    DATASETS_DIR as DATASETS_DIR,
    N_JOBS as N_JOBS,
    QT_BINDING as QT_BINDING,
    RENDER_ENGINE as RENDER_ENGINE,
    TQDM_DISABLE as TQDM_DISABLE,
    TQDM_LEAVE as TQDM_LEAVE,
)
from locan.constants import (  # type:ignore[attr-defined]
    ColorMaps as ColorMaps,
    DECODE_KEYS as DECODE_KEYS,
    ELYRA_KEYS as ELYRA_KEYS,
    FileType as FileType,
    HullType as HullType,
    NANOIMAGER_KEYS as NANOIMAGER_KEYS,
    PROPERTY_KEYS as PROPERTY_KEYS,
    PropertyDescription as PropertyDescription,
    PropertyKey as PropertyKey,
    RAPIDSTORM_KEYS as RAPIDSTORM_KEYS,
    RenderEngine as RenderEngine,
    SMAP_KEYS as SMAP_KEYS,
    SMLM_KEYS as SMLM_KEYS,
    THUNDERSTORM_KEYS as THUNDERSTORM_KEYS,
)
from locan.data import (  # type:ignore[attr-defined]
    AlphaComplex as AlphaComplex,
    AlphaShape as AlphaShape,
    AxisOrientedCuboid as AxisOrientedCuboid,
    AxisOrientedHypercuboid as AxisOrientedHypercuboid,
    Bins as Bins,
    BoundingBox as BoundingBox,
    ConvexHull as ConvexHull,
    Cuboid as Cuboid,
    Ellipse as Ellipse,
    EmptyRegion as EmptyRegion,
    Interval as Interval,
    LocData as LocData,
    MultiPolygon as MultiPolygon,
    OrientedBoundingBox as OrientedBoundingBox,
    Polygon as Polygon,
    Rectangle as Rectangle,
    Region as Region,
    Region1D as Region1D,
    Region2D as Region2D,
    Region3D as Region3D,
    RegionND as RegionND,
    Roi as Roi,
    RoiRegion as RoiRegion,
    Selector as Selector,
    bunwarp as bunwarp,
    cluster_by_bin as cluster_by_bin,
    cluster_dbscan as cluster_dbscan,
    cluster_hdbscan as cluster_hdbscan,
    distance_to_region as distance_to_region,
    distance_to_region_boundary as distance_to_region_boundary,
    exclude_sparse_points as exclude_sparse_points,
    expand_region as expand_region,
    filter_condition as filter_condition,
    histogram as histogram,
    inertia_moments as inertia_moments,
    link_locdata as link_locdata,
    load_metadata_from_toml as load_metadata_from_toml,
    localizations_in_cluster_regions as localizations_in_cluster_regions,
    max_distance as max_distance,
    merge_metadata as merge_metadata,
    message_scheme as message_scheme,
    metadata_from_toml_string as metadata_from_toml_string,
    metadata_to_formatted_string as metadata_to_formatted_string,
    overlay as overlay,
    random_subset as random_subset,
    randomize as randomize,
    range_from_collection as range_from_collection,
    ranges as ranges,
    rasterize as rasterize,
    regions_union as regions_union,
    register_cc as register_cc,
    register_icp as register_icp,
    select_by_condition as select_by_condition,
    select_by_image_mask as select_by_image_mask,
    select_by_region as select_by_region,
    serial_clustering as serial_clustering,
    standardize as standardize,
    statistics as statistics,
    surrounding_region as surrounding_region,
    track as track,
    transform_affine as transform_affine,
    transform_counts_to_photons as transform_counts_to_photons,
)
from locan.datasets import (  # type:ignore[attr-defined]
    load_npc as load_npc,
    load_tubulin as load_tubulin,
)
from locan.dependencies import (  # type:ignore[attr-defined]
    EXTRAS_REQUIRE as EXTRAS_REQUIRE,
    HAS_DEPENDENCY as HAS_DEPENDENCY,
    IMPORT_NAMES as IMPORT_NAMES,
    INSTALL_REQUIRES as INSTALL_REQUIRES,
    QtBindings as QtBindings,
    needs_package as needs_package,
)
from locan.gui import (  # type:ignore[attr-defined]
    file_dialog as file_dialog,
)
from locan.locan_io import (  # type:ignore[attr-defined]
    Files as Files,
    convert_property_names as convert_property_names,
    convert_property_types as convert_property_types,
    find_file_upstream as find_file_upstream,
    load_Elyra_file as load_Elyra_file,
    load_Elyra_header as load_Elyra_header,
    load_Nanoimager_file as load_Nanoimager_file,
    load_Nanoimager_header as load_Nanoimager_header,
    load_SMAP_file as load_SMAP_file,
    load_SMAP_header as load_SMAP_header,
    load_SMLM_file as load_SMLM_file,
    load_SMLM_header as load_SMLM_header,
    load_SMLM_manifest as load_SMLM_manifest,
    load_asdf_file as load_asdf_file,
    load_decode_file as load_decode_file,
    load_decode_header as load_decode_header,
    load_locdata as load_locdata,
    load_rapidSTORM_file as load_rapidSTORM_file,
    load_rapidSTORM_header as load_rapidSTORM_header,
    load_rapidSTORM_track_file as load_rapidSTORM_track_file,
    load_rapidSTORM_track_header as load_rapidSTORM_track_header,
    load_thunderstorm_file as load_thunderstorm_file,
    load_thunderstorm_header as load_thunderstorm_header,
    load_txt_file as load_txt_file,
    manifest_file_info_from_locdata as manifest_file_info_from_locdata,
    manifest_format_from_locdata as manifest_format_from_locdata,
    manifest_from_locdata as manifest_from_locdata,
    save_SMAP_csv as save_SMAP_csv,
    save_SMLM as save_SMLM,
    save_asdf as save_asdf,
    save_thunderstorm_csv as save_thunderstorm_csv,
)
from locan.simulation import (  # type:ignore[attr-defined]
    add_drift as add_drift,
    make_Matern as make_Matern,
    make_NeymanScott as make_NeymanScott,
    make_Poisson as make_Poisson,
    make_Thomas as make_Thomas,
    make_cluster as make_cluster,
    make_dstorm as make_dstorm,
    make_uniform as make_uniform,
    resample as resample,
    simulate_Matern as simulate_Matern,
    simulate_NeymanScott as simulate_NeymanScott,
    simulate_Poisson as simulate_Poisson,
    simulate_Thomas as simulate_Thomas,
    simulate_cluster as simulate_cluster,
    simulate_dstorm as simulate_dstorm,
    simulate_frame_numbers as simulate_frame_numbers,
    simulate_tracks as simulate_tracks,
    simulate_uniform as simulate_uniform,
)
from locan.tests import (  # type:ignore[attr-defined]
    test as test,
)
from locan.utils import (  # type:ignore[attr-defined]
    biased_variance as biased_variance,
    dependency_info as dependency_info,
    iterate_2d_array as iterate_2d_array,
    ratio_fwhm_to_sigma as ratio_fwhm_to_sigma,
    show_versions as show_versions,
    system_info as system_info,
    weighted_mean_variance as weighted_mean_variance,
)
from locan.visualize import (  # type:ignore[attr-defined]
    HistogramEqualization as HistogramEqualization,
    Trafo as Trafo,
    adjust_contrast as adjust_contrast,
    apply_window as apply_window,
    get_rois as get_rois,
    render_2d as render_2d,
    render_2d_mpl as render_2d_mpl,
    render_2d_napari as render_2d_napari,
    render_2d_napari_image as render_2d_napari_image,
    render_2d_rgb_mpl as render_2d_rgb_mpl,
    render_2d_rgb_napari as render_2d_rgb_napari,
    render_2d_scatter_density as render_2d_scatter_density,
    render_3d as render_3d,
    render_3d_napari as render_3d_napari,
    render_3d_napari_image as render_3d_napari_image,
    render_3d_rgb_napari as render_3d_rgb_napari,
    save_rois as save_rois,
    scatter_2d_mpl as scatter_2d_mpl,
    scatter_3d_mpl as scatter_3d_mpl,
    select_by_drawing_napari as select_by_drawing_napari,
)
