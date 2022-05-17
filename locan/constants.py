"""

Constants to be used throughout the project.

.. autosummary::
   :toctree: ./

   DATASETS_DIR
   RENDER_ENGINE
   PROPERTY_KEYS
   RAPIDSTORM_KEYS
   ELYRA_KEYS
   THUNDERSTORM_KEYS
   N_JOBS
   LOCDATA_ID
   COLORMAP_CONTINUOUS
   COLORMAP_DIVERGING
   COLORMAP_CATEGORICAL
   TQDM_LEAVE
   TQDM_DISABLE
"""
from __future__ import annotations

from enum import Enum, unique
from pathlib import Path
from dataclasses import dataclass

from locan.dependencies import HAS_DEPENDENCY
if HAS_DEPENDENCY["colorcet"]: from colorcet import m_fire, m_gray, m_coolwarm, m_glasbey_dark


__all__ = ['DATASETS_DIR', 'PROPERTY_KEYS', 'PropertyKey', 'PropertyDescription',
           'HullType',
           'FileType', 'RenderEngine', 'RENDER_ENGINE',
           'RAPIDSTORM_KEYS', 'ELYRA_KEYS', 'THUNDERSTORM_KEYS',
           'N_JOBS', 'LOCDATA_ID', 'TQDM_LEAVE', 'TQDM_DISABLE',
           'COLORMAP_CONTINUOUS', 'COLORMAP_DIVERGING', 'COLORMAP_CATEGORICAL'
           ]


#: Standard directory for example datasets.
DATASETS_DIR = Path.home() / 'LocanDatasets'


@dataclass
class PropertyDescription:
    """
    A property of a localization or group of localizations
    representing column names in LocData.data and LocData.properties.
    """
    name: str
    type: str
    unit_SI: str | None = None
    unit: str | None = None
    description: str = ""


@unique
class PropertyKey(Enum):
    """
    Standard properties for `locan.LocData`.
    """
    index = PropertyDescription(
        'index',
        'integer',
        description="a positive integer identifying the localization")
    original_index = PropertyDescription(
        'original_index',
        'integer')
    position_x = PropertyDescription(
        'position_x',
        'float',
        unit_SI='m',
        description="spatial coordinate.")
    position_y = PropertyDescription(
        'position_y',
        'float',
        unit_SI='m',
        description="spatial coordinate.")
    position_z = PropertyDescription(
        'position_z',
        'float',
        unit_SI='m',
        description="spatial coordinate.")
    frame = PropertyDescription(
        'frame',
        'integer',
        description="frame  number in which the localization occurs")
    frames_number = PropertyDescription(
        'frames_number',
        'integer',
        description="number of frames that contribute to a merged localization")
    frames_missing = PropertyDescription(
        'frames_missing',
        'integer',
        description="number of frames that occurred between two successive localizations")
    time = PropertyDescription(
        'time',
        'float',
        unit_SI="s")
    intensity = PropertyDescription(
        'intensity',
        'float',
        description="intensity or emission strength as estimated by the fitter")
    local_background = PropertyDescription(
        'local_background',
        'float',
        description="background in the neighborhood of localization as estimated by the fitter")
    local_background_sigma = PropertyDescription(
        'local_background_sigma',
        'float',
        description="variation of local background in terms of standard deviation")
    signal_noise_ratio = PropertyDescription(
        'signal_noise_ratio',
        'float',
        description="ratio between mean intensity (i.e. intensity for a single localization) "
                    "and the standard deviation of local_background "
                    "(i.e. local_background_sigma for a single localization)")
    signal_background_ratio = PropertyDescription(
        'signal_background_ratio',
        'float',
        description="ratio between mean intensity (i.e. intensity for a single localization) and the local_background")
    chi_square = PropertyDescription(
        'chi_square',
        'float',
        description="chi-square value of the fitting procedure as estimated by the fitter")
    two_kernel_improvement = PropertyDescription(
        'two_kernel_improvement',
        'float',
        description="a rapidSTORM parameter describing the improvement from two kernel fitting")
    psf_amplitude = PropertyDescription('psf_amplitude', 'float')
    psf_width = PropertyDescription(
        'psf_width',
        'float',
        description="full-width-half-max of the fitted Gauss-function - "
                    "being isotropic or representing the root-mean-square of psf_width_c for all dimensions")
    psf_width_x = PropertyDescription(
        'psf_width_x',
        'float',
        description="full-width-half-max of the fitted Gauss-function in x-dimension as estimated by the fitter")
    psf_width_y = PropertyDescription(
        'psf_width_y',
        'float',
        description="full-width-half-max of the fitted Gauss-function in y-dimension as estimated by the fitter")
    psf_width_z = PropertyDescription(
        'psf_width_z',
        'float',
        description="full-width-half-max of the fitted Gauss-function in z-dimension as estimated by the fitter")
    psf_half_width = PropertyDescription('psf_half_width', 'float')
    psf_half_width_x = PropertyDescription('psf_half_width_x', 'float')
    psf_half_width_y = PropertyDescription('psf_half_width_y', 'float')
    psf_half_width_z = PropertyDescription('psf_half_width_z', 'float')
    psf_sigma = PropertyDescription(
        'psf_sigma',
        'float',
        description="sigma of the fitted Gauss-function - "
                    "being isotropic or representing the root-mean-square of psf_sigma_c for all dimensions")
    psf_sigma_x = PropertyDescription(
        'psf_sigma_x',
        'float',
        description="sigma of the fitted Gauss-function in x-dimension as estimated by the fitter")
    psf_sigma_y = PropertyDescription(
        'psf_sigma_y',
        'float',
        description="sigma of the fitted Gauss-function in y-dimension as estimated by the fitter")
    psf_sigma_z = PropertyDescription(
        'psf_sigma_z',
        'float',
        description="sigma of the fitted Gauss-function in z-dimension as estimated by the fitter")
    uncertainty = PropertyDescription(
        'uncertainty',
        'float',
        description="localization error for all dimensions or representing a value proportional to "
                    "psf_sigma / sqrt(intensity) or "
                    "representing the root-mean-square of uncertainty_c for all dimensions.")
    uncertainty_x = PropertyDescription(
        'uncertainty_x',
        'float',
        description="localization error in x-dimension estimated by a fitter "
                    "or representing a value proportional to psf_sigma_x / sqrt(intensity)")
    uncertainty_y = PropertyDescription(
        'uncertainty_y',
        'float',
        description="localization error in y-dimension estimated by a fitter "
                    "or representing a value proportional to psf_sigma_y / sqrt(intensity)")
    uncertainty_z = PropertyDescription(
        'uncertainty_z',
        'float',
        description="localization error in z-dimension estimated by a fitter "
                    "or representing a value proportional to psf_sigma_z / sqrt(intensity)")
    channel = PropertyDescription(
        'channel',
        'integer',
        description="identifier for an imaging channel")
    slice_z = PropertyDescription(
        'slice_z',
        'float',
        description="identifier for an imaging slice")
    plane = PropertyDescription(
        'plane',
        'integer',
        description="identifier for an imaging plane")
    cluster_label = PropertyDescription(
        'cluster_label',
        'integer',
        description="identifier for a localization cluster")

    @classmethod
    def coordinate_labels(cls) -> list[PropertyKey]:
        return [cls.position_x, cls.position_y, cls.position_z]

    @classmethod
    def summary(cls) -> str:
        lines = []
        for element in cls:
            lines.append(f"{element.value.name}")
            lines.append(f"type: {element.value.type}, unit_SI: {element.value.unit_SI}, unit: {element.value.unit}")
            lines.append(f"{element.value.description}\n")
        return '\n'.join(lines)


#: Keys for the most common LocData properties.
#: Values suggest a type for conversion.
#: If 'integer', 'signed', 'unsigned', 'float' :func:`pandas.to_numeric` can be applied.
#: Otherwise :func:`pandas.astype` can be applied.
PROPERTY_KEYS_deprecated = {
    'index': 'integer', 'original_index': 'integer',
    'position_x': 'float', 'position_y': 'float', 'position_z': 'float',
    'frame': 'integer', 'frames_number': 'integer', 'frames_missing': 'integer',
    'time': 'float',
    'intensity': 'float',
    'local_background': 'float', 'local_background_sigma': 'float',
    'signal_noise_ratio': 'float', 'signal_background_ratio': 'float',
    'chi_square': 'float', 'two_kernel_improvement': 'float',
    'psf_amplitude': 'float',
    'psf_width': 'float', 'psf_width_x': 'float', 'psf_width_y': 'float', 'psf_width_z': 'float',
    'psf_half_width': 'float', 'psf_half_width_x': 'float', 'psf_half_width_y': 'float', 'psf_half_width_z': 'float',
    'psf_sigma': 'float', 'psf_sigma_x': 'float', 'psf_sigma_y': 'float', 'psf_sigma_z': 'float',
    'uncertainty': 'float', 'uncertainty_x': 'float', 'uncertainty_y': 'float', 'uncertainty_z': 'float',
    'channel': 'integer',
    'slice_z': 'float',
    'plane': 'integer',
    'cluster_label': 'integer',
}

PROPERTY_KEYS = {item.name: item.value.type for item in PropertyKey if item.value.type is not None}


class HullType(Enum):
    """
    Hull definitions that are supported for LocData objects.
    """
    BOUNDING_BOX = 'bounding_box'
    CONVEX_HULL = 'convex_hull'
    ORIENTED_BOUNDING_BOX = 'oriented_bounding_box'
    ALPHA_SHAPE = 'alpha_shape'


# File types
class FileType(Enum):
    """
    File types for localization files.

    The listed file types are supported with input/output functions in :func:`io.io_locdata`.
    The types correspond to the metadata keys for LocData objects. That is they are equal to the file types in
    the protobuf message `locan.data.metadata_pb2.Metadata`.
    """
    UNKNOWN_FILE_TYPE = 0
    CUSTOM = 1
    RAPIDSTORM = 2
    ELYRA = 3
    THUNDERSTORM = 4
    ASDF = 5
    NANOIMAGER = 6
    RAPIDSTORMTRACK = 7
    SMLM = 8
    DECODE = 9
    SMAP = 10


# Render engines
class RenderEngine(Enum):
    """
    Engine to be use for rendering and displaying localization data as 2d or 3d images.

    Each engine represents a library to be used as backend for rendering and plotting.
    """
    if not HAS_DEPENDENCY["mpl_scatter_density"]:
        _ignore_ = 'MPL_SCATTER_DENSITY'
    if not HAS_DEPENDENCY["napari"]:
        _ignore_ = 'NAPARI'
    MPL = 0
    """matplotlib"""
    MPL_SCATTER_DENSITY = 1
    """mpl-scatter-density"""
    NAPARI = 2
    """napari"""


#: Render engine.
RENDER_ENGINE = RenderEngine.MPL


#: Mapping column names in RapidSTORM files to LocData property keys
RAPIDSTORM_KEYS = {
    'Position-0-0': 'position_x',
    'Position-1-0': 'position_y',
    'Position-2-0': 'position_z',
    'ImageNumber-0-0': 'frame',
    'Amplitude-0-0': 'intensity',
    'FitResidues-0-0': 'chi_square',
    'LocalBackground-0-0': 'local_background',
    'TwoKernelImprovement-0-0': 'two_kernel_improvement',
    'Position-0-0-uncertainty': 'uncertainty_x',
    'Position-1-0-uncertainty': 'uncertainty_y',
    'Position-2-0-uncertainty': 'uncertainty_z'
}


#: Mapping column names in Zeiss Elyra files to LocData property keys
ELYRA_KEYS = {
    'Index': 'original_index',
    'First Frame': 'frame',
    'Number Frames': 'frames_number',
    'Frames Missing': 'frames_missing',
    'Position X [nm]': 'position_x',
    'Position Y [nm]': 'position_y',
    'Position Z [nm]': 'position_z',
    'Precision [nm]': 'uncertainty',
    'Number Photons': 'intensity',
    'Background variance': 'local_background_sigma',
    'Chi square': 'chi_square',
    'PSF half width [nm]': 'psf_half_width',
    'PSF width [nm]': 'psf_width',
    'Channel': 'channel',
    'Z Slice': 'slice_z'
}


#: Mapping column names in Thunderstorm files to LocData property keys
THUNDERSTORM_KEYS = {
    'id': 'original_index',
    'frame': 'frame',
    'x [nm]': 'position_x',
    'y [nm]': 'position_y',
    'z [nm]': 'position_z',
    'uncertainty [nm]': 'uncertainty',
    'uncertainty_xy [nm]': 'uncertainty_x',
    'uncertainty_z [nm]': 'uncertainty_z',
    'intensity [photon]': 'intensity',
    'offset [photon]': 'local_background',
    'bkgstd [photon]': 'local_background_sigma',
    'chi2': 'chi_square',
    'sigma1 [nm]': 'psf_sigma_x',
    'sigma2 [nm]': 'psf_sigma_y',
    'sigma [nm]': 'psf_sigma',
    'detections': 'frames_number'
}


#: Mapping column names in Nanoimager files to LocData property keys
NANOIMAGER_KEYS = {
    'Channel': 'channel',
    'Frame': 'frame',
    'X (nm)': 'position_x',
    'Y (nm)': 'position_y',
    'Z (nm)': 'position_z',
    'Photons': 'intensity',
    'Background': 'local_background'
}


#: Mapping column names in SMLM files to LocData property keys
SMLM_KEYS = {
    'id': 'original_index',
    'frame': 'frame',
    'x': 'position_x',
    'y': 'position_y',
    'z': 'position_z',
    'x_position': 'position_x',
    'y_position': 'position_y',
    'z_position': 'position_z',

    'uncertainty [nm]': 'uncertainty',
    'uncertainty_xy [nm]': 'uncertainty_x',
    'uncertainty_z [nm]': 'uncertainty_z',
    'intensity': 'intensity',
    'Amplitude_0_0': 'intensity',
    'background': 'local_background',
    'LocalBackground_0_0': 'local_background',
    'FitResidues_0_0': 'chi_square',
}

#: Mapping column names in DECODE files to LocData property keys
DECODE_KEYS = {
    'id': 'original_index',
    'frame_ix': 'frame',
    'x': 'position_x',
    'y': 'position_y',
    'z': 'position_z',
    'bg': 'local_background',
    'phot': 'intensity',
}

#: Mapping column names in SMAP files to LocData property keys
SMAP_KEYS = {
    'frame': 'frame',
    'xnm': 'position_x',
    'ynm': 'position_y',
    'znm': 'position_z',
    'bg': 'local_background',
    'phot': 'intensity',
    'channel': 'channel',
    'xnmerr': 'uncertainty_x',
    'ynmerr': 'uncertainty_y',
    'znmerr': 'uncertainty_z',
}


#: The number of cores that are used in parallel for some algorithms.
#: Following the scikit convention: n_jobs is the number of parallel jobs to run.
#: If -1, then the number of jobs is set to the number of CPU cores.
N_JOBS = 1

#: Leave tqdm progress bars after finishing the iteration.
#: Flag to leave tqdm progress bars.
TQDM_LEAVE = True

#: Disable tqdm progress bars.
#: Flag to disable all tqdm progress bars.
TQDM_DISABLE = False

#: LocData identifier
#: Identifier for LocData objects that is reset for each locan session and incremented with each LocData instantiation.
LOCDATA_ID = 0

#: Default colormaps for plotting
#: Default colormaps for continuous, diverging and categorical scales are set to colorcet colormaps if imported or
#: matplotlib if not. We chose fire, coolwarm and glasbey_dark (colorcet) or viridis, coolwarm adn tab20 (matplotlib).
if HAS_DEPENDENCY["colorcet"]:
    COLORMAP_CONTINUOUS = m_fire
    COLORMAP_DIVERGING = m_coolwarm
    COLORMAP_CATEGORICAL = m_glasbey_dark
else:
    COLORMAP_CONTINUOUS = 'viridis'
    COLORMAP_DIVERGING = 'coolwarm'
    COLORMAP_CATEGORICAL = 'tab20'
