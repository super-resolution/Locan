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
from enum import Enum
from pathlib import Path
from locan.dependencies import HAS_DEPENDENCY
if HAS_DEPENDENCY["colorcet"]: from colorcet import m_fire, m_gray, m_coolwarm, m_glasbey_dark


__all__ = ['DATASETS_DIR', 'PROPERTY_KEYS', 'HullType',
           'FileType', 'RenderEngine', 'RENDER_ENGINE',
           'RAPIDSTORM_KEYS', 'ELYRA_KEYS', 'THUNDERSTORM_KEYS',
           'N_JOBS', 'LOCDATA_ID', 'TQDM_LEAVE', 'TQDM_DISABLE',
           'COLORMAP_CONTINUOUS', 'COLORMAP_DIVERGING', 'COLORMAP_CATEGORICAL'
           ]


#: Standard directory for example datasets.
DATASETS_DIR = Path.home() / 'LocanDatasets'


#: Keys for the most common LocData properties.
#: Values suggest a type for conversion.
#: If 'integer', 'signed', 'unsigned', 'float' :func:`pandas.to_numeric` can be applied.
#: Otherwise :func:`pandas.astype` can be applied.
PROPERTY_KEYS = {'index': 'integer', 'original_index': 'integer',
                 'position_x': 'float', 'position_y': 'float', 'position_z': 'float',
                 'frame': 'integer', 'frames_number': 'integer', 'frames_missing': 'integer',
                 'time': 'float',
                 'intensity': 'float',
                 'local_background': 'float', 'local_background_sigma': 'float',
                 'signal_noise_ratio': 'float', 'signal_background_ratio': 'float',
                 'chi_square': 'float', 'two_kernel_improvement': 'float',
                 'psf_sigma': 'float', 'psf_sigma_x': 'float', 'psf_sigma_y': 'float', 'psf_sigma_z': 'float',
                 'psf_width': 'float', 'psf_half_width': 'float',
                 'uncertainty': 'float', 'uncertainty_x': 'float', 'uncertainty_y': 'float', 'uncertainty_z': 'float',
                 'channel': 'integer', 'cluster_label': 'integer',
                 'slice_z': 'float'
                 }


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
