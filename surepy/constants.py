"""

Constants to be used throughout the project.

.. autosummary::
   :toctree: ./

   ROOT_DIR
   PROPERTY_KEYS
   HULL_KEYS
   FileType
   RenderEngine
   RAPIDSTORM_KEYS
   ELYRA_KEYS
   THUNDERSTORM_KEYS
   N_JOBS
   LOCDATA_ID
   COLORMAP_CONTINUOUS
   COLORMAP_DIVERGING
   COLORMAP_CATEGORICAL

"""
import os
from enum import Enum
from pathlib import Path

# Optional imports
try:
    from colorcet import m_fire, m_gray, m_coolwarm, m_glasbey_dark
    _has_colorcet = True
except ImportError:
    _has_colorcet = False

try:
    import mpl_scatter_density
    _has_mpl_scatter_density = True
except ImportError:
    _has_mpl_scatter_density = False

try:
    import napari
    _has_napari = True
except ImportError:
    _has_napari = False


__all__ = ['ROOT_DIR', 'PROPERTY_KEYS', 'HULL_KEYS', 'FileType', 'RenderEngine', 'RAPIDSTORM_KEYS', 'ELYRA_KEYS', 'THUNDERSTORM_KEYS',
           'N_JOBS', 'LOCDATA_ID', 'COLORMAP_CONTINUOUS', 'COLORMAP_DIVERGING', 'COLORMAP_CATEGORICAL']


#: Root directory for path operations.
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(__file__).parent


#: Keys for the most common LocData properties.
PROPERTY_KEYS = ['index', 'original_index', 'position_x', 'position_y', 'position_z', 'frame', 'intensity',
                 'local_background', 'chi_square',
                 'psf_sigma_x', 'psf_sigma_y', 'psf_sigma_z', 'uncertainty_x', 'uncertainty_y', 'uncertainty_z',
                 'channel', 'index', 'cluster_label', 'two_kernel_improvement']


#: Keys for the most common hulls.
HULL_KEYS = {'bounding_box', 'convex_hull', 'oriented_bounding_box', 'alpha_shape'}


# File types
class FileType(Enum):
    """
    File types for localization files.

    The listed file types are supported with input/output functions in io.io_locdata.
    The types correspond to the metadata keys for LocData objects. That is they are equal to the file types in
    the protobuf message metadata_pb2.
    """
    UNKNOWN_FILE_TYPE = 0
    CUSTOM = 1
    RAPIDSTORM = 2
    ELYRA = 3
    THUNDERSTORM = 4
    ASDF = 5


# Render engines
class RenderEngine(Enum):
    """
    Engine to be use for rendering and displaying localization data as 2d or 3d images.

    Each engine represents a library to be used as backend for rendering and plotting.
    MPL: matplotlib
    MPL_SCATTER_DENSITY: mpl-scatter-density
    NAPARI: napari
    """
    if not _has_mpl_scatter_density:
        _ignore_ = 'MPL_SCATTER_DENSITY'
    if not _has_napari:
        _ignore_ = 'NAPARI'
    MPL = 0
    MPL_SCATTER_DENSITY = 1
    NAPARI = 2


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
    'Precision [nm]': 'precision',
    'Number Photons': 'intensity',
    'Background variance': 'local_background',
    'Chi square': 'chi_square',
    'PSF half width [nm]': 'psf_half_width',
    'Channel': 'channel',
    'Z Slice': 'slice_z'
}

# todo: map 'Number Frames', 'Frames Missing', 'Precision [nm]', 'psf_half_width', 'channel',
#  'slice_z' to surepy property

#: Mapping column names in Thunderstorm files to LocData property keys
THUNDERSTORM_KEYS = {
    'id': 'original_index',
    'frame': 'frame',
    'x [nm]': 'position_x',
    'y [nm]': 'position_y',
    'z [nm]': 'position_z',
    'uncertainty_xy [nm]': 'uncertainty_x',
    'uncertainty_z [nm]': 'uncertainty_z',
    'intensity [photon]': 'intensity',
    'offset [photon]': 'local_background',
    'chi2': 'chi_square',
    'sigma1 [nm]': 'psf_sigma_x',
    'sigma2 [nm]': 'psf_sigma_y',
    'sigma [nm]': 'psf_sigma_x'
}
# todo: uncertainty_xy and sigma [nm] are mapped to uncertainty_x and psf_sigma_x. Possible conflict?
# todo: map "bkgstd [photon]", "uncertainty [nm]" to something usefull

#: The number of cores that are used in parallel for some algorithms.
#: Following the scikit convention: n_jobs is the number of parallel jobs to run.
#: If -1, then the number of jobs is set to the number of CPU cores.
N_JOBS = 1

#: LocData identifier
#: Identifier for LocData objects that is reset for each surepy session.
LOCDATA_ID = 0

#: Default colormaps for plotting
#: Default colormaps for continuous, diverging and categorical scales are set to colorcet colormaps if imported or
#: matplotlib if not. We chose fire, coolwarm and glasbey_dark (colorcet) or viridis, coolwarm adn tab20 (matplotlib).
if _has_colorcet:
    COLORMAP_CONTINUOUS = m_fire
    COLORMAP_DIVERGING = m_coolwarm
    COLORMAP_CATEGORICAL = m_glasbey_dark
else:
    COLORMAP_CONTINUOUS = 'viridis'
    COLORMAP_DIVERGING = 'coolwarm'
    COLORMAP_CATEGORICAL = 'tab20'
