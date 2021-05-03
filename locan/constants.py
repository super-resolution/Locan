"""

Constants to be used throughout the project.

.. autosummary::
   :toctree: ./

   ROOT_DIR
   DATASETS_DIR
   QT_BINDINGS
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

"""
from enum import Enum
from pathlib import Path
import os
import warnings
import importlib.util


__all__ = ['ROOT_DIR', 'DATASETS_DIR', 'PROPERTY_KEYS', 'HullType',
           'QtBindings', 'QT_BINDINGS', 'FileType', 'RenderEngine', 'RENDER_ENGINE',
           'RAPIDSTORM_KEYS', 'ELYRA_KEYS', 'THUNDERSTORM_KEYS',
           'N_JOBS', 'LOCDATA_ID',
           'COLORMAP_CONTINUOUS', 'COLORMAP_DIVERGING', 'COLORMAP_CATEGORICAL'
           ]


# Provide list of dependencies with corresponding module names for import in python.
# Should reflect the dependencies specified in setup.cfg.
# Some package names (as recommended for pip install) are different from the names for import.
INSTALL_REQUIRES = ['asdf', 'tifffile', 'ruamel.yaml', 'fast_histogram', 'boost_histogram', 'hdbscan', 'lmfit',
                    'google.protobuf',
                    'shapely', 'networkx', 'sklearn', 'skimage', 'matplotlib', 'scipy', 'pandas', 'numpy', 'tqdm',
                    'numba', 'cython']

EXTRAS_REQUIRE = {'Colormaps': ["colorcet"], 'Track': ["trackpy"], 'Register': ["open3d"],
                  'Render': ["napari", "mpl_scatter_density"], 'QT': ["PySide2"],
                  'Dev': ['twine', 'sphinx', 'ipython', 'myst-nb', 'sphinx-copybutton', 'sphinx_rtd_theme', 'furo']
                  }


# Possible python bindings to interact with QT
class QtBindings(Enum):
    """
    Python bindings used to interact with Qt.
    """
    NONE = 'none'
    PYSIDE2 = 'pyside2'
    PYQT5 = 'pyqt5'


# Force python binding - only for testing purposes:
# os.environ['QT_API'] = QtBindings.PYQT5.value


# Determine Python bindings for QT interaction.
if 'QT_API' in os.environ:  # this is the case that QT_API has been set.
    try:
        QT_BINDINGS = QtBindings(os.environ['QT_API'])
    except KeyError:
        warnings.warn(f'The requested QT_API {os.environ["QT_API"]} cannot be imported. Will continue without QT.')
        QT_BINDINGS = QtBindings.NONE

    if QT_BINDINGS == QtBindings.PYSIDE2:
        if importlib.util.find_spec("PySide2") is None:
            warnings.warn(f'The requested QT_API {QT_BINDINGS.value} cannot be imported. Will continue without QT.')
            QT_BINDINGS = QtBindings.NONE

    elif QT_BINDINGS == QtBindings.PYQT5:
        if importlib.util.find_spec("PyQt5") is None:
            warnings.warn(f'The requested QT_API {QT_BINDINGS.value} cannot be imported. Will continue without QT.')
            QT_BINDINGS = QtBindings.NONE

else:  # this is the case that QT_API has not been set.
    if importlib.util.find_spec("PySide2") is not None:
        QT_BINDINGS = QtBindings.PYSIDE2
    elif importlib.util.find_spec("PyQt5") is not None:
        QT_BINDINGS = QtBindings.PYQT5
    else:
        QT_BINDINGS = QtBindings.NONE  # this is the case that no qt bindings are available.

# In order to force napari and other QT-using libraries to import with the correct Qt bindings
# the environment variable QT_API has to be set.
# See use of qtpy in napari which default to pyqt5 if both bindings are installed.
if QT_BINDINGS != QtBindings.NONE:
    os.environ['QT_API'] = QT_BINDINGS.value


# Optional imports
_has_cupy = importlib.util.find_spec("cupy") is not None
_has_mpl_scatter_density = importlib.util.find_spec("mpl_scatter_density") is not None
_has_napari = importlib.util.find_spec("napari") is not None
_has_open3d = importlib.util.find_spec("open3d") is not None
_has_trackpy = importlib.util.find_spec("trackpy") is not None
_has_boost_histogram = importlib.util.find_spec("boost_histogram") is not None

try:
    from colorcet import m_fire, m_gray, m_coolwarm, m_glasbey_dark
    _has_colorcet = True
except ImportError:
    _has_colorcet = False

#: Root directory for path operations.
ROOT_DIR = Path(__file__).parent


#: Standard directory for example datasets.
DATASETS_DIR = ROOT_DIR.parent.parent / 'Locan_datasets'


#: Keys for the most common LocData properties.
#: Values suggest a type for conversion.
#: If 'integer', 'signed', 'unsigned', 'float' :func:`pandas.to_numeric` can be applied.
#: Otherwise :func:`pandas.astype` can be applied.
PROPERTY_KEYS = {'index': 'integer', 'original_index': 'integer',
                 'position_x': 'float', 'position_y': 'float', 'position_z': 'float',
                 'frame': 'integer', 'frames_number': 'integer', 'frames_missing': 'integer',
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


# Render engines
class RenderEngine(Enum):
    """
    Engine to be use for rendering and displaying localization data as 2d or 3d images.

    Each engine represents a library to be used as backend for rendering and plotting.
    """
    if not _has_mpl_scatter_density:
        _ignore_ = 'MPL_SCATTER_DENSITY'
    if not _has_napari:
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
if _has_colorcet:
    COLORMAP_CONTINUOUS = m_fire
    COLORMAP_DIVERGING = m_coolwarm
    COLORMAP_CATEGORICAL = m_glasbey_dark
else:
    COLORMAP_CONTINUOUS = 'viridis'
    COLORMAP_DIVERGING = 'coolwarm'
    COLORMAP_CATEGORICAL = 'tab20'
