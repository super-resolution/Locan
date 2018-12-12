"""

Constants to be used throughout the project.

.. autosummary::
   :toctree: ./

   ROOT_DIR
   PROPERTY_KEYS
   HULL_KEYS
   RAPIDSTORM_KEYS
   ELYRA_KEYS
   THUNDERSTORM_KEYS
   N_JOBS

"""
import os
from enum import Enum


#: Root directory for path operations.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


#: Keys for the most common LocData properties.
PROPERTY_KEYS = ['Index', 'Original_index', 'Position_x', 'Position_y', 'Position_z', 'Frame', 'Intensity', 'Local_background', 'Chi_square',
                 'Psf_sigma_x', 'Psf_sigma_y', 'Psf_sigma_z', 'Uncertainty_x', 'Uncertainty_y', 'Uncertainty_z',
                 'Channel', 'Index', 'Cluster_label', 'Two_kernel_improvement']


#: Keys for the most common hulls.
HULL_KEYS = {'bounding_box', 'convex_hull', 'oriented_bounding_box', 'alpha_shape'}

# File types
class File_type(Enum):
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


#: Mapping column names in RapidSTORM files to LocData property keys
RAPIDSTORM_KEYS = {
    'Position-0-0': 'Position_x',
    'Position-1-0': 'Position_y',
    'Position-2-0': 'Position_z',
    'ImageNumber-0-0': 'Frame',
    'Amplitude-0-0': 'Intensity',
    'FitResidues-0-0': 'Chi_square',
    'LocalBackground-0-0': 'Local_background',
    'TwoKernelImprovement-0-0': 'Two_kernel_improvement'
}


#: Mapping column names in Zeiss Elyra files to LocData property keys
ELYRA_KEYS = {
    'Index': 'Original_index',
    'First Frame': 'Frame',
    'Number Frames': 'Frames_number',
    'Frames Missing': 'Frames_missing',
    'Position X [nm]': 'Position_x',
    'Position Y [nm]': 'Position_y',
    'Position Z [nm]': 'Position_z',
    'Precision [nm]': 'Precision',
    'Number Photons': 'Intensity',
    'Background variance': 'Local_background',
    'Chi square': 'Chi_square',
    'PSF half width [nm]': 'Psf_half_width',
    'Channel': 'Channel',
    'Z Slice': 'Slice_z'
}

#: Mapping column names in Thunderstorm files to LocData property keys
THUNDERSTORM_KEYS = {
    'id': 'Original_index',
    'frame': 'Frame',
    'x [nm]': 'Position_x',
    'y [nm]': 'Position_y',
    'z [nm]': 'Position_z',
    'uncertainty_xy [nm]': 'Uncertainty_x',
    'uncertainty_z [nm]': 'Uncertainty_z',
    'intensity [photon]': 'Intensity',
    'offset [photon]': 'Local_background',
    'chi2': 'Chi_square',
    'sigma1 [nm]': 'Psf_sigma_x',
    'sigma2 [nm]': 'Psf_sigma_y',
    'sigma [nm]': 'Psf_sigma_x'
}
# todo: uncertainty_xy and sigma [nm] are mapped to Uncertainty_x and Psf_sigma_x. Possible conflict?
# todo: map "bkgstd [photon]", "uncertainty [nm]" to something usefull

#: The number of cores that are used in parallel for some algorithms.
#: Following the scikit convention: n_jobs is the number of parallel jobs to run.
#: If -1, then the number of jobs is set to the number of CPU cores.
N_JOBS = 1
