"""
Constants to be used throughout the project
"""
import os


# set root directory for path operations

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Keys (i.e. names) for the most common LocData properties

PROPERTY_KEYS = ['Index', 'Position_x', 'Position_y', 'Position_z', 'Frame', 'Intensity', 'Local_background', 'Chi_square',
                 'Psf_sigma_x', 'Psf_sigma_y', 'Psf_sigma_z', 'Uncertainty_x', 'Uncertainty_y', 'Uncertainty_z',
                 'Channel', 'Index', 'Cluster_label', 'Two_kernel_improvement']

# Keys (i.e. names) for the most common hulls

HULL_KEYS = {'bounding_box', 'convex_hull', 'oriented_bounding_box', 'alpha_shape'}


# Mapping column names in RapidSTORM files to LocData property keys

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


# Mapping column names in Zeiss Elyra files to LocData property keys
# todo: add Elyra keys to documentation.

ELYRA_KEYS = {
    'Index': 'Index',
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


