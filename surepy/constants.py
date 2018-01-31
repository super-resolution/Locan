"""
Constants to be used throughout the project
"""
import os


# set root directory for path operations

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Keys (i.e. names) for the most common LocData properties

PROPERTY_KEYS = ['Position_x', 'Position_y', 'Position_z', 'Frame', 'Intensity', 'Local_background', 'Chi_square',
                 'Psf_sigma_x', 'Psf_sigma_y', 'Psf_sigma_z', 'Uncertainty_x', 'Uncertainty_y', 'Uncertainty_z',
                 'Channel', 'Index', 'Cluster_label', 'Two_kernel_improvement']

# Keys (i.e. names) for the most common hulls

HULL_KEYS = {'bounding_box', 'convex_hull', 'oriented_bounding_box', 'alpha_shape'}


# Mapping column names in RapidSTORM files to LocData property keys

RAPIDSTORM_KEYS = {
    'Position-0-0': 'Position_x',
    'Position-1-0': 'Position_y',
    'ImageNumber-0-0': 'Frame',
    'Amplitude-0-0': 'Intensity',
    'FitResidues-0-0': 'Chi_square',
    'LocalBackground-0-0': 'Local_background',
    'TwoKernelImprovement-0-0': 'Two_kernel_improvement'
}


# Mapping column names in Zeiss Elyra files to LocData property keys

ELYRA_KEYS = {
    'Position-0-0': 'Position_x',
    'Position-1-0': 'Position_y',
    'ImageNumber-0-0': 'Frame',
    'Amplitude-0-0': 'Intensity',
    'FitResidues-0-0': 'Chi_square',
    'LocalBackground-0-0': 'Local_background',
    'TwoKernelImprovement-0-0': 'Two_kernel_improvement'
}


# Standard keys for metadata in LocData

META_DICT = {
    'Identifier':'',
    'Comment': '',
    'Production date': '',
    'Modification date': '', # last modification
    'State': '',  # [raw, modified]
    'Source': '', # [experiment, simulation, design, import]
    'Experimental setup': {},
    'Experimental sample': {},
    'File type': '', # [rapidStorm, Elyra]
    'File path': '',
    'Simulation': {'Description': '', 'Method': '', 'Parameter': {}},
    'Number of elements': '',
    'Number of frames': '',
    'Units': {},
    'History': [{'Method:': '', 'Parameter': {}}], # list of modifications with function name and parameter for applied method
    'Ancestor id': '',
    }

