"""
Standard analysis procedures.

This module contains classes and functions for carrying out standardized
analysis procedures on localization data.
All functions typically take LocData objects as input and provide an
analysis class with derived analysis results
and standard functions for presentation.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   accumulation_analysis
   blinking
   cbc
   convex_hull_expectation
   drift
   grouped_property_expectation
   localization_precision
   localization_property
   localization_property_2d
   localization_property_correlations
   localizations_per_frame
   nearest_neighbor
   pipeline
   position_variance_expectation
   ripley
   subpixel_bias
   uncertainty
"""
from __future__ import annotations

from importlib import import_module

from locan.analysis.accumulation_analysis import *
from locan.analysis.blinking import *
from locan.analysis.cbc import *
from locan.analysis.convex_hull_expectation import *
from locan.analysis.drift import *
from locan.analysis.grouped_property_expectation import *
from locan.analysis.localization_precision import *
from locan.analysis.localization_property import *
from locan.analysis.localization_property_2d import *
from locan.analysis.localization_property_correlations import *
from locan.analysis.localizations_per_frame import *
from locan.analysis.nearest_neighbor import *
from locan.analysis.pipeline import *
from locan.analysis.position_variance_expectation import *
from locan.analysis.ripley import *
from locan.analysis.subpixel_bias import *
from locan.analysis.uncertainty import *

submodules: list[str] = [
    "accumulation_analysis",
    "blinking",
    "cbc",
    "convex_hull_expectation",
    "drift",
    "grouped_property_expectation",
    "localization_precision",
    "localization_property",
    "localization_property_2d",
    "localization_property_correlations",
    "localizations_per_frame",
    "nearest_neighbor",
    "pipeline",
    "position_variance_expectation",
    "ripley",
    "subpixel_bias",
    "uncertainty",
]

__all__: list[str] = []

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="locan.analysis")
    __all__.extend(module_.__all__)
