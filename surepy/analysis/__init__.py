"""
Standard analysis procedures.

This module contains classes and functions for carrying out standardized analysis procedures on localization data.
All functions typically take LocData objects as input and provide an analysis class with derived analysis results
and standard functions for presentation.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   localization_precision
   localization_property
   localizations_per_frame
   nearest_neighbor
   pipeline
   ripley
   uncertainty
   cbc
   accumulation_analysis
   blinking

"""

from surepy.analysis.analysis_example import AnalysisExampleAlgorithm_1, AnalysisExampleAlgorithm_2

from surepy.analysis.localization_precision import LocalizationPrecision
from surepy.analysis.localization_property import LocalizationProperty
from surepy.analysis.localizations_per_frame import LocalizationsPerFrame
from surepy.analysis.nearest_neighbor import NearestNeighborDistances
from surepy.analysis.pipeline import Pipeline
from surepy.analysis.ripley import RipleysKFunction, RipleysLFunction, RipleysHFunction
from surepy.analysis.uncertainty import LocalizationUncertaintyFromIntensity
from surepy.analysis.cbc import CoordinateBasedColocalization
from surepy.analysis.accumulation_analysis import AccumulationClusterCheck
from surepy.analysis.blinking import BlinkStatistics
