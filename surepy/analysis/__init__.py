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

# from surepy.analysis.analysis_example import AnalysisExampleAlgorithm_1, AnalysisExampleAlgorithm_2
from surepy.analysis.localization_precision import *
from surepy.analysis.localization_property import *
from surepy.analysis.localizations_per_frame import *
from surepy.analysis.nearest_neighbor import *
from surepy.analysis.pipeline import *
from surepy.analysis.ripley import *
from surepy.analysis.uncertainty import *
from surepy.analysis.cbc import *
from surepy.analysis.accumulation_analysis import *
from surepy.analysis.blinking import *

__all__ = []
__all__.extend(localization_precision.__all__)
__all__.extend(localization_property.__all__)
__all__.extend(localizations_per_frame.__all__)
__all__.extend(nearest_neighbor.__all__)
__all__.extend(pipeline.__all__)
__all__.extend(ripley.__all__)
__all__.extend(uncertainty.__all__)
__all__.extend(cbc.__all__)
__all__.extend(accumulation_analysis.__all__)
__all__.extend(blinking.__all__)
