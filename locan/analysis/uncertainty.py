"""
Compute localization uncertainty.

Localization uncertainty depends on a number of experimental factors including camera and photophysical characteristics
as outlined in [1]_ [2]_. We provide functions to compute an uncertainty estimate from available localization properties.

References
----------
.. [1] K.I. Mortensen, L. S. Churchman, J. A. Spudich, H. Flyvbjerg, Nat. Methods 7 (2010): 377–384.
.. [2] Rieger B., Stallinga S., The lateral and axial localization uncertainty in super-resolution light microscopy.
   Chemphyschem 17;15(4), 2014:664-70. doi: 10.1002/cphc.201300711

"""
import logging

import numpy as np
import pandas as pd

from locan.analysis.analysis_base import _Analysis

__all__ = ["LocalizationUncertaintyFromIntensity"]

logger = logging.getLogger(__name__)


# The algorithms


def _localization_uncertainty_from_intensity(locdata):

    results = {}
    for v in ["x", "y", "z"]:
        if (
            "position_" + v in locdata.data.keys()
            and "intensity" in locdata.data.keys()
        ):
            if "psf_sigma_" + v in locdata.data.keys():
                results.update(
                    {
                        "uncertainty_"
                        + v: locdata.data["psf_sigma_" + v]
                        / np.sqrt(locdata.data["intensity"])
                    }
                )
            else:
                results.update(
                    {"uncertainty_" + v: 1 / np.sqrt(locdata.data["intensity"])}
                )
        else:
            pass

    return pd.DataFrame(results)


# The specific analysis classes


class LocalizationUncertaintyFromIntensity(_Analysis):
    """
    Compute the localization uncertainty for each localization's spatial coordinate in locdata.

    Uncertainty is computed as Psf_sigma / Sqrt(Intensity) for each spatial dimension.
    If Psf_sigma is not available Uncertainty is 1 / Sqrt(Intensity).

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        The number of localizations per frame or
        the number of localizations per frame normalized to region_measure(hull).
    """

    count = 0

    def __init__(self, meta=None):
        super().__init__(meta=meta)
        self.results = None

    def compute(self, locdata):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.results = _localization_uncertainty_from_intensity(locdata=locdata)
        return self
