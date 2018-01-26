import numpy as np
import pandas as pd
from surepy.analysis.analysis import Analysis


class Localization_uncertainty_from_intensity(Analysis):
    """
    Compute the localization uncertainty for each localization's spatial coordinate in locdata.

    Uncertainty is computed as Psf_sigma / Sqrt(Intensity) for each spatial dimension.
    If Psf_sigma is not available Uncertainty is 1 / Sqrt(Intensity).

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    normalization : str or int or float or None
        The data is normalized to a property of locdata (specified by property name) or to a number. It is not
        normalized for normalization being None (default: None).

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    results : pandas data frame
        The number of localizations per frame or
        the number of localizations per frame normalized to region_measure(hull).
    meta : dict
        meta data
    """
    count = 0

    def __init__(self, locdata, meta=None):
        super().__init__(locdata, meta=meta)


    def _compute_results(self, locdata, dummy=None):

        results = {}
        for v in ['x', 'y', 'z']:
            if 'Position_' + v in locdata.data.keys() and 'Intensity' in locdata.data.keys():
                if 'Psf_sigma_' + v in locdata.data.keys():
                    results.update(
                        {'Uncertainty_' + v: locdata.data['Psf_sigma_' + v] / np.sqrt(locdata.data['Intensity'])}
                    )
                else:
                    results.update({'Uncertainty_' + v: 1 / np.sqrt(locdata.data['Intensity'])})
            else:
                pass  # todo: what sort of exception or warning would be appropriate?

        return pd.DataFrame(results)

        # dict = {}
        # for v in ['x', 'y', 'z']:
        #     if 'Position_' + v in locdata.data.keys() and 'Intensity' in locdata.data.keys():
        #         if 'Psf_sigma_' + v in locdata.data.keys():
        #             dict.update(
        #                 {'Uncertainty_' + v: locdata.data['Psf_sigma_' + v] / np.sqrt(locdata.data['Intensity'])})
        #         else:
        #             dict.update({'Uncertainty_' + v: 1 / np.sqrt(locdata.data['Intensity'])})
        #     else:
        #         pass  # todo: what sort of exception or warning would be appropriate?
        #
        # return dict

