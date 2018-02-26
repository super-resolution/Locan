import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from surepy.analysis.analysis_tools import _init_meta, _update_meta


# The algorithms

def _localization_uncertainty_from_intensity(locdata):

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
            pass

    return pd.DataFrame(results)


# The base analysis class

class _Localization_uncertainty():
    """
    The base class for specialized analysis classes to be used on LocData objects.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    kwargs :
        Parameter that are passed to the algorithm.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    locdata : LocData object
        Localization data.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.
    """
    count = 0

    def __init__(self, locdata, meta, **kwargs):
        self.__class__.count += 1

        self.locdata = locdata
        self.parameter = kwargs
        self.meta = _init_meta(self)
        self.meta = _update_meta(self, meta)
        self.results = None


    def __del__(self):
        """ updating the counter upon deletion of class instance. """
        self.__class__.count -= 1

    def __str__(self):
        """ Return results in a printable format."""
        return str(self.results)

    def save_results(self, path):
        return save_results(self, path)



    def plot(self):
        raise NotImplementedError

    def hist(self):
        raise NotImplementedError

    def compute(self):
        """ Apply analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError

    def save(self, path):
        """ Save Analysis object."""
        raise NotImplementedError

    def load(self, path):
        """ Load Analysis object."""
        raise NotImplementedError

    def report(self, ax):
        """ Show a report about analysis results."""
        raise NotImplementedError



# The specific analysis classes

class Localization_uncertainty_from_intensity(_Localization_uncertainty):
    """
    Compute the localization uncertainty for each localization's spatial coordinate in locdata.

    Uncertainty is computed as Psf_sigma / Sqrt(Intensity) for each spatial dimension.
    If Psf_sigma is not available Uncertainty is 1 / Sqrt(Intensity).

    Parameters
    ----------
    locdata : LocData object
        Localization data.

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

    def __init__(self, locdata=None, meta=None):
        super().__init__(locdata=locdata, meta=meta)

    def compute(self):
        data = self.locdata
        self.results = _localization_uncertainty_from_intensity(locdata=data)
        return self
