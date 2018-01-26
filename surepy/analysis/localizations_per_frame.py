"""
This module provides methods for analysis.
"""
from surepy.analysis.analysis import Analysis


class Localizations_per_frame(Analysis):
    """
    Compute the number of localizations in each frame.

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

    def __init__(self, locdata, meta=None, normalization=None):
        super().__init__(locdata, meta=meta, normalization=normalization)


    def _compute_results(self, locdata, normalization=None):

        if normalization is None:
            normalization_factor = 1
        elif isinstance(normalization, str):
            normalization_factor = locdata.properties[normalization]
        elif isinstance(normalization, (int, float)):
            normalization_factor = normalization
        else:
            raise TypeError('normalization should be None, a number or a valid property name.')

        return locdata.data.groupby('Frame').size() / normalization_factor


    def hist(self, ax, bins='auto'):
        """ Provide histogram as matplotlib axes object showing hist(results). """
        ax.hist(self.results.values, bins=bins, normed=True, log=False)
        ax.set(title = 'Localizations per Frame',
               xlabel = 'number of localizations',
               ylabel = 'PDF'
               )

    def plot(self, ax, window=1):
        """ Provide plot as matplotlib axes object showing the running average of results over window size. """
        self.results.rolling(window=window, center=True).mean().plot(ax=ax)
        ax.set(title = 'Localizations per Frame',
               xlabel = 'Frame',
               ylabel = 'number of localizations'
               )
        ax.text(0.1,0.9,
                "window = " + str(window),
                transform = ax.transAxes
                )

    def save_as_yaml(self):
        """ Save results in a YAML format, that can e.g. serve as Origin import."""
        raise NotImplementedError