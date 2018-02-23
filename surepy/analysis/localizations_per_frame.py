"""
This module provides methods for analysis.
"""
import matplotlib.pyplot as plt

from surepy.analysis.analysis import Analysis


def localizations_per_frame(data, normalization=None):
    if normalization is None:
        normalization_factor = 1
    elif isinstance(normalization, str):
        normalization_factor = data.properties[normalization]
    elif isinstance(normalization, (int, float)):
        normalization_factor = normalization
    else:
        raise TypeError('normalization should be None, a number or a valid property name.')

    return data.data.groupby('Frame').size() / normalization_factor


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

    def __init__(self, locdata, algorithm=localizations_per_frame, meta=None, **kwargs):
        super().__init__()
        self.locdata = locdata
        self.algorithm = algorithm
        self.parameter = kwargs
        self.meta = self._init_meta(meta=meta)

        self.results = None
        self.secondary_results = None

    def compute(self):
        data = self.locdata  # take certain elements from locdata
        self.results = self.algorithm(data=data, **self.parameter)  # some complicated algorithm
        return None


    def plot(self, ax=None, show=True, window=1):
        '''
        Provide plot as matplotlib axes object showing the running average of results over window size.
        '''
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        self.results.rolling(window=window, center=True).mean().plot(ax=ax)

        ax.set(title = 'Localizations per Frame',
               xlabel = 'Frame',
               ylabel = 'number of localizations'
               )

        ax.text(0.1,0.9,
                "window = " + str(window),
                transform = ax.transAxes
                )

        # show figure
        if show:  # this part is needed if anyone wants to modify the figure
            plt.show()

        return None


    def hist(self, ax=None, show=True, bins='auto'):
        '''
        Provide histogram as matplotlib axes object showing hist(results).
        '''
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.hist(self.results.values, bins=bins, normed=True, log=False)
        ax.set(title = 'Localizations per Frame',
               xlabel = 'number of localizations',
               ylabel = 'PDF'
               )

        # show figure
        if show:  # this part is needed if anyone wants to modify the figure
            plt.show()

        return None