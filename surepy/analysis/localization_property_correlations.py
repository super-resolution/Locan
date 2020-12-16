"""
Analyze cross dependencies between localization properties.

Analyze cross dependencies as indicated by the correlation coefficients between any two localization properties.
"""
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

from surepy.render.render2d import render_2d_mpl
from surepy.analysis.analysis_base import _Analysis, _list_parameters
from surepy.render.render2d import histogram
from surepy.constants import COLORMAP_DIVERGING


__all__ = ['LocalizationPropertyCorrelations']


##### The algorithms

def _localization_property_correlations(locdata, loc_properties=None):
    if loc_properties is None:
        results = locdata.data.corr()
    else:
        results = locdata.data[loc_properties].corr()
    return results

##### The specific analysis classes

class LocalizationPropertyCorrelations(_Analysis):
    """
    Compute and analyze correlation coefficients between any two localization properties.

    Parameters
    ----------
    meta : surepy.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    loc_properties : list, None
        Localization properties to be analyzed. If None all are used.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : surepy.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        The correlation coefficients..
    """
    def __init__(self, meta=None, loc_properties=None):
        super().__init__(meta=meta, loc_properties=loc_properties)
        self.results = None

    def compute(self, locdata=None):
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
        self.results = _localization_property_correlations(locdata=locdata, **self.parameter)
        return self

    def report(self):
        print('Fit results for:\n')
        print(self.results.model_result.fit_report(min_correl=0.25))
        # print(self.results.fit_results.best_values)


    def plot(self, ax=None, cbar=True, colorbar_kws=None, **kwargs):
        """
        Provide heatmap of all correlation values as matplotlib.axes.Axes object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to show the image
        cbar : bool
            If true draw a colorbar.
        colorbar_kws : dict
            Keyword arguments for `matplotlib.pyplot.colorbar`.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to `matplotlib.pyplot.imshow()`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        im = ax.imshow(self.results, **dict(dict(vmin=-1, vmax=1, cmap=COLORMAP_DIVERGING), **kwargs))
        columns = self.results.columns

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(columns)))
        ax.set_yticks(np.arange(len(columns)))

        # ensure correct scaling
        ax.set_xticks(np.arange(len(columns) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(columns) + 1) - .5, minor=True)
        ax.tick_params(which="minor", bottom=False, left=False)

        # ... and label them with the respective list entries
        ax.set_xticklabels(columns)
        ax.set_yticklabels(columns)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(columns)):
            for j in range(len(columns)):
                text = ax.text(j, i, round(self.results.values[i, j], 2),
                               ha="center", va="center", color="w")

        ax.set_title("Localization Property Correlations")

        # Create colorbar
        if cbar:
            if colorbar_kws is None:
                cbar = ax.figure.colorbar(im)
            else:
                cbar = ax.figure.colorbar(im, **colorbar_kws)
            # cbar.ax.set_ylabel('correlation', rotation=-90, va="bottom")

        ax.figure.tight_layout()
        return ax
