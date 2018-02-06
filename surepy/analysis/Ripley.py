"""
This module provides methods for computing Ripley's k function.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from astropy.stats import RipleysKEstimator

from surepy.analysis.analysis import Analysis


class Ripleys_h_function(Analysis):
    """
    Compute Ripley's h function.

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

    def __init__(self, locdata, meta=None, radii=np.linspace(0, 100, 10)):
        super().__init__(locdata, meta=meta, radii=radii)


    def _compute_results(self, locdata, radii = np.linspace(0, 100, 10)):

        if locdata.coordinate_labels == {'Position_x', 'Position_y', 'Position_z'}:
            raise NotImplementedError('Ripley\'s k function is only implemented for 2D data.')

        x_min, y_min, x_max, y_max = [float(n) for n in locdata.bounding_box.hull.flatten()]
        area = float(locdata.properties['Region_measure_bb'])

        RKest = RipleysKEstimator(area, x_max, y_max, x_min, y_min)

        res_data = RKest.Hfunction(data=locdata.coordinates, radii=radii, mode='none')
        #res_csr = RKest.poisson(radii)

        # return pd.DataFrame({'radius': radii, 'Ripley_h_data':res_data, 'Ripley_h_csr':res_csr})
        return pd.DataFrame({'radius': radii, 'Ripley_h_data': res_data})



    def plot(self, ax):
        """ Provide plot as matplotlib axes object showing the running average of results over window size. """
        self.results.plot(x='radius', ax=ax)
        ax.set(title = 'Ripley\'s h function',
               xlabel = 'Radius',
               ylabel = 'Ripley\'s h function'
               )
        ax.text(0.1,0.9,
                "Maximum: " + 'not yet',
                transform = ax.transAxes
                )

    def save_as_yaml(self):
        """ Save results in a YAML format, that can e.g. serve as Origin import."""
        raise NotImplementedError