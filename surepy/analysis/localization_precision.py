"""
This module provides methods for analysis.
"""
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from scipy import stats

from surepy.analysis.analysis_tools import _init_meta, _update_meta


#### The algorithms

def _localization_precision(locdata, radius=50):
    # group localizations
    grouped = locdata.data.groupby('Frame')

    # find nearest neighbors
    min = locdata.data['Frame'].unique().min()
    max = locdata.data['Frame'].unique().max()

    results = pd.DataFrame()

    for i in range(min, max - 1):
        points = grouped.get_group(i)[locdata.coordinate_labels]
        other_points = grouped.get_group(i + 1)[locdata.coordinate_labels]

        # print(points)

        nn = NearestNeighbors(radius=radius, metric='euclidean').fit(other_points)
        distances, indices = nn.radius_neighbors(points)

        if len(distances):
            for n, (dists, inds) in enumerate(zip(distances, indices)):
                if len(dists):
                    min_distance = np.amin(dists)
                    min_position = np.argmin(dists)
                    min_index = inds[min_position]
                    difference = points.iloc[n] - other_points.iloc[min_index]

                    df = difference.to_frame().T
                    df = df.rename(columns={'Position_x': 'Position_delta_x',
                                            'Position_y': 'Position_delta_y',
                                            'Position_z': 'Position_delta_z'})
                    df = df.assign(Position_distance=min_distance)
                    df = df.assign(Frame=i)
                    results = results.append(df)

    results.reset_index(inplace=True, drop=True)
    return (results)


# The base analysis class

class _Localization_precision():
    """
    Compute the localization precision from consecutive nearby localizations.

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

    def plot(self, ax, property=None, window=1):
        return plot(self, ax, property, window)

    def hist(self, ax, property='Position_distance', bins='auto', fit=True):
        return hist(self, ax, property, bins, fit)


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


class Localization_precision(_Localization_precision):
    """
    Compute the localization precision from consecutive nearby localizations.


    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    radius : int or float
        Search radius for nearest-neighbor searches.

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

    def __init__(self, locdata=None, meta=None, radius=50):
        super().__init__(locdata=locdata, meta=meta, radius=radius)

    def compute(self):
        data = self.locdata
        self.results = _localization_precision(locdata=data, **self.parameter)
        return self



#### Interface functions


def hist(self, ax, property='Position_distance', bins='auto', fit=True):
    """ Provide histogram as matplotlib axes object showing hist(results). """
    ax.hist(self.results[property].values, bins=bins, normed=True, log=False)
    ax.set(title = 'Localizations Precision',
           xlabel = property,
           ylabel = 'PDF'
           )

    # fit distributions:
    if fit:
        if 'Position_delta_' in property:
            # MLE fit of distribution on data
            loc, scale = stats.norm.fit(self.results[property].values)

            # plot
            x_values = np.linspace(stats.norm.ppf(0.01, loc=loc, scale=scale), stats.norm.ppf(0.99, loc=loc, scale=scale), 100)
            ax.plot(x_values, stats.norm.pdf(x_values, loc=loc, scale=scale), 'r-', lw = 3, alpha = 0.6, label = 'norm pdf')
            ax.text(0.1, 0.9,
                    'center: ' + str(loc) + '\n' + 'sigma: ' + str(scale),
                    transform=ax.transAxes
                    )

            attribute_center = property + '_center'
            attribute_sigma = property + '_center'
            self.attribute_center = loc
            self.attribute_sigma = scale

        elif property == 'Position_distance':
            # define continous distribution class
            class Pairwise_distance_distribution_2d(stats.rv_continuous):

                def _pdf(self, x, sigma):
                    return x / (2 * sigma ** 2) * np.exp(- x ** 2 / (4 * sigma ** 2))

            pairwise_2D = Pairwise_distance_distribution_2d(name='pairwise', a=0.)

            # MLE fit of distribution on data with fixed loc and scale
            sigma, loc, scale = pairwise_2D.fit(self.results[property].values, floc=0, fscale=1)

            # plot
            x_values =  np.linspace(pairwise_2D.ppf(0.01, sigma=sigma), pairwise_2D.ppf(0.99, sigma=sigma), 100)
            ax.plot(x_values, pairwise_2D.pdf(x_values, sigma=sigma), 'r-', lw=3, alpha=0.6)
            ax.text(0.1, 0.9,
                    'sigma: ' + str(sigma),
                    transform=ax.transAxes
                    )

            self.Position_distance_sigma = sigma



def plot(self, ax, property=None, window=1):
    """ Provide plot as matplotlib axes object showing the running average of results over window size. """
    self.results.rolling(window=window, center=True).mean().plot(ax=ax, x='Frame', y=property, legend=False)
    ax.set(title = 'Localizations Precision',
           xlabel = 'Frame',
           ylabel = property
           )
    ax.text(0.1,0.9,
            "window = " + str(window),
            transform = ax.transAxes
            )
