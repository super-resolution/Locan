"""
This module provides a template for an analysis class.
"""


META_DICT = {
    'Analysis': {
        'Method': '', # analysis class or method
        'Parameter': {} # parameter for analysis function
    },
    'Data': {}, # input locdata.meta
    'Results type': '', # results type
    'Comments': '' # user comments
}

class Analysis():
    """
    An abstract class for analysis methods to be used on LocData objects.

    The analysis code should go in _compute_results() which is automatically called upon instantiation.

    Parameters
    ----------
    locdata : LocData
        Input data.
    kwargs : kwarg
        Parameters for the analysis routine.

    Attributes
    -----------
    count : int (class attribute)
        A counter for counting Analysis instantiations.
    locdata : LocData
        reference to the LocData object specified as input data.
    results : pandas data frame or array or array of arrays or None
        The numeric results as derived from the analysis method.
    parameter : dict
        Current parameters for the analysis routine.
    meta : dict
        meta data
    """
    count=0

    def __init__(self, locdata, meta=None, **kwargs):
        """ Provide default atributes."""
        Analysis.count += 1
        self.__class__.count += 1

        self.parameter = kwargs
        self.locdata = locdata
        self.results = self._compute_results(locdata, **kwargs)

        self.meta = {
                'Analysis': {
                    'Method': self.__class__,
                    'Parameter': self.parameter
                    },
                'Data': locdata.meta
                }
        if meta is not None:
            self.meta.update(meta)


    def __del__(self):
        """ updating the counter upon deletion of class instance. """
        Analysis.count -= 1
        self.__class__.count -= 1

    def __str__(self):
        """ Return results in a printable format."""
        return str(self.results)

    def _compute_results(self, locdata, **kwargs):
        """ Apply analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError

    def save(self):
        # todo: an appropriate file format needs to be identified.
        """ Save results."""
        raise NotImplementedError

    def save_as_txt(self):
        """ Save results in a text format, that can e.g. serve as Origin import."""
        raise NotImplementedError

    def save_as_yaml(self):
        """ Save results in a text format, that can e.g. serve as Origin import."""
        raise NotImplementedError

    def load(self, results):
        """ Load results."""
        raise NotImplementedError

    def plot(self, ax=None):
        """ Provide an axes instance with plot of results."""
        if ax is None:
            ax = plt.gca()
        # ax.plot(results)
        raise NotImplementedError

    def hist(self, ax):
        """ Provide an axes instance with histogram of results."""
        # ax.hist(results)
        raise NotImplementedError
