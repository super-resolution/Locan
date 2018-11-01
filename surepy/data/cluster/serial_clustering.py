"""
This module provides methods to run and analyse a series of clustering processes to identify optimal parameters.
"""
import sys
import itertools
from inspect import signature

from surepy import LocData


def serial_clustering(locdata, algorithm, parameter_lists, **kwargs):
    """
    Run and analyse a series of clustering processes to identify optimal parameters.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    algorithm : callable
        The surepy clustering algorithm to use on locdata.
    parameter_lists : dict
        A dictionary with all parameter lists that are to be iterated. The keys should be identical to parameter names
        of the used algorithm.
    kwargs : dict
        Optional keyword arguments that are passed to the algorithm.

    Returns
    -------
    Tuple of Locdata
        The first element is a LocData object with a selection of all localizations that are defined as noise.
        If noise is false this element will be None.
        The second element is a new LocData instance assembling all generated selections (i.e. localization cluster).

    """
    parameter = locals()

    keys = parameter_lists.keys()
    products = list(itertools.product(*parameter_lists.values()))
    products_transposed = zip(*products)
    dictionary = {key: values for key, values in zip(keys, products_transposed)}
    results = []
    for l in products:
        iterated_arguments = {k: v for k, v in zip(keys, l)}
        results.append(algorithm(locdata, **iterated_arguments, **kwargs))

    if signature(algorithm).parameters['noise'].default or\
            ('noise' in kwargs and kwargs['noise']):

        noise_locdata = [res[0] for res in results]
        noise_collection = LocData.from_collection(*noise_locdata)
        noise_collection.dataframe = noise_collection.dataframe.assign(**dictionary)
        collection_locdata = [res[1] for res in results]
        collection = LocData.from_collection(*collection_locdata)
        collection.dataframe = collection.dataframe.assign(**dictionary)

        # metadata for noise_collection
        del noise_collection.meta.history[:]
        noise_collection.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    else:
        noise_collection = None
        collection = LocData.from_collection(*results)
        collection.dataframe = collection.dataframe.assign(**dictionary)

    # metadata for collection
    del collection.meta.history[:]
    collection.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return noise_collection, collection

