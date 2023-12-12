"""

Utility methods for clustering locdata.

"""
from __future__ import annotations

import itertools
import sys
from collections.abc import Callable
from typing import Any

from locan.data.locdata import LocData

__all__: list[str] = ["serial_clustering"]


def serial_clustering(
    locdata: LocData,
    algorithm: Callable[..., Any],
    parameter_lists: dict[str, Any],
    **kwargs: Any,
) -> tuple[LocData, ...]:
    """
    Run and analyse a series of clustering processes to identify optimal
    parameters.

    Parameters
    ----------
    locdata
        Localization data.
    algorithm
        The locan clustering algorithm to use on locdata.
    parameter_lists
        A dictionary with all parameter lists that are to be iterated.
        The keys should be identical to parameter names of the used algorithm.
    kwargs
        Optional keyword arguments that are passed to the algorithm.

    Returns
    -------
    tuple[LocData, ...]
        The first element is a LocData object with a selection of all
        localizations that are defined as noise.
        If noise is false this element will be None.
        The second element is a new LocData instance assembling all generated
        selections (i.e. localization cluster).
    """
    parameter = locals()

    keys = parameter_lists.keys()
    products = list(itertools.product(*parameter_lists.values()))
    products_transposed = zip(*products)
    dictionary = {key: values for key, values in zip(keys, products_transposed)}
    results = []
    for product in products:
        iterated_arguments = {k: v for k, v in zip(keys, product)}
        results.append(algorithm(locdata, **iterated_arguments, **kwargs))

    noise_locdata = [result[0] for result in results]
    noise_collection = LocData.from_collection(noise_locdata)
    noise_collection.dataframe = noise_collection.dataframe.assign(**dictionary)
    collection_locdata = [result[1] for result in results]
    collection = LocData.from_collection(collection_locdata)
    collection.dataframe = collection.dataframe.assign(**dictionary)

    # metadata for noise_collection
    del noise_collection.meta.history[:]
    noise_collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    # metadata for collection
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return noise_collection, collection
