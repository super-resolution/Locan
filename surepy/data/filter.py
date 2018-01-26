'''

Methods for filtering LocData objects.

'''

from surepy import LocData


def select_by_condition(locdata, condition, **kwargs):
    """
    Select by specifying conditions on data properties.

    Parameters
    ----------
    locdata : LocData
        Specifying the localization data from which to select localization data.
    condition : string
        Conditions as input in select method.
        More precise: query specifications to be used with pandas query.

    Returns
    -------
    LocData
        a new instance of LocData referring to the specified dataset.
    """

    # select
    new_indices = locdata.data.query(condition).index.values.tolist()

    # instantiate
    new_locdata = LocData.from_selection(locdata=locdata, indices=new_indices, **kwargs)

    # metadata
    new_locdata.meta['History'].append({'Method:': 'select_by_condition', 'Parameter': [locdata, condition]})

    return new_locdata


def select_by_region(selection, region, **kwargs):
    """
    Select localizations within specified rectangle, circle, polygon or 3D equivalents.

    Parameters
    ----------
    selection : Selection
        specifying the localization data from which to select localization data.
    region :
        region of interest.

    Returns
    -------
    Selection
        a new instance of Selection referring to the specified dataset.
    """
    raise NotImplementedError


def select_by_image_mask(selection, mask, pixel_size, **kwargs):
    """
    Select by masking using a binary image(e.g. generated by thresholding a transmitted-light microscopy image.

    Parameters
    ----------
    selection : Selection
        specifying the localization data from which to select localization data.
    mask :
        binary image.
    pixel_size : tuple(float)
        pixel sizes for each dimension in units of localization coordinates.

    Returns
    -------
    Selection
        a new instance of Selection referring to the specified dataset.
    """
    raise NotImplementedError



def exclude_sparce_points(selection, threshold_density, **kwargs):
    """
    Exclude localizations by thresholding a local density.

    Parameters
    ----------
    selection : Selection
        specifying the localization data on which to perform the manipulation.
    threshold_density : float
        all points with a local density below the threshold will be excluded.

    Returns
    -------
    Selection
        a new instance of Selection referring to the same Dataset as the input Selection.
    """
    raise NotImplementedError
