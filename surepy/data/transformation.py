def transform(selection, *args):
    """
    Transform coordinates e.g. for correcting chromatic aberrations.

    Parameters
    ----------
    selection : Selection
        specifying the localization data on which to perform the manipulation.
    args :
        transformation parameters

    Returns
    -------
    Selection
        a new instance of Selection referring to the same Dataset as the input Selection.
    """
    raise NotImplementedError

