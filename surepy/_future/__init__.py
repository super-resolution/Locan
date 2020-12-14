"""
Unstable functions and classes.

The module members are under development and require additional testing.
The members will be incorporated into surepy at a later stage.
"""


def pair_files(files, source, target):
    """
    Pair file names depending on corresponding parts in the file name.

    A source string is replaced by the target string and the resulting pairs are returned as dictionary.

    Parameters
    ----------
    files : list of str or os.PathLikes
        List of file names containing all elements.
    source : str
        The string identifying source names.
    target : str
        The string being inserted to identify target names.

    Returns
    -------
    tuple
        Tuple with source and target file names or os.PathLike.
    """
    pairs = [(file, file.with_name(file.name.replace(source, target))) for file in files if source in str(file)]
    return pairs
