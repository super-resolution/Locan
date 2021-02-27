"""
Utility functions to deal with examplary datasets.

The data is located in a separate directory `Surepy_datasets`.
The path to the datasets directory can be set by the `surepy.constants.DATASETS_DIR` variable.

Available datasets:

1) npc: localization data of nuclear pore complexes imaged by dSTORM.
2) tubulin: localization data of microtubules in COS-7 cells imaged by dSTORM.
"""
from pathlib import Path

from surepy.io.io_locdata import load_rapidSTORM_file
from surepy.constants import DATASETS_DIR


__all__ = ['load_npc', 'load_tubulin']


def load_npc(**kwargs):
    """
    Locdata representing nuclear pore complexes.

    The data was generated by dSTORM [1]_.
    It shows the gp210 protein of the nuclear pore complex labeled with AlexaFluor647.

    References
    ----------
    .. [1] Löschberger A, van de Linde S, Dabauvalle MC, Rieger B, Heilemann M, Krohne G, Sauer M.,
       Super-resolution imaging visualizes the eightfold symmetry of gp210 proteins around the nuclear pore complex
       and resolves the central channel with nanometer resolution.
       J Cell Sci. 2012, 125:570-5, doi: 10.1242/jcs.098822.

    Parameters
    ----------
    kwargs : dict
        Parameters passed to `surepy.load_rapidSTORM_file()`.

    Returns
    -------
    LocData
    """
    path = Path(DATASETS_DIR) / 'smlm_data/npc_gp210.txt'
    locdata = load_rapidSTORM_file(path, **kwargs)
    return locdata


def load_tubulin(**kwargs):
    """
    Locdata representing microtubules.

    The data was generated by dSTORM [1]_.
    It shows alpha-tubulin as part of microtubules within COS-7 cells.
    Tubulin was targeted by primary IgG-antibodies labeled with AlexaFluor647 (2.1 degree of labeling)
    and recorded over 75_000 frames.

    References
    ----------
    .. [1] Dominic A. Helmerich, Gerti Beliu, and Markus Sauer,
       Multiple-Labeled Antibodies Behave Like Single Emitters in Photoswitching Buffer
       ACS Nano 2020, 14, 10, 12629–12641, DOI: 10.1021/acsnano.0c06099

    Parameters
    ----------
    kwargs : dict
        Parameters passed to `surepy.load_rapidSTORM_file()`.

    Returns
    -------
    LocData
    """
    path = Path(DATASETS_DIR) / 'smlm_data/tubulin_cos7.txt'
    locdata = load_rapidSTORM_file(path, **kwargs)
    return locdata
