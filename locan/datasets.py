"""
Utility functions to deal with exemplary datasets.

The data is located in a separate repository `https://github.com/super-resolution/LocanDatasets`.

When calling a function the datasets are expected to reside in a directory specified by the
`locan.constants.DATASETS_DIR` variable.
If the directory does not exist the exemplary files are downloaded from GitHub.
"""
from pathlib import Path

from locan.configuration import DATASETS_DIR
from locan.dependencies import HAS_DEPENDENCY, needs_package
from locan.locan_io.locdata.asdf_io import load_asdf_file

if HAS_DEPENDENCY["requests"]:
    import requests


__all__ = ["load_npc", "load_tubulin"]


@needs_package("requests")
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
        Parameters passed to `locan.load_asdf_file()`.

    Returns
    -------
    LocData
    """
    path = Path(DATASETS_DIR) / "npc_gp210.asdf"
    if not path.exists():
        DATASETS_DIR.mkdir(exist_ok=True)
        url = "https://raw.github.com/super-resolution/LocanDatasets/main/smlm_data/npc_gp210.asdf"
        response = requests.get(url)
        if response.status_code != requests.codes.ok:
            raise ConnectionError("response.status_code != requests.codes.ok")
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
    locdata = load_asdf_file(path, **kwargs)
    return locdata


@needs_package("requests")
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
        Parameters passed to `locan.load_rapidSTORM_file()`.

    Returns
    -------
    LocData
    """
    path = Path(DATASETS_DIR) / "tubulin_cos7.asdf"
    if not path.exists():
        DATASETS_DIR.mkdir(exist_ok=True)
        url = "https://raw.github.com/super-resolution/LocanDatasets/main/smlm_data/tubulin_cos7.asdf"
        response = requests.get(url)
        if response.status_code != requests.codes.ok:
            raise ConnectionError("response.status_code != requests.codes.ok")
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
    locdata = load_asdf_file(path, **kwargs)
    return locdata
