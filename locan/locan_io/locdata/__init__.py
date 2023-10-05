"""

File input/output for localization data.

There are functions for reading the following file structures
(with an indicator string in parenthesis - see also `locan.constants.FileType`):

* custom text file (CUSTOM)
* rapidSTORM file format (RAPIDSTORM) [1]_
* Elyra file format (ELYRA)
* Thunderstorm file format (THUNDERSTORM) [2]_
* asdf file format (ASDF) [3]_
* Nanoimager file format (NANOIMAGER)
* rapidSTORM track file format (RAPIDSTORMTRACK) [1]_
* smlm file format (SMLM) [4]_, [5]_
* decode file (DECODE) [6]_
* smap file format (SMAP) [7]_


References
----------
.. [1] Wolter S, Löschberger A, Holm T, Aufmkolk S, Dabauvalle MC, van de Linde S, Sauer M.,
   rapidSTORM: accurate, fast open-source software for localization microscopy.
   Nat Methods 9(11):1040-1, 2012, doi: 10.1038/nmeth.2224

.. [2] M. Ovesný, P. Křížek, J. Borkovec, Z. Švindrych, G. M. Hagen.
   ThunderSTORM: a comprehensive ImageJ plugin for PALM and STORM data analysis and super-resolution imaging.
   Bioinformatics 30(16):2389-2390, 2014.

.. [3] Greenfield, P., Droettboom, M., & Bray, E.,
   ASDF: A new data format for astronomy.
   Astronomy and Computing, 12: 240-251, 2015, doi:10.1016/j.ascom.2015.06.004

.. [4] Ouyang et al.,
   Deep learning massively accelerates super-resolution localization microscopy.
   Nat. Biotechnol. 2018, doi:10.1038/nbt.4106.

.. [5] https://github.com/imodpasteur/smlm-file-format

.. [6] Artur Speiser, Lucas-Raphael Müller, Philipp Hoess, Ulf Matti, Christopher J. Obara, Wesley R. Legant,
   Anna Kreshuk, Jakob H. Macke, Jonas Ries, and Srinivas C. Turaga,
   Deep learning enables fast and dense single-molecule localization with high accuracy.
   Nat Methods 18: 1082–1090, 2021, doi:10.1038/s41592-021-01236-x

.. [7] Ries, J.,
   SMAP: a modular super-resolution microscopy analysis platform for SMLM data.
   Nat Methods 17: 870–872, 2020, doi.org/10.1038/s41592-020-0938-1


Submodules:
-----------

.. autosummary::
   :toctree: ./

   io_locdata
   utilities
   rapidstorm_io
   thunderstorm_io
   elyra_io
   nanoimager_io
   asdf_io
   smlm_io
   decode_io
   smap_io

"""
from __future__ import annotations

from importlib import import_module

from .asdf_io import *
from .decode_io import *
from .elyra_io import *
from .io_locdata import *
from .nanoimager_io import *
from .rapidstorm_io import *
from .smap_io import *
from .smlm_io import *
from .thunderstorm_io import *
from .utilities import *

submodules: list[str] = [
    "asdf_io",
    "decode_io",
    "elyra_io",
    "io_locdata",
    "nanoimager_io",
    "rapidstorm_io",
    "smap_io",
    "smlm_io",
    "thunderstorm_io",
    "utilities",
]

__all__: list[str] = []

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="locan.locan_io.locdata")
    __all__.extend(module_.__all__)
