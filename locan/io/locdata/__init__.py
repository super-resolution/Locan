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


Submodules:
-----------

.. autosummary::
   :toctree: ./

   io_locdata
   utilities
   rapidstorm
   smlm_file

"""
from .io_locdata import *
from .utilities import *
from .rapidstorm import *
from .smlm_file import *

__all__ = []
__all__.extend(io_locdata.__all__)
__all__.extend(utilities.__all__)
__all__.extend(rapidstorm.__all__)
__all__.extend(smlm_file.__all__)
