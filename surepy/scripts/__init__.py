"""
Command-line utility scripts

This subpackage contains implementations of command-line scripts that are used for certain Surepy tasks.
Scripts are installed in bin/ as simple wrappers for these modules and can be run directly from a terminal
as long as the correct environment is activated.

Surepy.scripts consists of the following modules:

.. autosummary::
   :toctree: ./

   draw_roi

"""
from .command_1 import command_1
from .draw_roi import draw_roi