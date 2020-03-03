"""
Command-line utility scripts

The surepy package provides a command-line interface.
It is accessible by the base command ``surepy`` and provides a number of compound commands.
See available commands using ``surepy -h``.

This subpackage contains implementations of command-line scripts.
Scripts are installed in bin/ as simple wrappers for these modules.
They can be run directly from a terminal through ``surepy command options``
as long as the correct environment is activated.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   script_check
   script_rois
   script_draw_roi
   script_napari
   script_show_versions

"""
