.. _directories:

===========================
Directories
===========================

TheProject is organized by the following directory structure:

::

    - Surepy

        - build
        - dist

        - docs

        - surepy

            - data
                - properties
                    - statistics.py
                    - ...

                - locdata.py
                - hulls.py
                - ...

            - analysis
                __init__: from localizations_per_frame import *

                - analysis.py
                - localizations_per_frame.py
                - k_distance_diagram.py
                - ...

            - io
            - render
            - simulation
            - gui

            - tests
