"""
This module provides methods for batch processing. Batch processing refers to running an analysis procedure or
pipeline over a sequence of localization data objects.
"""

from pathlib import Path

from surepy import LocData
import surepy.io.io_locdata as io
from surepy.data.rois import Roi


def batch_process(elements, pipeline):
    """
    A batch process carrying out an analysis routine as specified in pipeline class.

    Parameters
    ----------
    elements : list of LocData or path or Roi objects
        Elements that refer to localization data to be processed serially. Path objects should point to
        localization files or roi.yaml files to be analyzed.

    pipeline : class name
        Class definition for a specific analysis pipeline

    Returns
    -------
    List of Pipeline objects
        Each pipeline object carries the various analysis results for each file/roi combination.
    """

    results = []
    path = None

    for element in elements:

        if isinstance(element, LocData):
            print('Computing pipeline for locdata objects.')

            # run pipeline
            pipe = pipeline(element)
            pipe.roi = 0
            pipe.compute()
            results.append(pipe)
            print('Finished: {}  (roi: {}) '.format(pipe.identifier, pipe.roi))


        elif isinstance(element, str) or isinstance(element, Path):
            path = Path(element)

            if path.suffix == '.txt':
                print('Computing pipeline for localization-file path objects.')
                # Load data
                locdata = io.load_rapidSTORM_file(path)

                # run pipeline
                pipe = pipeline(locdata)
                pipe.roi = 0
                pipe.compute()
                results.append(pipe)
                print('Finished: {}  (roi: {}) '.format(pipe.identifier, pipe.roi))


            elif path.suffix == '.yaml':
                print('Computing pipeline for roi-file path objects.')
                # Load roi file
                roim = Roi_manager()
                roim.load(path)

                # Load data
                locdatas = roim.locdatas

                # Loop over selections
                for i, roi in enumerate(roim.rois):
                    # run pipeline
                    pipe = pipeline(locdatas[i])
                    pipe.identifier = roim.reference
                    pipe.roi = i
                    pipe.compute()
                    results.append(pipe)
                    print('Finished: {}  (roi: {}) '.format(pipe.identifier, pipe.roi))


        elif isinstance(element, Roi_manager):
            # Load data
            locdatas = element.locdatas

            # Loop over selections
            for i, roi in enumerate(element.rois):
                # run pipeline
                pipe = pipeline(locdatas[i])
                pipe.identifier = element.reference
                pipe.roi = i
                pipe.compute()
                results.append(pipe)
                print('Finished: {}  (roi: {}) '.format(pipe.identifier, pipe.roi))

    return results


