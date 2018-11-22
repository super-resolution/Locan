"""
Methods for batch processing.

Batch processing refers to running an analysis procedure or pipeline over a sequence of localization data objects.
"""

from pathlib import Path

from surepy import LocData
import surepy.io.io_locdata as io
from surepy.data.rois import Roi


def batch_process(elements, type='locdata', pipeline):
    """
    A batch process carrying out an analysis routine as specified in pipeline class.

    Parameters
    ----------
    elements : list of LocData or path or Roi objects
        Elements that refer to localization data to be processed serially. Path objects should point to
        localization files or roi.yaml files to be analyzed.
    type : str, int, list of str, list of int
        Indicator what type of data is specified by elements. Choose from one of the following: "locdata"
        for LocData objects, integer or indicator string for file type, "roi" for Roi object with valid file reference.
    pipeline : class name
        Class definition for a specific analysis pipeline

    Returns
    -------
    List of Pipeline objects
        Each pipeline object carries the various analysis results for each file/roi combination.
    """

    if isinstance(type, list):
        if len(elements)!= len(type):
            raise TypeError('The length of type as to be the same as the length of elements.')
        types = type
    else:
        types = [type] * len(elements)


    if isinstance(type, str):
        pass

    elif isinstance(type, int):
        raise NotImplementedError


    def run_pipe(element):
        '''
        Run pipeline
        '''

        pipe = pipeline(element)
        pipe.roi = 0
        pipe.compute()

        return pipe


    results = []
    path = None
    for element, _type in zip(elements, types):

        if _type == 'locdata':
            run_pipe(element)

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


