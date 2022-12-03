"""

Transform localization intensities.

Localization intensities can be given in counts, electrons or photons.
This module provides often used transformation functions.

"""
import logging
import sys

import numpy as np

from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta

__all__ = ["transform_counts_to_photons"]

logger = logging.getLogger(__name__)


def _transform_counts_to_photons(
    intensities, offset=0, gain=1, electrons_per_count=1
) -> np.ndarray:
    """
    Convert camera analog-to-digital converter (ADC) counts into photo
    electrons.
    Quantum efficiency at the detected wavelength and collection efficiency
    of the optical system are not taken into account here.

    Parameters
    ----------
    intensities : list | tuple | numpy.ndarray
        Intensity values in ADC counts.
    offset : int | float
        Camera offset
    gain : int | float
        Camera gain
    electrons_per_count : int | float
        Conversion factor for photo electrons per ADC count.

    Returns
    -------
    numpy.ndarray
        Photo electrons
    """
    intensities = np.asarray(intensities)
    return (intensities - offset) * electrons_per_count / gain


def transform_counts_to_photons(locdata, loc_properties=None, metadata=None) -> LocData:
    """
    Convert camera analog-to-digital converter (ADC) counts into photo
    electrons.
    Quantum efficiency at the detected wavelength and collection efficiency
    of the optical system are not taken into account here.

    Parameters
    ----------
    locdata : numpy.ndarray or LocData
        Localization data on which to perform the manipulation.
    loc_properties : str | list[str] | None
        Localization properties to be converted.
        If None the `intensity` values of locdata are used.
    metadata : locan.data.metadata_pb2.Camera
        Camera metadata with attribute offset, gain, electrons_per_count
        If None, locdata.meta.experiment.setups[0].optical_units[0].detection.camera
        is used.

    Returns
    -------
    LocData
        Localization data with converted intensity values.
    """
    local_parameter = locals()

    if not len(locdata):
        logger.warning("Locdata is empty.")
        return locdata

    if loc_properties is None:
        loc_properties = ["intensity"]
    elif isinstance(loc_properties, str):
        loc_properties = [loc_properties]

    # instantiate
    # todo: possibly add inplace keyword
    new_locdata = LocData.from_dataframe(dataframe=locdata.data)

    for loc_property in loc_properties:
        if loc_property not in locdata.data.columns:
            logger.warning(f"Localization property {loc_property} is not available.")
            break

        try:
            index = [prob.name for prob in locdata.meta.localization_properties].index(
                loc_property
            )
            unit = locdata.meta.localization_properties[index].unit
        except ValueError:
            index = None
            unit = None

        if index is not None:
            if locdata.meta.localization_properties[index].unit == "photons":
                logger.warning(
                    f"Localization property {loc_property} is already provided with "
                    f"unit photons"
                )
                break

        intensities = getattr(locdata.data, loc_property)

        if metadata is None:
            metadata = (
                locdata.meta.experiment.setups[0].optical_units[0].detection.camera
            )

        offset = metadata.offset
        gain = metadata.gain
        electrons_per_count = metadata.electrons_per_count

        new_intensities = _transform_counts_to_photons(
            intensities,
            offset=offset,
            gain=gain,
            electrons_per_count=electrons_per_count,
        )

        if index is None:
            new_locdata.meta.localization_properties.add()
            new_locdata.meta.localization_properties[-1].name = loc_property
            new_locdata.meta.localization_properties[-1].unit = "photons"
            new_locdata.meta.localization_properties[-1].type = "float"
        else:
            new_locdata.meta.localization_properties[index].unit = "photons"
            new_locdata.meta.localization_properties[index].type = "float"

        dataframe = new_locdata.dataframe.assign(**{loc_property: new_intensities})
        new_locdata = new_locdata.update(dataframe=dataframe)

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=new_locdata.meta,
    )
    new_locdata.meta = meta_

    return new_locdata