"""

Transform localization intensities.

Localization intensities can be given in counts, electrons or photons.
This module provides often used transformation functions.

"""
from __future__ import annotations

import copy
import logging
import sys

import numpy as np
import numpy.typing as npt

from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta

__all__: list[str] = ["transform_counts_to_photons"]

logger = logging.getLogger(__name__)


def _transform_counts_to_photons(
    intensities: npt.ArrayLike,
    offset: int | float = 0,
    gain: int | float = 1,
    electrons_per_count: int | float = 1,
) -> npt.NDArray[np.float_]:
    """
    Convert camera analog-to-digital converter (ADC) counts into photo
    electrons.
    Quantum efficiency at the detected wavelength and collection efficiency
    of the optical system are not taken into account here.

    Parameters
    ----------
    intensities
        Intensity values in ADC counts.
    offset
        Camera offset
    gain
        Camera gain
    electrons_per_count
        Conversion factor for photo electrons per ADC count.

    Returns
    -------
    npt.NDArray[np.float_]
        Photo electrons
    """
    intensities = np.asarray(intensities)
    return (intensities - offset) * electrons_per_count / gain


def transform_counts_to_photons(
    locdata: LocData,
    loc_properties: str | list[str] | None = None,
    metadata: metadata_pb2.Camera | None = None,
) -> LocData:
    """
    Convert camera analog-to-digital converter (ADC) counts into photo
    electrons.
    Quantum efficiency at the detected wavelength and collection efficiency
    of the optical system are not taken into account here.

    Parameters
    ----------
    locdata
        Localization data on which to perform the manipulation.
    loc_properties
        Localization properties to be converted.
        If None the `intensity` values of locdata are used.
    metadata
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
    new_locdata = copy.copy(locdata)

    loc_properties_converted = []
    for loc_property in loc_properties:
        if loc_property not in locdata.data.columns:
            logger.warning(f"Localization property {loc_property} is not available.")
            continue

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
                continue

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

        assert new_locdata.dataframe is not None  # type narrowing # noqa: S101
        dataframe = new_locdata.dataframe.assign(**{loc_property: new_intensities})
        new_locdata = new_locdata.update(dataframe=dataframe)
        loc_properties_converted.append(loc_property)

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=new_locdata.meta,
    )
    new_locdata.meta = meta_

    logger.info(f"Successfully converted: {loc_properties_converted}.")
    return new_locdata
