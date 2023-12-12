import logging
from copy import deepcopy

import numpy as np
import pytest

from locan import transform_counts_to_photons
from locan.data.transform.intensity_transformation import _transform_counts_to_photons


def test__transform_counts_to_photons():
    intensities = np.arange(10, 20)
    new_intensities = _transform_counts_to_photons(
        intensities=intensities, offset=10, gain=10, electrons_per_count=10
    )
    assert np.array_equal(new_intensities, np.arange(0, 10))

    intensities = list(range(10, 20))
    new_intensities = _transform_counts_to_photons(
        intensities=intensities, offset=10, gain=10, electrons_per_count=10
    )
    assert np.array_equal(new_intensities, np.arange(0, 10))


def test_transform_counts_to_photons(locdata_2d, caplog):
    caplog.set_level(logging.INFO)
    locdata = deepcopy(locdata_2d)
    df = locdata.dataframe.assign(local_background=[100] * len(locdata))
    locdata.update(dataframe=df)

    # metadata not available
    with pytest.raises(IndexError):
        transform_counts_to_photons(locdata=locdata)

    # provide camera metadata
    setup = locdata.meta.experiment.setups.add()
    ou = setup.optical_units.add()
    ou.detection.camera.offset = 10
    ou.detection.camera.gain = 10
    ou.detection.camera.electrons_per_count = 2

    # test
    new_locdata = transform_counts_to_photons(locdata=locdata)
    assert np.array_equal(
        new_locdata.data.intensity, [18.0, 28.0, 20.0, 14.0, 19.0, 17.0]
    )
    index = [prob.name for prob in new_locdata.meta.localization_properties].index(
        "intensity"
    )
    assert new_locdata.meta.localization_properties[index].unit == "photons"

    new_locdata = transform_counts_to_photons(
        locdata=locdata, loc_properties="local_background"
    )
    assert np.array_equal(new_locdata.data.local_background, [18.0] * len(new_locdata))
    index = [prob.name for prob in new_locdata.meta.localization_properties].index(
        "local_background"
    )
    assert new_locdata.meta.localization_properties[index].unit == "photons"

    new_locdata = transform_counts_to_photons(
        locdata=locdata, loc_properties=["intensity", "local_background"]
    )
    assert np.array_equal(
        new_locdata.data.intensity, [18.0, 28.0, 20.0, 14.0, 19.0, 17.0]
    )
    index = [prob.name for prob in new_locdata.meta.localization_properties].index(
        "intensity"
    )
    assert new_locdata.meta.localization_properties[index].unit == "photons"
    assert np.array_equal(new_locdata.data.local_background, [18.0] * len(new_locdata))
    index = [prob.name for prob in new_locdata.meta.localization_properties].index(
        "local_background"
    )
    assert new_locdata.meta.localization_properties[index].unit == "photons"

    new_locdata = transform_counts_to_photons(
        locdata=locdata,
        metadata=locdata.meta.experiment.setups[0].optical_units[0].detection.camera,
    )
    assert np.array_equal(
        new_locdata.data.intensity, [18.0, 28.0, 20.0, 14.0, 19.0, 17.0]
    )
    index = [prob.name for prob in new_locdata.meta.localization_properties].index(
        "intensity"
    )
    assert new_locdata.meta.localization_properties[index].unit == "photons"

    new_locdata = transform_counts_to_photons(
        locdata=locdata, loc_properties=["intensity", "not_present", "local_background"]
    )
    assert caplog.record_tuples[-2] == (
        "locan.data.transform.intensity_transformation",
        30,
        "Localization property not_present is not available.",
    )
    assert caplog.record_tuples[-1] == (
        "locan.data.transform.intensity_transformation",
        20,
        "Successfully converted: ['intensity', 'local_background'].",
    )

    new_locdata = transform_counts_to_photons(locdata=locdata)
    new_locdata_2 = transform_counts_to_photons(locdata=new_locdata)
    assert np.array_equal(new_locdata_2.data.intensity, new_locdata.data.intensity)
    assert caplog.record_tuples[-2] == (
        "locan.data.transform.intensity_transformation",
        30,
        "Localization property intensity is already provided with unit photons",
    )
