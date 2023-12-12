import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import locan
from locan import ROOT_DIR, LocData
from locan.data import metadata_pb2
from locan.locan_io.locdata.io_locdata import load_txt_file
from locan.rois.roi import RoiLegacy_0

# fixtures


@pytest.fixture()
def locdata():
    locdata_dict = {
        "position_x": [0, 1, 2, 3, 0, 1, 4, 5],
        "position_y": [0, 1, 2, 3, 1, 4, 5, 1],
        "position_z": [0, 1, 2, 3, 4, 4, 4, 5],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def points():
    return np.array([[0, 0], [0.5, 0.5], [100, 100], [1, 1], [1.1, 0.9]])


# tests


def test_Roi_2d(locdata_2d):
    roi = RoiLegacy_0(
        reference=locdata_2d, region_specs=((0, 0), 2, 2, 0), region_type="rectangle"
    )
    new_dat = roi.locdata()
    assert len(new_dat) == 1
    assert new_dat.region is roi.region
    assert isinstance(roi.region, locan.RoiRegion)
    roi.region = locan.RoiRegion(
        region_specs=((0, 0), 2, 2, 0), region_type="rectangle"
    )
    assert isinstance(roi.region, locan.RoiRegion)

    roi = RoiLegacy_0(
        reference=locdata_2d,
        region_specs=((0, 0), 4, 4, 0),
        region_type="rectangle",
        properties_for_roi=("position_y", "frame"),
    )
    new_dat = roi.locdata()
    assert len(new_dat) == 2
    assert new_dat.region is None

    locdata_empty = LocData()
    roi_empty = RoiLegacy_0(
        reference=locdata_empty,
        region_specs=((0, 0), 2, 1, 10),
        region_type="rectangle",
    )
    empty_dat = roi_empty.locdata()
    assert len(empty_dat) == 0
    assert new_dat.region is None


def test_Roi_3d(locdata_3d):
    roi = RoiLegacy_0(
        reference=locdata_3d, region_specs=((0, 0), 2, 2, 0), region_type="rectangle"
    )
    new_dat = roi.locdata()
    assert len(new_dat) == 1
    assert new_dat.region is None
    assert isinstance(roi.region, locan.RoiRegion)
    roi.region = locan.RoiRegion(
        region_specs=((0, 0), 2, 1, 0), region_type="rectangle"
    )
    assert isinstance(roi.region, locan.RoiRegion)

    roi = RoiLegacy_0(
        reference=locdata_3d,
        region_specs=((0, 0), 2, 3, 0),
        region_type="rectangle",
        properties_for_roi=("position_y", "position_z"),
    )
    new_dat = roi.locdata()
    assert len(new_dat) == 1
    assert new_dat.region is None


def test_pickling_locdata_from_Roi(locdata_2d):
    import pickle

    roi = RoiLegacy_0(
        reference=locdata_2d, region_specs=((0, 0), 2, 1, 0), region_type="rectangle"
    )
    new_dat = roi.locdata()

    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / "pickled_locdata.pickle"
        with open(file_path, "wb") as file:
            pickle.dump(new_dat, file, pickle.HIGHEST_PROTOCOL)
        with open(file_path, "rb") as file:
            locdata = pickle.load(file)  # noqa S301
        assert len(new_dat) == len(locdata)
        assert isinstance(locdata.meta, metadata_pb2.Metadata)


def test_Roi_io(locdata):
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / "roi.yaml"
        # file_path = ROOT_DIR / 'tests/test_data/roi.yaml'

        roi = RoiLegacy_0(
            reference=locdata, region_type="rectangle", region_specs=((0, 0), 2, 1, 10)
        )
        with pytest.deprecated_call():
            roi.to_yaml(path=file_path)

        roi_new = RoiLegacy_0.from_yaml(path=file_path)
        assert roi_new.reference is None

        roi = RoiLegacy_0(
            reference=dict(
                file_path=ROOT_DIR / "tests/test_data/five_blobs.txt", file_type=1
            ),
            region_type="rectangle",
            region_specs=((0, 0), 2, 1, 10),
        )
        assert isinstance(roi.reference, (metadata_pb2.Metadata, Path))
        with pytest.deprecated_call():
            roi.to_yaml(path=file_path)

        roi_new = RoiLegacy_0.from_yaml(path=file_path)
        assert roi_new

        # test region specs with numpy floats
        roi = RoiLegacy_0(
            reference=dict(file_path=file_path, file_type=1),
            region_type="rectangle",
            region_specs=(np.array([0, 0], dtype=float), float(2), float(1), float(10)),
        )
        assert isinstance(roi.reference, metadata_pb2.Metadata)
        with pytest.deprecated_call():
            roi.to_yaml(path=file_path)

        roi_new = RoiLegacy_0.from_yaml(path=file_path)
        assert roi_new

        meta_ = metadata_pb2.Metadata()
        meta_.file.path = str(file_path)
        meta_.file.type = 1
        locdata_2 = LocData.from_selection(locdata, meta=meta_)
        roi = RoiLegacy_0(
            reference=locdata_2,
            region_type="rectangle",
            region_specs=((0, 0), 2, 1, 10),
        )
        assert isinstance(roi.reference.meta, metadata_pb2.Metadata)
        with pytest.deprecated_call():
            roi.to_yaml(path=file_path)

        roi_new = RoiLegacy_0.from_yaml(path=file_path)
        assert roi_new
        assert isinstance(roi_new.reference, metadata_pb2.Metadata)


def test_roi_locdata_from_file():
    locdata = load_txt_file(path=ROOT_DIR / "tests/test_data/five_blobs_3D.txt")

    roi = RoiLegacy_0(
        reference=locdata,
        region_type="rectangle",
        region_specs=((100, 500), 200, 200, 10),
    )
    dat = roi.locdata()
    assert len(dat) == 10

    roi = RoiLegacy_0(
        reference=locdata,
        region_type="rectangle",
        region_specs=((100, 100), 200, 200, 10),
        properties_for_roi=["position_x", "position_z"],
    )
    dat = roi.locdata()
    assert len(dat) == 10


def test_as_artist():
    roi_rectangle = RoiLegacy_0(
        reference=locdata, region_specs=((0, 0), 0.5, 0.5, 10), region_type="rectangle"
    )
    roi_ellipse = RoiLegacy_0(
        reference=locdata,
        region_specs=((0.5, 0.5), 0.2, 0.3, 10),
        region_type="ellipse",
    )
    roi_polygon = RoiLegacy_0(
        reference=locdata,
        region_specs=((0, 0.8), (0.1, 0.9), (0.6, 0.9), (0.7, 0.7), (0, 0.8)),
        region_type="polygon",
    )
    fig, ax = plt.subplots()
    ax.add_patch(roi_rectangle._region.as_artist())
    ax.add_patch(roi_ellipse._region.as_artist())
    ax.add_patch(roi_polygon._region.as_artist())
    # plt.show()

    plt.close("all")
